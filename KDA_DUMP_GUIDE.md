# KDA GPU 采集 → NPU 双标杆 使用指南

在 GPU 上批量采集 `chunk_kda` 前向算子 I/O，供 NPU Ascend KDA 实现做精度对标。

对应分支：`feat/kda-gpu-dump`（[flash-linear-attention](https://github.com/Coding-Pangolin/flash-linear-attention) fork）

---

## 1. 概述

| 项目 | 说明 |
|------|------|
| 算子 | `fla.ops.kda.chunk_kda`（`chunk.py` → `chunk_kda_fwd`） |
| 用例矩阵 | `kda_cases.json`（8 个典型 MHA/GVA 定长/变长 shape） |
| 布局 | GPU `[B,T,H/HV,D]`；dump 存 CPU tensor，meta 标注 `BTHD` |
| 默认 kernel 配置 | `use_qk_l2norm_in_kernel=True`, `use_gate_in_kernel=True`, `use_beta_sigmoid_in_kernel=True`, `safe_gate=True`, `lower_bound=-5` |

---

## 2. GPU 环境准备

```bash
git clone https://github.com/Coding-Pangolin/flash-linear-attention.git
cd flash-linear-attention
git checkout feat/kda-gpu-dump
pip install -e .
```

需要：CUDA GPU、PyTorch（CUDA 版）。

---

## 3. 批量采集

### 3.1 快速开始

```bash
chmod +x run_kda_dump_cases.sh

# 预览
./run_kda_dump_cases.sh --dry-run

# 冒烟 3 个 case
./run_kda_dump_cases.sh --phase smoke --dump-dir /data/kda_dump/smoke --skip-done

# 全部 8 个 case
./run_kda_dump_cases.sh --dump-dir /data/kda_dump/all --skip-done

# 指定 case
./run_kda_dump_cases.sh --names gva_t4096_v256,smoke_gva_fix --dump-dir /data/kda_dump/pick
```

等价 Python 入口：

```bash
python3 scripts/run_kda_dump_cases.py \
  --dump-dir /data/kda_dump/all \
  --skip-done
```

### 3.2 单 case 手动 dump

```bash
export KDA_DUMP_DIR=/tmp/kda_dump
export KDA_DUMP_CASE=manual
python3 - <<'PY'
import torch
from fla.ops.kda import chunk_kda

B, T, H, HV, K, V = 1, 256, 2, 4, 128, 128
dev = "cuda:0"
q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=dev)
k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=dev)
v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device=dev)
g = torch.randn(B, T, HV, K, dtype=torch.bfloat16, device=dev)
beta = torch.randn(B, T, HV, dtype=torch.bfloat16, device=dev)
A_log = torch.log(torch.empty(HV, device=dev).uniform_(1, 16))
dt_bias = torch.randn(HV * K, dtype=torch.float32, device=dev)
h0 = torch.randn(B, HV, K, V, dtype=torch.float32, device=dev)

with torch.inference_mode():
    o, ht = chunk_kda(
        q, k, v, g, beta,
        A_log=A_log, dt_bias=dt_bias,
        initial_state=h0, output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True, lower_bound=-5.0,
    )
print("o", o.shape, "ht", ht.shape if ht is not None else None)
PY
```

### 3.3 输出目录

```
/data/kda_dump/all/
  smoke_mha_fix/
    case_meta.json
    manifest.json
    001_chunk_kda_fwd.pt
  gva_t4096_v256/
    ...
  kda_dump_report.json
```

每个 `.pt` 结构：

```python
{
  "op": "chunk_kda_fwd",
  "step": 1,
  "layout": {"storage": "BTHD", ...},
  "inputs": {
    "q", "k", "v", "g", "beta", "A_log", "dt_bias",
    "initial_state", "scale", "cu_seqlens", "chunk_indices"
  },
  "outputs": {"o", "final_state"},
  "meta": {kernel flags, chunk_size, cu_seqlens list, ...}
}
```

---

## 4. 用例矩阵（kda_cases.json）

| name | 类型 | B | T | Hk | Hv | K | V | cs | varlen |
|------|------|---|---|----|----|---|---|-----|--------|
| smoke_mha_fix | MHA 冒烟 | 1 | 256 | 2 | 2 | 128 | 128 | 64 | |
| smoke_gva_fix | GVA 冒烟 | 1 | 256 | 2 | 4 | 128 | 128 | 64 | |
| smoke_mha_var | MHA 变长 | 1 | 512 | 4 | 4 | 128 | 128 | 64 | ✓ |
| mha_t2048 | MHA 中长 | 2 | 2048 | 4 | 4 | 128 | 128 | 64 | |
| gva_t4096_v256 | GVA 大模型 | 1 | 4096 | 16 | 32 | 128 | 256 | 64 | |
| gva_t4096_v128 | GVA 长序列 | 1 | 4096 | 8 | 16 | 128 | 128 | 64 | |
| mha_cs32 | cs=32 | 1 | 512 | 4 | 4 | 128 | 128 | 32 | |
| gva_var_t1024 | GVA 变长 | 1 | 1024 | 4 | 8 | 128 | 128 | 64 | ✓ |

---

## 5. 同步到 NPU 机

```bash
rsync -av /data/kda_dump/all/ npu_host:/data/kda_dump/all/
```

NPU 侧后续可参照 GDN 的 `gpu_dump_loader.py` 模式，读取 `.pt` 做 `ct.dual(npu_out, fp64_gt, gpu_bench)`。

---

## 6. 环境变量

| 变量 | 默认 | 含义 |
|------|------|------|
| `KDA_DUMP_DIR` | (unset) | 启用 dump 的根目录 |
| `KDA_DUMP_CASE` | `default` | 子目录名 |
| `KDA_DUMP_OPS` | `chunk_kda_fwd` | 算子过滤 |
| `KDA_DUMP_DTYPE` | native | 设 `fp32` 统一浮点精度 |
| `KDA_DUMP_EXIT` | `0` | `1` = dump 后 exit |
