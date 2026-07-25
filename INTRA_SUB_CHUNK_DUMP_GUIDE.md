# Intra Sub-Chunk GPU 采集 → NPU 双标杆 使用指南

在 GPU 上批量采集 `chunk_kda_fwd_kernel_intra_sub_chunk` 的输入/输出，供 Ascend NPU `npu_chunk_kda_fwd_intra_sub_chunk` 做双标杆对比。

对应分支：`20260724_222750_intra-sub-chunk-gpu-dump`  
参考实现：`feat/kda-gpu-dump` / `feat/gdn-gpu-dump` / `20260721_193023_chunk-kda-fwd-intra-sub-chunk-gpu-cpu-dual`

---

## 1. 概述

| 项目 | 说明 |
|------|------|
| 算子 | Triton `chunk_kda_fwd_kernel_intra_sub_chunk`（`fla/ops/kda/chunk_intra.py`） |
| CPU 标杆 | `tests/ops/chunk_kda_fwd_intra_sub_chunk_ref.py`（**同一份 GPU 输入**） |
| 用例矩阵 | `intra_sub_chunk_cases.json`（smoke + GDN 泛化表代表 case） |
| 布局 | dump / 生成：`BTHD`；NPU：`transpose(1,2)` → `BNSD` |
| 随机数 | **默认 CPU RNG**（`rng_on_cpu=True`），再 `.to(cuda)`；与 NPU seed-dual 同 seed 对齐 |
| 种子 | `seed_i = --seed + case_index * 9973`（`case_index` = 过滤后列表下标） |
| 默认配置 | `gate=lin_mild`，`l2norm=True`，`dtype=bf16`；CPU golden 默认 `fp32` |
| GPU 限制 | 仅 `chunk_size ∈ {32,64}`；`cs=128` 自动跳过 |

输出（同一份 inputs）：

- GPU：`aqk` `[B,T,HV,BT]`，`akkd` `[B,T,HV,BC]`（fp32）
- CPU：`aqk_cpu` / `akkd_cpu`（同布局；另有独立 `002_*_cpu.pt`）

---

## 2. GPU 环境准备

在有 CUDA 的机器上：

```bash
cd /path/to/flash-linear-attention
git fetch pangolin   # 或你们的 remote
git checkout 20260724_222750_intra-sub-chunk-gpu-dump
pip install -e .
```

需要：CUDA GPU、PyTorch（CUDA 版）、Triton（随 fla 依赖）。

本机若无 CUDA，脚本会在真正跑 dump 时退出；`--dry-run` 可在无 GPU 时预览 case。

---

## 3. 批量采集

```bash
chmod +x run_intra_sub_chunk_dump_cases.sh

# 预览（含 GPU 不支持的 cs=128 SKIP）
./run_intra_sub_chunk_dump_cases.sh --dry-run

# 冒烟
./run_intra_sub_chunk_dump_cases.sh --phase smoke \
  --dump-dir /data/isub_dump/smoke --skip-done

# 全部可跑 case
./run_intra_sub_chunk_dump_cases.sh \
  --dump-dir /data/isub_dump/gdn --skip-done

# 指定 case
./run_intra_sub_chunk_dump_cases.sh \
  --names BSND_noGVA_V128_14,BSND_GVA_V256_28 \
  --dump-dir /data/isub_dump/pick
```

等价 Python：

```bash
python3 scripts/run_intra_sub_chunk_dump_cases.py \
  --dump-dir /data/isub_dump/gdn \
  --phase smoke \
  --skip-done
```

### 常用参数

| 参数 | 含义 |
|------|------|
| `--phase smoke` | 仅 `smoke_*` |
| `--phase gdn` | 仅 `BSND_*` |
| `--phase varlen` / `gva` | 变长 / GVA |
| `--names a,b` | 指定 case |
| `--skip-done` | 已有 `manifest.json` 则跳过 |
| `--dtype-save fp32` | 浮点统一存 fp32（体积更大） |
| `--cpu-dtype fp32\|fp64` | CPU 标杆计算精度（默认 fp32） |
| `--no-cpu` | 只 dump GPU，不跑 CPU |
| `--seed N` | 基种子；第 i 个 case 用 `N + i*9973` |

变长 `cu_seqlens` 用与 NPU `prec_gdn_isub` 相同的随机未对齐构造，并写入 `case_meta.json`，便于 NPU 复现。

---

## 4. 输出目录

```
/data/isub_dump/gdn/
  smoke_mha_fix/
    case_meta.json
    manifest.json
    001_chunk_kda_fwd_intra_sub_chunk.pt      # inputs + GPU + CPU
    002_chunk_kda_fwd_intra_sub_chunk_cpu.pt  # 同 inputs，outputs 仅 CPU
  BSND_noGVA_V128_14/
    ...
  intra_sub_chunk_dump_report.json
```

`001_*.pt`：

```python
{
  "op": "chunk_kda_fwd_intra_sub_chunk",
  "step": 1,
  "layout": {"storage": "BTHD", ...},
  "inputs": {
    "q", "k", "g", "beta", "scale",
    "cu_seqlens", "chunk_indices", "chunk_size"
  },
  "outputs": {"aqk", "akkd", "aqk_cpu", "akkd_cpu"},
  "meta": {B, T, H, HV, K, seed, cpu_dtype, ...}
}
```

---

## 5. NPU 侧加载提示

```python
import torch

payload = torch.load("001_chunk_kda_fwd_intra_sub_chunk.pt", map_location="cpu", weights_only=False)
inp, out = payload["inputs"], payload["outputs"]

# BTHD → BNSD
q = inp["q"].transpose(1, 2).contiguous()       # [B,H,T,K]
k = inp["k"].transpose(1, 2).contiguous()
g = inp["g"].transpose(1, 2).contiguous()       # [B,HV,T,K]
beta = inp["beta"].transpose(1, 2).contiguous() # [B,HV,T]
aqk_gpu = out["aqk"].transpose(1, 2).contiguous()
akkd_gpu = out["akkd"].transpose(1, 2).contiguous()
aqk_cpu = out["aqk_cpu"].transpose(1, 2).contiguous()
akkd_cpu = out["akkd_cpu"].transpose(1, 2).contiguous()
```

用同一份 `inputs` 调 NPU，可同时对标 GPU（`aqk`/`akkd`）与 CPU（`aqk_cpu`/`akkd_cpu`）。

---

## 6. 与现有 dual 测试的关系

同仓还有：

```bash
python tests/ops/test_chunk_kda_fwd_intra_sub_chunk_gpu_cpu_dual.py --smoke
```

那是 **GPU Triton vs CPU golden** 的即时对比；本 dump 流程是 **落盘 GPU I/O**，便于拷到 NPU 服务器做离线双标杆。

注意：dual 脚本里曾写死 MHA-only；dump runner 已支持 GVA（`HV` 参数）。若 dual 测 GVA，需用本仓带 `HV` 的 kernel（`pip install -e .`）。

---

## 7. 拷贝到 NPU 服务器

```bash
# GPU 机
tar czf isub_gpu_dump.tgz -C /data/isub_dump gdn

# NPU 机
scp ... isub_gpu_dump.tgz
tar xzf isub_gpu_dump.tgz
```

建议先跑 `--phase smoke` 验证通路，再跑完整 GDN 矩阵。
