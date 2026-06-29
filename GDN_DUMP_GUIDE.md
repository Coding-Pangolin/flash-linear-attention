# GDN GPU 采集 → NPU 双标杆 使用指南

本文档说明如何在 GPU 上批量采集 GDN 子算子 I/O，并在 NPU 上用同一份输入做双标杆对比。

对应分支：`feat/gdn-gpu-dump`（[flash-linear-attention](https://github.com/Coding-Pangolin/flash-linear-attention) fork）

---

## 1. 概述

| 项目 | 说明 |
|------|------|
| 目的 | 采集 GPU 竞品各子算子边界的 inputs/outputs，作为 NPU 精度标杆 |
| 对齐算子 | `recompute_wu` → `fwd_h` → `fwd_o` → `bwd_dv_local` → `bwd_dhu` → `bwd_dqkwg` → `prepare_wy_repr_bwd` |
| 用例矩阵 | `cases.json`（`phase_1_*` 一阶段，`gva_*` 二阶段 GVA） |
| 布局 | dump 存 GPU `[B,T,H,D]`；NPU 加载时 `transpose(1,2)` → `[B,H,T,D]` |
| GPU 限制 | 仅 `chunk_size=64` 的 case 参与 GPU 双标杆（128 自动跳过） |

---

## 2. GPU 环境准备

```bash
git clone https://github.com/Coding-Pangolin/flash-linear-attention.git
cd flash-linear-attention
git checkout feat/gdn-gpu-dump
pip install -e .
```

需要：CUDA GPU、PyTorch（CUDA 版）。

---

## 3. GPU 批量采集

### 3.1 快速开始

```bash
chmod +x run_gdn_dump_cases.sh

# 预览将运行的 case（chunk_size=128 显示 [SKIP/gpu]）
./run_gdn_dump_cases.sh --phase 2 --dry-run

# 二阶段 GVA 矩阵（默认 phase=2，含 enabled=false 的 case）
./run_gdn_dump_cases.sh --dump-dir /data/gdn_dump/gva --skip-done

# 一阶段 / legacy
./run_gdn_dump_cases.sh --phase 1 --dump-dir /data/gdn_dump/phase1
./run_gdn_dump_cases.sh --phase legacy --dump-dir /data/gdn_dump/legacy
```

等价 Python 入口：

```bash
python3 scripts/run_gdn_dump_cases.py \
  --dump-dir /data/gdn_dump/gva \
  --phase 2 \
  --include-disabled \
  --skip-done
```

### 3.2 常用参数

| 参数 / 环境变量 | 含义 |
|----------------|------|
| `--phase 2` / `gva` | 仅 `gva_*` case |
| `--phase 1` | 仅 `phase_1_*` |
| `--phase legacy` | `fix_hk_eq_hv_*` / `var_hk_eq_hv_*` |
| `--names a,b` | 指定 case 名 |
| `--skip-done` | 已有 `manifest.json` 则跳过 |
| `--dry-run` | 只列表不跑 |
| `GDN_DUMP_DIR` | 输出根目录 |
| `GDN_DUMP_OPS` | 默认 `npu`（7 算子） |
| `GDN_DUMP_NPU_LAYOUT=1` | 可选：dump 时额外存 NPU 布局（占双倍空间，一般不需要） |

### 3.3 输出目录

```
/data/gdn_dump/gva/
  gva_fix_1/
    case_meta.json      # B/T/Hk/Hv/K/V、seed、cu_seqlens、chunk_size
    manifest.json
    001_recompute_wu.pt # meta.phase=fwd
    002_fwd_h.pt
    003_fwd_o.pt
    004_recompute_wu.pt # meta.phase=bwd
    005_bwd_dv_local.pt
    006_bwd_dhu.pt
    007_bwd_dqkwg.pt
    008_prepare_wy_repr_bwd.pt
  dump_report.json
```

每个 `.pt` 结构：

```python
{
  "op": "bwd_dhu",
  "inputs": {...},    # GPU [B,T,H,D]，仅此一份
  "outputs": {...},
  "meta": {
    "scale": 0.088,
    "chunk_size": 64,
    "cu_seqlens": [0, 128, ...],
    "chunk_indices_npu": [0, 0, 0, 1, ...],
  },
}
```

### 3.4 传到 NPU 机器

```bash
rsync -av /data/gdn_dump/gva/ npu-host:/data/gdn_dump/gva/
```

---

## 4. NPU 侧执行与对比

### 4.1 加载 dump（自动转 NPU 布局）

```python
import torch
import torch_npu
import fla_npu  # noqa: F401

from fla.ops.gated_delta_rule.npu_layout import load_dump_for_npu

device = "npu:0"
torch.npu.set_device(0)

case_dir = "/data/gdn_dump/gva/gva_fix_1"
pt_path = f"{case_dir}/006_bwd_dhu.pt"  # 按 manifest 序号调整

inp, meta, ref = load_dump_for_npu(pt_path, device=device)
```

`load_dump_for_npu` 会：

- `inputs`/`outputs`：`[B,T,H,*]` → `[B,H,T,*]`（`transpose(1,2)`）
- `h`/`dh`：`[B,NT,H,K,V]` → `[B,H,NT,K,V]`（`permute`）
- `beta` → `float32`
- 若旧文件含 `inputs_npu`，则直接使用（兼容）

### 4.2 各算子调用示例

**bwd_dhu**

```python
dh, _, dv2 = torch.ops.npu.npu_chunk_gated_delta_rule_bwd_dhu(
    inp["q"], inp["k"], inp["w"], inp["do"], inp["dv"],
    scale=meta["scale"],
    chunk_size=meta["chunk_size"],
    g=inp["g"],
    gK=None, h0=inp.get("h0"), dht=inp.get("dht"),
    cu_seqlens=meta.get("cu_seqlens"),
    chunk_indices=meta.get("chunk_indices_npu"),
    use_exp2=False,
    transpose_state_layout=False,
)
```

**recompute_wu**（取 `meta.phase=bwd` 那份）

```python
w, u = torch.ops.npu.npu_recompute_w_u_fwd(
    inp["k"], inp["v"], inp["beta"], inp["A"],
    meta["chunk_size"], g=inp["g"], gk=None,
    cu_seqlens=meta.get("cu_seqlens"),
    chunk_indices=meta.get("chunk_indices_npu"),
)
```

**fwd_h / fwd_o / bwd_dv_local / bwd_dqkwg / prepare_wy_repr_bwd**：同样 `load_dump_for_npu` 后调对应 `torch.ops.npu.npu_*`，参数见各算子 README。

### 4.3 精度对比

```python
def compare(name, npu_out, gpu_ref, atol=1e-2, rtol=1e-2):
    a, b = npu_out.float().cpu(), gpu_ref.float().cpu()
    ok = torch.allclose(a, b, atol=atol, rtol=rtol)
    err = (a - b).abs().max().item()
    print(f"{name}: max_err={err:.6f} allclose={ok}")
    return ok

compare("dh", dh, ref["dh"])
compare("dv2", dv2, ref["dv2"])
```

- `ref` 来自 GPU 同精度输出 → **同精度双标杆**
- 也可将 `ref` 先转 `float64` 作升精度标杆

---

## 5. 注意事项

### GVA（Hk ≠ Hv）

- dump / NPU 单算子路径：**q/k 为 Hk，w/g/do/dv 为 Hv**（原生 GVA）
- `examples/flash_gated_delta_rule.py` **全链路**会对 q/k `repeat_interleave` 到 Hv，与单算子 dump **不等价**

### gate / scale

- cases.json 默认 `negative_linear` gate（对齐 bwd_dhu 单测）
- flash Example 默认 `logsigmoid`；可在 case 中加 `"gate_function": "logsigmoid"`
- `scale = K**-0.5`

### chunk_size

| chunk_size | GPU 双标杆 | NPU 单测 |
|------------|-----------|---------|
| 64 | ✅ 采集 | ✅ |
| 128 | ❌ 自动跳过 | ✅ 单独测 |

---

## 6. 相关文件

| 文件 | 作用 |
|------|------|
| `run_gdn_dump_cases.sh` | 批量采集 shell 入口 |
| `scripts/run_gdn_dump_cases.py` | 批量采集 Python 实现 |
| `scripts/gdn_case_utils.py` | cases.json 解析、输入生成 |
| `cases.json` | 用例矩阵 |
| `fla/ops/gated_delta_rule/dump.py` | dump 插桩 |
| `fla/ops/gated_delta_rule/npu_layout.py` | `load_dump_for_npu` |
| `README_GDN_DUMP.md` | 插桩与环境变量说明 |
