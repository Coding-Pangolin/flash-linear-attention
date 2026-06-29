# GPU GDN 算子 I/O Dump

用于采集 GPU 竞品（[flash-linear-attention](https://github.com/Coding-Pangolin/flash-linear-attention)）各 GDN 子算子的输入/输出，供 NPU 双标杆对比（GPU 升精度 + GPU 同精度）。

## 目录说明

本目录为 GPU 竞品代码副本（来源：`fla/ops/gated_delta_rule` 调用链）。若 GitHub 拉取失败，可从本地 `0605/flash-linear-attention` 同步：

```bash
rsync -a --exclude='.git' /path/to/flash-linear-attention/ gpu/
```

## 启用 Dump

设置环境变量后运行任意会调用 `chunk_gated_delta_rule` 的脚本/测试：

```bash
export GDN_DUMP_DIR=/path/to/dump_root
export GDN_DUMP_CASE=smoke_varlen_t256_v128   # 可选，默认 default
export GDN_DUMP_OPS=npu                       # 默认仅 NPU 仓 7 个算子；设 all 可采全部插桩
# export GDN_DUMP_EXIT=1                      # 命中 GDN_DUMP_OPS 后立即退出
# export GDN_DUMP_DTYPE=fp32                  # 统一保存为 float32
```

输出目录结构（默认 `GDN_DUMP_OPS=npu`）：

```
$GDN_DUMP_DIR/$GDN_DUMP_CASE/
  manifest.json
  001_recompute_wu.pt   # fwd 路径，meta.phase=fwd
  002_fwd_h.pt
  003_fwd_o.pt
  004_recompute_wu.pt   # bwd 路径，meta.phase=bwd
  005_bwd_dv_local.pt
  006_bwd_dhu.pt
  007_bwd_dqkwg.pt
  008_prepare_wy_repr_bwd.pt
  case_meta.json        # 批量脚本额外写入
```

每个 `.pt` 文件格式：

```python
{
  "op": "fwd_h",
  "step": 5,
  "inputs": {"k": Tensor, "w": Tensor, ...},
  "outputs": {"h": Tensor, "v_new": Tensor, ...},
  "meta": {"scale": 0.088, "cu_seqlens": [...], "chunk_indices": [...], ...},
}
```

## NPU 布局：使用时再转置（默认只存一份）

`.pt` **默认只存** GPU 布局 `inputs` / `outputs`（`[B,T,H,D]`），体积约为存两套的一半。

NPU 侧加载时再转：

```python
from fla.ops.gated_delta_rule.npu_layout import load_dump_for_npu

inp, meta, ref = load_dump_for_npu("006_bwd_dhu.pt", device="npu:0")
# 内部对 3D/4D 做 transpose(1,2)；h/dh 做 permute(0,2,1,3,4)；beta 转 fp32
```

手写等价于：

```python
x_npu = x_gpu.transpose(1, 2).contiguous()  # BTH* -> BHT*
```

若仍想在 dump 时预写 NPU 副本（占双倍空间），设 `export GDN_DUMP_NPU_LAYOUT=1`。

`beta` 在 `load_dump_for_npu` 中会转 **fp32**（NPU recompute 惯例）。

## GPU 不支持的 case 自动跳过

GPU `chunk_gated_delta_rule` 当前仅 **chunk_size=64** 可靠。`cases.json` 里 `chunk_size=128` 的 case **不做 GPU 双标杆**，批量脚本默认跳过（`--dry-run` 会标 `[SKIP/gpu]`）。

## NPU 对齐说明

GPU 存 **`[B,T,H,D]`**，NPU aclnn 要 **`[B,H,T,D]`**。`meta` 含 `scale`、`chunk_size`、`cu_seqlens`、`chunk_indices_npu`。

**喂 NPU 示例**（`bwd_dhu`）：

```python
from fla.ops.gated_delta_rule.npu_layout import load_dump_for_npu

inp, m, ref = load_dump_for_npu("006_bwd_dhu.pt", device="npu:0")
dh, _, dv2 = torch.ops.npu.npu_chunk_gated_delta_rule_bwd_dhu(
    inp["q"], inp["k"], inp["w"], inp["do"], inp["dv"],
    scale=m["scale"], chunk_size=m["chunk_size"], g=inp["g"],
    cu_seqlens=m.get("cu_seqlens"), chunk_indices=m.get("chunk_indices_npu"),
)
```

### 与 `flash_gated_delta_rule.py` 的差异（需注意）

| 项 | GPU dump / cases.json | NPU `flash_gated_delta_rule` |
|----|----------------------|------------------------------|
| q/k/v 布局 | 存 BTH；加载时 transpose | 原生 BHT |
| GVA | q/k 用 Hk，v/g/beta 用 Hv（原生 GVA） | 全链路透传前对 q/k `repeat_interleave` 到 Hv |
| gate 默认 | `negative_linear`（对齐 bwd_dhu 单测） | 默认 `logsigmoid`；可在 case 加 `"gate_function":"logsigmoid"` |
| scale | `K**-0.5`（与 flash 一致） | 同左 |
| 变长 cu_seqlens | `cases.json` 的 `mean_len` = cu_seqlens 长度 | `_build_mean_1k_cu_seqlens` 另一套生成逻辑 |

单算子双标杆：`load_dump_for_npu` 得到 BHT 输入，与 `ref`（GPU 标杆输出）对比。

## 采集点（与 NPU 单算子对齐）

默认仅 dump 以下 7 个算子（`GDN_DUMP_OPS=npu` 或未设置时）：

| 顺序 | dump op 名 | 对应 NPU 算子 |
|------|------------|---------------|
| 1 | `recompute_wu` | `recompute_w_u_fwd` |
| 2 | `fwd_h` | `chunk_gated_delta_rule_fwd_h` |
| 3 | `fwd_o` | `chunk_fwd_o` |
| 4 | `bwd_dv_local` | `chunk_bwd_dv_local` |
| 5 | `bwd_dhu` | `chunk_gated_delta_rule_bwd_dhu` |
| 6 | `bwd_dqkwg` | `chunk_bwd_dqkwg` |
| 7 | `prepare_wy_repr_bwd` | `prepare_wy_repr_bwd` |

完整 fwd+bwd 一次运行会产生 **8** 个 `.pt`（`recompute_wu` 在 fwd/bwd 各采一次，以 `meta.phase` 区分）。

## 批量采集（对齐 cases.json）

用例矩阵见 `gpu/cases.json`：

| 前缀 | 阶段 | 说明 |
|------|------|------|
| `fix_hk_eq_hv_*` / `var_hk_eq_hv_*` | legacy | 早期 HK==HV 冒烟 |
| `phase_1_*` | 一阶段 | 大 B/T 定长/变长泛化 |
| `gva_*` | 二阶段 | GVA（`query_head`≠`value_head`）矩阵 |

```bash
cd gpu
pip install -e .
chmod +x run_gdn_dump_cases.sh

# 二阶段 GVA（默认 --phase 2）
./run_gdn_dump_cases.sh --dump-dir /data/gdn_dump/gva --skip-done

# 一阶段
./run_gdn_dump_cases.sh --phase 1 --dump-dir /data/gdn_dump/phase1

# 指定用例（可配合 --include-disabled 跑 enabled=false 的大用例）
python3 scripts/run_gdn_dump_cases.py \
  --dump-dir /data/gdn_dump/gva \
  --names gva_fix_1,gva_var_1 \
  --include-disabled

# 仅列出将运行的用例
./run_gdn_dump_cases.sh --phase 2 --dry-run
```

每个 case 输出目录：`$GDN_DUMP_DIR/<case_name>/`，含 `manifest.json`、`NNN_<op>.pt` 及 `case_meta.json`（shape/seed/cu_seqlens 等）。汇总报告：`$GDN_DUMP_DIR/dump_report.json`。

环境变量（shell 包装脚本）：

- `GDN_DUMP_DIR`：输出根目录
- `GDN_DUMP_PHASE`：`all` | `1` | `2` | `legacy`
- `GDN_DUMP_OPS`：默认 `npu`（7 算子）；`all` 可恢复全部插桩
- `GDN_DUMP_DEVICE`：默认 `cuda:0`

## 单 case 示例

```bash
cd gpu
pip install -e .

export GDN_DUMP_DIR=/tmp/gdn_dump
export GDN_DUMP_CASE=smoke
export GDN_DUMP_OPS=fwd_h
export GDN_DUMP_EXIT=1

python - <<'PY'
import torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

B, T, H, HV, K, V = 1, 256, 16, 32, 128, 128
q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device='cuda')
g = torch.randn(B, T, HV, dtype=torch.float32, device='cuda')
beta = torch.sigmoid(torch.randn(B, T, HV, dtype=torch.bfloat16, device='cuda'))
o, _ = chunk_gated_delta_rule(q, k, v, g, beta)
o.sum().backward()
PY
```

加载 dump：

```python
import torch
d = torch.load("/tmp/gdn_dump/smoke/005_fwd_h.pt", map_location="cpu")
print(d["op"], d["outputs"].keys())
```
