# CP 前处理 kernel 的 H20 基线采集

采集对象（上游实现）：
[`fla/ops/cp/chunk_delta_h.py::pre_process_fwd_kernel_merged`](../../fla/ops/cp/chunk_delta_h.py)

采集脚本：[`bench_pre_process_h20.py`](bench_pre_process_h20.py)

## 1. 这个 kernel 做什么

它是上下文并行（CP, context parallel）场景下 `chunk_gated_delta_rule_fwd_h` 的**前处理**：
为某个 rank 的**一个序列窗口（一个 part）**计算两样东西，打包写进 `hm`：

| 输出 | 位置 | 含义 |
| --- | --- | --- |
| `h` | `hm[:, :, 0:V]` | 窗口内从零状态起累积出来的边界状态（K×V） |
| `m` | `hm[:, :, V:V+K]` | 窗口的仿射链 `m = Π(diag(decay) ∓ kᵀw)`，即单位阵起累乘（K×K） |

这两个量合起来把窗口表达成一个仿射变换 `S_out = m @ S_in + h`，于是各 rank 之间可以靠
「前缀复合」拼出每个 rank 真正的 `initial_state`，而不必把整条序列串起来扫。
跨 rank 的 `all_gather` 与 `merge_fwd_bwd_kernel` **不属于本 kernel**。

调用方：`fla/ops/kda/chunk_fwd.py`（gk）、`fla/ops/gated_delta_rule/chunk.py`（g）、
`fla/ops/generalized_delta_rule/dplr/chunk.py`（gk + bg）。

## 2. 为什么要单独采这一个 kernel

NPU 侧要写一个同名算子对标它，验收口径被定为「**1.0 倍 H20**」，且 NPU 侧的判定指标是
`msprof` `op_summary` 的 `Task Duration(us)`——那是**纯 device 上的算子执行时间**。
所以 H20 侧也必须采**纯 kernel 时长**，两边才可比。

## 3. 采集口径（重要）

| 方式 | 量到的是什么 | 用在哪 |
| --- | --- | --- |
| 脚本默认（CUDA event） | 流上事件间隔，**含 host launch 空隙** | 交叉验证，不能当基线 |
| `nsys` | GPU 自己的 kernel 时间戳 → **纯 kernel 时长** | **基线主数** |
| `--torch-profiler`（CUPTI） | 同上，纯 device 时长，不需要外部工具 | 基线主数的替代/复核 |
| `ncu` | 单 kernel 指标与时长 | **只做瓶颈分析，不能当基线**（会串行化 + replay，时长失真） |

event 法会把 H20 的数**报大**（triton 的 launch 开销约 10–30 µs 也被算进去），
方向上是「让 NPU 更容易达标」，所以不能采用。

脚本默认开 `--nvtx`，会在测量区间外套一层 NVTX range，方便 nsys 用
`nvtx_gpu_proj_sum` 报告把 warmup 和 autotune 的 launch 全部滤掉。

## 4. 前置

```bash
git clone https://github.com/fla-org/flash-linear-attention.git
cd flash-linear-attention
pip install -e .
# 需要 NVIDIA 版 torch + triton，以及 nsys（Nsight Systems）
```

## 5. 快速开始

```bash
# 看内置 case
python -m benchmarks.cp.bench_pre_process_h20 --list

# 三条算法路径各测一次
python -m benchmarks.cp.bench_pre_process_h20 --case model-g      # GDN（g，HK<HV）
python -m benchmarks.cp.bench_pre_process_h20 --case model-gk     # KDA（gk）
python -m benchmarks.cp.bench_pre_process_h20 --case model-dplr   # DPLR（gk+bg）
```

输出末尾有一段 `summary`，格式统一，直接回贴即可。

## 6. `T` 是窗口长度，不是整条序列

CP 场景下这个 kernel **一次只处理一个 rank 的一个窗口（一个 part）**。所以：

* 模型 case 是 `T_total=11264`、CP 用 `W` 张卡时，每个 rank 的窗口约 `T_total / W`
  （zigzag 布局下每个 rank 持有 front/back 两个窗口）；
* `--t` 要填**窗口长度**，不是整条序列；
* 默认给的是 `11264`，等价于 `world_size = 1` 的非 CP 情形。

```bash
python -m benchmarks.cp.bench_pre_process_h20 --case model-gk --t 2816
```

其它维度（`--hk --hv --k --v --bt`）显式给值时会覆盖 case 预设。

## 7. 三种 `AFFINE_CHAIN_PRECISION`

`m` 的链式乘 `M_c @ m` 上游默认按 `ieee`（真 FP32）走，NVIDIA 上可切 `tf32x3`。
脚本默认把三种都测一遍：

| 值 | 含义 |
| --- | --- |
| `default` | Triton 对 fp32 dot 的默认；NVIDIA 上是 **tf32** |
| `tf32x3` | 模型开 `use_tf32x3_affine_chain` 时走的路，精度接近 fp32 |
| `ieee` | 真 fp32；H20 上会明显更慢 |

知道模型实际用哪个就加 `--precision` 只测那一个：

```bash
python -m benchmarks.cp.bench_pre_process_h20 --case model-gk --precision tf32x3
```

## 8. 采纯 kernel 时长

```bash
# 主数：nsys
nsys profile -o prof_gk --force-overwrite=true --trace=cuda,nvtx \
    python -m benchmarks.cp.bench_pre_process_h20 --case model-gk \
    --warmup 20 --repeat 100
nsys stats --report nvtx_gpu_proj_sum,cuda_gpu_kern_sum prof_gk.nsys-rep

# 复核：CUPTI（不用装 nsys）
python -m benchmarks.cp.bench_pre_process_h20 --case model-gk \
    --warmup 20 --repeat 100 --torch-profiler
```

## 9. 导出可复现用例

```bash
python -m benchmarks.cp.bench_pre_process_h20 --case model-gk --save-io ./case_gk
```

会落盘：

* `case.pt`：该 case 的**全部输入 + 参考输出 `hm`**（CPU 张量字典）；
* `case.json`：shape、seed、grid、MAC 估算、耗时结果、torch/triton 版本。

有这份数据，NPU 侧就能用**完全相同的输入**跑自己的算子，直接和 H20 的 `hm` 逐元素比，
功能对标和性能对标一次做完。

## 10. 语义约定（与上游 kernel 一致，容易采错的三点）

1. **布局是 token-major `[T, H, D]`**（不是 `[B, H, T, D]`）。
2. **`cu_seqlens` 传 `[0, T]`**。上游 CP 路径恒走 varlen 分支
   （`cu_win = cu_seqlens[fns-1:fns+1]` 是个 2 元素切片），此时 `T` 参数会被 `eos - bos`
   覆盖。传 `None` 会跑到定长分支去，不是模型实际执行的代码。想对比可加 `--no-cu-seqlens`。
3. **gate 是 base-2 的 chunk 内累积对数衰减**，kernel 内部用 `exp2`。

启动参数与上游 CP 包裹完全一致：

```python
grid = (triton.cdiv(V, BS) + triton.cdiv(K, BS), HV)   # BS = 32 if K <= 64 else 64
BK1  = triton.next_power_of_2(K)
kernel[grid](k=, v=, w=, g=, gk=, bg=, u=, hm=, cu_seqlens=, T=,
             H=, HV=, K=, V=, BT=, BK1=, BLOCK_SIZE=BS,
             MULTI_SEQS=False, AFFINE_CHAIN_PRECISION=...)
```

`hm` 形状 `[HV, K, V+K]`，FP32。`h` 在进 dot 前降到 BF16，`v_new` 也降到 BF16，
`m` 的链在 FP32 内累加——这三个舍入点是精度对齐的关键。

## 11. 输入生成与发散（首次运行踩过的坑）

脚本按 delta 规则的语义构造输入，而不是用独立随机张量：

* `k` 沿 head 维单位化；
* `w = beta * k`，`beta ~ U(0, --beta-scale)`（默认 `0.02`）；
* `bg = gamma * k`（仅 DPLR），`gamma ~ U(0, --bg-scale)`（默认 `0.02`）；
* gate 按 chunk 累积，每 chunk 总 log2 衰减 `--decay-per-chunk`（默认 `0.013`，
  即 chunk 内衰减率 `exp2(-0.013) ≈ 0.991`，整窗口 176 个 chunk 后累计约 `0.2`）。

**为什么不能直接用独立随机的 `w`**：状态更新是

```
h ← decay·h + kᵀ(v − w h) = (decay·I − kᵀw)·h + kᵀv
```

`w` 若与 `k` 无关地随机采样，`kᵀw` 的谱范数会远大于 1，递推在几十个 chunk 内就发散成
`NaN`（脚本第一版正是如此，输出 `finite=False`）。

发散时**计时结果仍然可用**：这个 kernel 没有数据相关分支，NaN 不改变 tensor core 吞吐。
但 `hm` 不能当参考输出——脚本在输出非有限时会给出警告，并且**不写 `case.pt`**。

参数不合适时的调整方向：调小 `--beta-scale`（更收缩），或调大 `--decay-per-chunk`
（衰减更快）。

## 12. 已知限制

* **不走 `chunk_gated_delta_rule_fwd_h_pre_process` 包裹**：本脚本只量单次 kernel，
  不含 `all_gather_into_tensor` 与 `merge_fwd_bwd_kernel`。若目标口径是「一个 rank 的
  整个 pre_process（front/back 两次 kernel + 通信 + merge）」，需要另外的测量。
* 单进程，不初始化 `torch.distributed`。
* 会触发 `@triton.autotune`（6 个 config），第一次调用较慢；脚本的 warmup 已覆盖，
  `summary` 报的是 autotune 之后的稳态值。

## 13. 回贴格式

```
case=model-gk variant=gk T=11264 HK=32 HV=32 K=128 V=128 BT=64 prec=default
      p50=xxx.xus p90=xxx.xus min=xxx.xus
nsys   nvtx_gpu_proj_sum: pre_process_fwd_kernel_merged xxx.x us (n=100)
cupti  pre_process_fwd_kernel_merged avg xxx.x us
device=NVIDIA H20 torch=... triton=...
```

三个数互相印证；如果 nsys 与 event 法差出几倍，通常是 autotune 还没收敛或有别的
kernel 混进来了，需要重采。
