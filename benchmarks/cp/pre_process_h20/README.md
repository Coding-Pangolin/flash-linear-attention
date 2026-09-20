# CP 前处理标杆与 GPU 对齐

本目录放 `pre_process_fwd_kernel_merged` 的 **CPU 标杆**与**对齐工具**，配合
[`../README_pre_process_h20.md`](../README_pre_process_h20.md) 里的采集脚本使用。

目的：确认 CPU 标杆与上游 Triton kernel 算的是同一个东西。对齐通过后，标杆就成为
Ascend 算子精度验收的唯一依据。

| 文件 | 作用 |
| --- | --- |
| `reference.py` | 唯一可编辑的标杆源码：纯 PyTorch、设备无关、FP64 累加 |
| `compare_with_gpu.py` | 把 `--save-io` 导出的 `case.pt` 与标杆对齐 |
| `calibrate_reference.py` | 复现值域校准（量化契约舍入点的影响，为精度策略定阈值） |
| `precision-policy.json` | 已校准的精度策略（`hm` 是 FP32） |

## 1. 标杆的语义

输入按 token-major `[T, H, D]`；一次调用处理**一个窗口（一个 part）**；输出
`hm[HV, K, V+K]`，左 `[0,V)` 是 `h`，右 `[V, V+K)` 是 `m`。gate 是 **base-2 的
chunk 内累积对数衰减**（kernel 用 `exp2`）。

### 被刻意保留的三个舍入点

它们改变结果的程度远超容差，属于接口契约，标杆必须复刻：

1. `h` 进 `w @ h` 前降到输入 dtype（BF16）
2. `v_new` 进 `kᵀ @ v_new` 前降到输入 dtype（BF16）
3. `M_c @ m` 的链在 FP32 内累加，每个 chunk 更新后回落 FP32（上游 `input_precision="ieee"`）

其余部分用 FP64 累加，保证标杆精度高于实现。构造函数里有对应开关
（`round_h_to_input_dtype` / `round_v_new_to_input_dtype` /
`round_affine_chain_to_float32`），关掉即为“纯数学”版本，仅用于校准对照。

## 2. 在 GPU 机器上做对齐

```bash
# 1) 采集：一次拿到输入 + H20 的 hm
python -m benchmarks.cp.bench_pre_process_h20 --case model-gk --precision ieee \
    --warmup 5 --repeat 10 --save-io ./case_gk

# 2) 对齐（标杆默认跑在 CPU 上）
python benchmarks/cp/pre_process_h20/compare_with_gpu.py \
    --case ./case_gk/case.pt \
    --policy benchmarks/cp/pre_process_h20/precision-policy.json \
    --precision ieee
```

`--skill-dir` 是可选的：不传就只打印分区诊断表（下面那张表），传了
`cannbot-skills/ops/catlass-linear-attention-workflow` 才会再走一遍
工作流统一的 `compare_case()` 出正式报告。

### 关于 `--precision`（重要）

它填的是第 1 步采集时用的 `AFFINE_CHAIN_PRECISION`，只影响对 `m` 半边偏差的**预期**：

| 值 | 含义 |
| --- | --- |
| `ieee` | `m` 的链式乘按真 FP32，与标杆一致 → **严格对齐必须用这个** |
| `tf32x3` | 近似 FP32，偏差较小 |
| `default` | NVIDIA 上是 TF32（尾数 10 位）。`m` 是 176 次链式乘，误差会累积到 ~1e-1 相对量级，**必然超出策略** |

也就是说 `--precision default` 采出来的 `hm` 适合做**性能**基线，不适合做**正确性**对齐。
两者分开采：性能用模型真实精度，正确性用 `ieee`。

## 3. 怎么看结果

输出是一张分区表：

```
region        elements   matched     err     max_abs        MARE
ALL            1048576  1.000000       0   4.767e-07   5.952e-08
h_half          524288  1.000000       0   4.767e-07   5.952e-08
m_half          524288  1.000000       0   0.000e+00   0.000e+00
head0            32768  1.000000       0   4.649e-07   5.895e-08
head_last        32768  1.000000       0   4.607e-07   5.911e-08
```

**失败时的特征可以直接定位问题**（本地用注入误差的用例验证过）：

| 现象 | 含义 |
| --- | --- |
| `h_half` FAIL、`m_half` PASS | 标杆与上游在 **h 半边的舍入点**上没对齐（`bf16(h)` / `bf16(v_new)`），或 gate 语义不一致。绝对误差量级约 2e-2 |
| `h_half` PASS、`m_half` FAIL | `m` 的**链式乘精度**不够（低精度累积），或 `M_c` 的构造（对角 / 正负号）不对 |
| 两半都 FAIL | 输入布局、head 展开或窗口语义错了 |
| 全 PASS | 标杆与上游一致，可以进入 Ascend 实现 |

## 4. 复现值域校准

```bash
python benchmarks/cp/pre_process_h20/calibrate_reference.py --t 11264
```

打印 6 个变体（`HK=HV` / `HK=HV/2` × `g`/`gk`/`dplr`）的：

* 输出值域 `|h|max` / `|m|max`
* **契约舍入 vs 纯 FP64** 的最大偏差
* **独立实现 vs 标杆** 的最大偏差（即“复刻了舍入点、只是累加顺序不同”的正常偏差量级）

`precision-policy.json` 里的 `max_abs_limit` / `rtol` 就是按第 3 列定的：
模板默认的 `max_abs_limit=0.01` 会导致两个都正确的实现在互相比较时被判 FAIL，
实测最坏值 1.47e-2，故取 `0.1`（约 7 倍余量）；`rtol` 由 9.77e-4 提到 2e-3
（实测最坏 5.46e-4 的约 4 倍）。

## 5. 已知限制

* 标杆只覆盖**单窗口单 part**；跨 rank 的 `all_gather` 与 `merge_fwd_bwd_kernel` 不在范围内。
* 标杆默认在 CPU 上以 FP64 运行；`--device cuda` 可切到 GPU 做交叉核对，
  但正式验收以 CPU 为准。
* `hm` 视为**全区域有效**（算子输出完整的 `K×(V+K)`），不做无效区裁剪。
* 本算子没有结构性零区或单位区，因此不设 `critical_*` 分区，也不启用严格 ULP 检查。
