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

### 契约的数值精度（四个点）

它们改变结果的程度远超容差，属于接口契约，标杆必须复刻：

1. **累加精度 FP32** —— 上游的 `h`/`m` 累加器都是 `tl.zeros(..., dtype=tl.float32)`，
   所有 `tl.dot` 都往 FP32 累加器累加，所以标杆默认 `accum_dtype=torch.float32`
2. `h` 进 `w @ h` 前降到输入 dtype（BF16）——`round_h_to_input_dtype`
3. `v_new` 进 `kᵀ @ v_new` 前降到输入 dtype（BF16）——`round_v_new_to_input_dtype`
4. `M_c @ m` 每个 chunk 更新后回落 FP32（上游 `input_precision="ieee"`）
   ——`round_affine_chain_to_float32`

把 `accum_dtype` 换成 `float64`、三个开关全关，就得到“纯数学”版本，仅用于灵敏度对照。

> 第 1 条是 2026-09-20 与 H20 实测对齐时才发现的：标杆原本用 FP64 累加，与 H20 `ieee`
> 在 h 半边差 9.371e-03；单独把累加器换成 FP32（其余完全相同）就产生 8.735e-03。
> **用 FP64 基准会让任何忠实的 FP32 实现都平白多出约 9e-3 的绝对偏差。**

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

[严格复核] m_half 用 atol=1e-6 / rtol=0.002 复算: matched=1.000000 max_abs=0.000e+00
```

### h 半边是弱检查，m 半边是强检查

由算法本身决定，**h 半边天生带约 8e-3 的绝对噪声底**：`bf16(h)` 把状态量化成 bf16，
而量化是**不连续**的 —— 两个实现只要在 h 上差一点点，就可能在量化边界处翻转整整一个
bf16 ulp，再经反馈环放大。实测（把 token 累加按 16 分组，模拟 NPU 的 Cube tile 累加）：

| 对照（同为 FP32） | h_half | m_half |
| --- | --- | --- |
| tile=16 | abs 7.213e-3 / rel 4.01e-4 | **0** |
| tile=8 | abs 8.029e-3 / rel 4.47e-4 | **0** |

所以：

* **h 半边的阈值必然偏松**（策略 `atol=1.5e-2`，相对 |h|≈18 只有 8e-4）。副作用是
  “没做 `bf16(h)` 转换”这类错误只产生 1.05e-2 偏差，落在接受带附近 —— 单靠 h 半边
  无法可靠区分。
* **m 半边对累加顺序完全免疫（差异恒为 0）**，阈值可以收到 `atol=1e-6`。它覆盖了
  k/w 读取、head 展开、gate 对角构造、正负号约定和链式累加，是**最强的正确性证据**。
  脚本每次都会额外打印这一行“严格复核”。
* 真正约束 h 半边的要靠**短窗口用例**（`NT` 小，反馈放大尚未累积）与结构检查，
  这部分在 05 阶段补。

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

打印两组数据：

* 6 个变体（`HK=HV` / `HK=HV/2` × `g`/`gk`/`dplr`）的 `|h|max` / `|m|max` 与
  **契约 vs 纯 FP64** 偏差（实测 3.0e-2 ~ 4.6e-2 绝对）
* **同为 FP32、仅累加顺序不同**的偏差（实测 h 半边 7.2e-3 ~ 8.0e-3 绝对，
  m 半边恒为 0）

`precision-policy.json` 就是按第二组定的：`atol=1.5e-2`（覆盖 8.03e-3 的约 1.9 倍）、
`rtol=2e-3`（覆盖 4.47e-4 的约 4.5 倍）、`max_abs_limit=0.05`（落在接受带 ~1e-2 与
违反契约 3.0e-2~4.6e-2 之间）。模板默认的 `atol=1.53e-5` 比 h 半边的噪声底小 500 倍，
`max_abs_limit=0.01` 也必然把正确实现判 FAIL。

## 5. 已知限制

* 标杆只覆盖**单窗口单 part**；跨 rank 的 `all_gather` 与 `merge_fwd_bwd_kernel` 不在范围内。
* 标杆默认在 CPU 上以 FP64 运行；`--device cuda` 可切到 GPU 做交叉核对，
  但正式验收以 CPU 为准。
* `hm` 视为**全区域有效**（算子输出完整的 `K×(V+K)`），不做无效区裁剪。
* 本算子没有结构性零区或单位区，因此不设 `critical_*` 分区，也不启用严格 ULP 检查。
