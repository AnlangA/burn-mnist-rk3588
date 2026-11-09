# Heterogeneous Neural Network Inference on Rockchip RK3588: A Reproducible Study Across Multi-Core CPU, Mali-G610 GPU, and 3-Core RKNPU

**面向瑞芯微 RK3588 异构 SoC 的轻量级卷积神经网络跨计算单元（CPU / GPU / NPU）可复现训练、编译与微架构性能表征开源框架**

[![Rust](https://img.shields.io/badge/Rust-2024_Edition-orange.svg)](https://www.rust-lang.org/)
[![Burn](https://img.shields.io/badge/Burn-0.22.1-red.svg)](https://burn.dev/)
[![Platform](https://img.shields.io/badge/SoC-Rockchip_RK3588-blue.svg)](https://www.rock-chips.com/)
[![Kernel](https://img.shields.io/badge/Linux-7.2.5_Mainline_%28panthor_%2B_rknpu%29-success.svg)](#4-实验平台与硬件级性能计数器计量方法-experimental-setup--hardware-telemetry-methodology)
[![Artifact](https://img.shields.io/badge/Artifact-Reproducible_%26_Evaluated-brightgreen.svg)](#7-完整可复现性指南-artifact-evaluation--step-by-step-reproducibility-guide)

> **学术论文与技术报告导航 (Research Papers & Artifact Navigation)**：
> - 📄 **[`PAPER_RK3588_HETERO_INFERENCE.md`](PAPER_RK3588_HETERO_INFERENCE.md)**：深度学术研究论文主稿（含实验可视化图表 [`docs/images/`](docs/images) 与跨层理论建模）
> - 📊 **[`REPORT_RK3588_CPU_VS_GPU.md`](REPORT_RK3588_CPU_VS_GPU.md)**：RK3588 五后端（CPU / GPU Vulkan / GPU GLES / NPU FP16 / NPU INT8）完整定量评测与微架构分析报告
> - 🖥️ **[`bsp-digit-recognizer/`](bsp-digit-recognizer)**：板端生产级 Slint `LinuxKMS` / DRM 60 FPS 交互式手写数字识别系统源码（默认搭载 RKNPU 后端，实机屏幕帧缓冲截图见 [`docs/images/npu_screen.png`](docs/images/npu_screen.png)、[`docs/images/cpu_screen.png`](docs/images/cpu_screen.png) 与 [`docs/images/gpu_screen.png`](docs/images/gpu_screen.png)）
> - 🗂️ **[`artifacts/`](artifacts)**：同源训练产出的多格式权重（`.bpk`, `.safetensors`, `.onnx`, `.rknn`, `_int8.rknn`）与原始硬件基准测试 JSON 数据集

---

## 摘要 (Abstract)

在边缘计算（Edge Computing）场景中，异构片上系统（SoC）集成了通用多核 CPU、移动 GPGPU 与专用神经网络加速器（NPU），但不同计算架构在低算术强度（Low Arithmetic Intensity）神经网络上的真实延迟-吞吐量边界尚缺乏严密的跨层定量表征。本仓库提供了一套基于 **Rust + Burn 0.22.1** 与 **Rockchip RKNPU2 (`librknnrt v1.5.2`)** 的全栈可复现研究制品（Reproducible Research Artifact），在搭载主线 **Linux `7.2.5-atk-dlrk3588+`** 内核的 **Rockchip RK3588** 平台上，对同一训练权重的卷积神经网络（`MnistCnn`，108,618 参数，1.54 MFLOPs/样本）进行了横跨五种执行后端的系统性评测：**CPU (`Burn Flex` ARMv8.2 NEON SIMD + Rayon)**、**GPU Vulkan (`Burn WGPU` / Mesa `PanVK`)**、**GPU OpenGL ES (`Burn WGPU` / Mesa `Panfrost`)**、**NPU FP16 (`RKNPU` 3-Core)** 以及 **NPU INT8 (`RKNPU` 3-Core PTQ)**。

通过结合应用层分位数延迟统计（Mean/P50/P95/P99）与内核态硬件计数器（Mali-G610 `drm-engine-panthor` 纳秒级计时去重与 RKNPU `RKNN_QUERY_PERF_RUN` 微秒级硬件寄存器计时），本研究证实：
1. **片上 L3 缓存驻留引发的 CPU 对移动 GPU 性能反超**：由于 424.3 KB 的 FP32 权重完全常驻于 RK3588 的 3 MB DynamIQ 共享 L3 缓存，CPU 在单样本（`B = 1`）下达到 **0.982 ms** 的稳态延迟，比受困于命令提交与 Staging Buffer 映射回读开销的 Mali-G610 GPU Vulkan（**3.116 ms**）与 OpenGL ES（**5.840 ms**）快 **3.17×–5.95×**；
2. **NPU 整图编译与感受野不变零拷贝 DMA 对齐的加速效应**：通过离线算子融合、片上 SRAM 激活驻留及本文提出的**感受野不变 `W = 32` 步长对齐定理**，RKNPU 在 `B = 1` 下实现 **0.163 ms**（INT8，纯硬件耗时 **88.5 μs**）与 **0.184 ms**（FP16，纯硬件耗时 **113.1 μs**）的超低延迟，相较 CPU 加速 **5.33×–6.02×**，相较 GPU 加速 **16.92×–35.80×**；
3. **三核（Core 0/1/2）并发流水线的高吞吐扩展**：利用自定义的 `RKNB` 多批次模型容器与三核并行波次调度，在 `B = 256` 下 NPU INT8 吞吐量达到 **35,975.4 samples/s**（单样本均摊 **27.8 μs**），在 10,000 张 MNIST 测试集上仅需 **280.12 ms** 完成全量推理，同时维持 **98.64%–98.66%** 的跨后端精度一致性。

---

## 1. 研究背景与问题定义 (Introduction & Problem Formulation)

现代边缘智能系统要求在单芯片功耗包络（5–15 W）内同时满足**亚毫秒级实时响应**与**高帧率图形人机交互（60 FPS GUI）**。对于参数量在百千字节至数兆字节量级的轻量级视觉骨干网，传统将算子盲目卸载至移动 GPU 的策略往往因以下三个机理而失效：
- **内核启动与同步开销主导**（Dispatch & Synchronization Dominance）：通用 GPGPU 框架（如 WebGPU / Vulkan Compute）按算子粒度派发着色器，当单层计算耗时（数十微秒）远小于内核驱动 `ioctl` 提交与主机端 `fence` 等待耗时（数毫秒）时，加速器的算术逻辑单元（ALU）处于严重饥饿状态；
- **即时编译引起的尾延迟抖动**（JIT Compilation Jitter）：首次遇到新张量维度时，运行时编译 WGSL/SPIR-V 着色器会引入数百毫秒的停顿；
- **渲染与计算队列争用**（Graphics-Compute Contention）：在无桌面合成器（Direct LinuxKMS/DRM）的嵌入式终端上，GPGPU 推理会与 60 FPS 界面渲染争用同一组着色器核心与内存控制器。

本研究通过形式化建模与硬件实测，系统性地探索 CPU、GPU 与 NPU 在不同批次规模 $B \in \lbrace 1, 16, 64, 256 \rbrace$ 下的性能边界，并提供零算力争用的生产级异构部署方案。

---

## 2. 模型拓扑与计算复杂度分析 (Model Topology, FLOPs & Memory Footprint Analysis)

目标模型 `MnistCnn`（[`src/model.rs`](src/model.rs)）对归一化后的输入张量 $X^{(0)} \in \mathbb{R}^{B \times 1 \times 28 \times 28}$ 依次执行两级卷积池化特征提取与双层全连接分类映射：

- **Stage 1 — 第一卷积池化块**（`Conv2d(1→16, 3×3) + ReLU + MaxPool2d(2×2, S=2)`，空间维 $28 \times 28 \to 26 \times 26 \to 13 \times 13$）：

<p align="center">$X^{(1)} = \mathrm{MaxPool2d}\left(\mathrm{ReLU}\left(\mathrm{Conv2d}\left(X^{(0)};\quad W^{(1)}, b^{(1)}\right)\right)\right), \qquad X^{(0)} \in \mathbb{R}^{B \times 1 \times 28 \times 28} \quad\longrightarrow\quad X^{(1)} \in \mathbb{R}^{B \times 16 \times 13 \times 13}$</p>

- **Stage 2 — 第二卷积池化块**（`Conv2d(16→32, 3×3) + ReLU + MaxPool2d(2×2, S=2)`，空间维 $13 \times 13 \to 11 \times 11 \to 5 \times 5$）：

<p align="center">$X^{(2)} = \mathrm{MaxPool2d}\left(\mathrm{ReLU}\left(\mathrm{Conv2d}\left(X^{(1)};\quad W^{(2)}, b^{(2)}\right)\right)\right), \qquad X^{(1)} \in \mathbb{R}^{B \times 16 \times 13 \times 13} \quad\longrightarrow\quad X^{(2)} \in \mathbb{R}^{B \times 32 \times 5 \times 5}$</p>

- **Stage 3 — 展平与全连接分类头**（`Flatten(800) → Linear(800→128) + ReLU → Linear(128→10) + Softmax`）：

<p align="center">$P = \mathrm{Softmax}\left(W^{(4)} \mathrm{ReLU}\left(W^{(3)} \mathrm{Flatten}\left(X^{(2)}\right) + b^{(3)}\right) + b^{(4)}\right), \qquad \mathrm{Flatten}\left(X^{(2)}\right) \in \mathbb{R}^{B \times 800} \quad\longrightarrow\quad P \in [0, 1]^{B \times 10}$</p>

### 逐层复杂度与 Roofline 算术强度分解

表 2-1 给出了单样本（`B = 1`）下的精确参数量、乘加次数（MACs）、浮点运算量（FLOPs）及 FP32 算术强度 $\mathcal{I}$（FLOPs/Byte）：

| 层名称 (Layer) | 输入维度 | 输出维度 | 参数量 ($N_{\text{param}}$) | 乘加运算 (MACs) | 浮点运算 (FLOPs) | FP32 权重字节 | 算术强度 $\mathcal{I}_ {B=1}$ (FLOPs/B) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`conv1`** (`Conv2d 1→16, 3×3`) | `[1, 1, 28, 28]` | `[1, 16, 26, 26]` | 160 (0.15%) | 97,344 (12.83%) | 194,688 | 640 B | 4.14 |
| **`act1 + pool1`** (`ReLU + Pool 2×2`) | `[1, 16, 26, 26]` | `[1, 16, 13, 13]` | 0 | 0 | 18,928 | 0 B | 0.35 |
| **`conv2`** (`Conv2d 16→32, 3×3`) | `[1, 16, 13, 13]` | `[1, 32, 11, 11]` | 4,640 (4.27%) | 557,568 (**73.50%**) | 1,115,136 | 18,560 B | 24.86 |
| **`act2 + pool2`** (`ReLU + Pool 2×2`) | `[1, 32, 11, 11]` | `[1, 32, 5, 5]` | 0 | 0 | 6,272 | 0 B | 0.34 |
| **`fc1 + act3`** (`Linear 800→128 + ReLU`) | `[1, 800]` | `[1, 128]` | 102,528 (**94.39%**) | 102,400 (13.50%) | 204,928 | 410,112 B | 0.50 |
| **`fc2 + softmax`** (`Linear 128→10`) | `[1, 128]` | `[1, 10]` | 1,290 (1.19%) | 1,280 (0.17%) | 2,610 | 5,160 B | 0.46 |
| **全网总计 (Total)** | **`[1, 1, 28, 28]`** | **`[1, 10]`** | **108,618** | **758,592** | **1,542,562 (≈ 1.54 M)** | **434,472 B (424.3 KB)** | **3.01** |

---

## 3. 云边协同可复现训练与多后端编译管线 (Cloud-to-Edge Reproducible Training & Multi-Backend Compilation Pipeline)

1. **Google Colab + Google Drive 双向容错缓存**（[`scripts/colab_runtime_train.py`](scripts/colab_runtime_train.py)）：
   自动校验 MNIST IDX Gzip 魔数（`0x00000803` / `0x00000801`）与 `$CARGO_HOME/registry` 缓存归档，支持离线缓存恢复与双镜像源容错下载。
2. **Burn 0.22.1 Autodiff 训练**（[`src/bin/train.rs`](src/bin/train.rs)）：
   使用 Adam 优化器（ $\eta = 10^{-3},\quad B = 128,\quad E = 5$ ）在 60,000 张训练图像上完成训练（总耗时 124.21 s），测试集（10,000 张）准确率达到 **98.66%**（训练日志见 [`artifacts/training_metrics.json`](artifacts/training_metrics.json)）。
3. **多格式模型导出与感受野不变 `W = 32` 零拷贝 RKNN 编译**（[`scripts/export_and_build_rknn.py`](scripts/export_and_build_rknn.py)）：
   - 从同源训练权重导出 `mnist_cnn.bpk`（435,752 B）、`mnist_cnn.safetensors`（435,136 B）与 `mnist_cnn.onnx`（435,577 B）；
   - 针对 `librknnrt v1.5.2` 在 $B > 1$ 时强制要求四维张量宽度按 8 对齐（`w_stride = 32`）的硬件约束，利用感受野不变性将输入零填充至 $W' = 32$ 并将 `fc1` 权重扩展为 `[128, 32, 5, 6]`（第 5 列置零），在无需插入 `Slice` 算子的前提下保证输出 Logits 与 `W = 28` 模型严格恒等；
   - 生成包含 $B_ {\text{sub}} \in \lbrace 1, 6, 22, 86 \rbrace$（对应总批次 `1, 16, 64, 256` 的三核切分）FP16 与 INT8 共 8 个子模型的 `RKNB` 复合容器 `mnist_cnn.rknn`（6,581,460 B）及单批次量化模型 `mnist_cnn_int8.rknn`（162,192 B）。

---

## 4. 实验平台与硬件级性能计数器计量方法 (Experimental Setup & Hardware Telemetry Methodology)

- **实验平台**：ATK-DLRK3588（Rockchip RK3588，4× Cortex-A76 @ 2.4 GHz + 4× Cortex-A55 @ 1.8 GHz，3 MB L3 Cache；ARM Mali-G610 MP4 GPU；3-Core RKNPU2 @ 800 MHz），运行 Ubuntu 24.04 ARM64 与主线 Linux **`7.2.5-atk-dlrk3588+`** 内核（搭载 [`drivers/rknpu.c`](drivers/rknpu.c) `rknpu.ko v0.9.6`）。
- **GPU 硬件计时器**（`drm-engine-panthor`）：激活 `/sys/bus/platform/devices/fb000000.gpu/profiling = 3`，解析 `/proc/<pid>/fdinfo/*` 并按 `drm-client-id` 去重（[`read_panthor_gpu_ns`](src/model.rs#L305-L343)），消除多文件描述符重复计数误差。
- **NPU 硬件计时器**（`RKNN_QUERY_PERF_RUN`）：通过 `librknnrt.so` 的 `rknn_query(ctx, RKNN_QUERY_PERF_RUN, ...)` 直接读取 RKNPU 硬件性能计数器微秒级执行时长 `run_duration`（[`src/npu.rs`](src/npu.rs)）。
- **冷热阶段正交分离**：严格分离模型加载（`model_load_ms`）、首次计算图/着色器编译预热（`single_sample_warmup_ms`、`test_set_warmup_ms`）与多轮预热后的稳态推理延迟（`mean_batch_ms`、`p50`、`p95`、`p99`）。

---

## 5. 实验结果与多维定量评估 (Quantitative Evaluation)

### 5.1 全量测试集（10,000 张）与板载手写样本（20 张 PNG）端到端表现

| 评测维度 | NPU INT8 (`RKNPU 3-Core INT8`) | NPU FP16 (`RKNPU 3-Core FP16`) | CPU (`Burn Flex SIMD+Rayon`) | GPU Vulkan (`Burn WGPU / PanVK`) | GPU OpenGL ES (`Burn WGPU / Panfrost`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **单样本冷启动预热 (`single_sample_warmup_ms`)** | **0.407 ms** (综合 **0.281 ms**) | 0.458 ms (综合 0.409 ms) | 11.845 ms (综合 7.799 ms) | 123.927 ms (综合 120.806 ms) | 143.684 ms (综合 145.318 ms) |
| **MNIST 10,000 张测试集准确率** | **98.64%** (9,864 / 10,000) | **98.65%** (9,865 / 10,000) | **98.66%** (9,866 / 10,000) | **98.66%** (9,866 / 10,000) | **98.66%** (9,866 / 10,000) |
| **MNIST 10,000 张稳态总耗时 (`test_set_eval_ms`)** | **283.64 ms** (综合 **280.12 ms**) | **485.77 ms** (综合 **388.41 ms**) | 1,647.67 ms (综合 1,651.92 ms) | 3,343.89 ms (综合 3,336.91 ms) | 3,549.23 ms (综合 3,524.72 ms) |
| **20 张板载 PNG 准确率与平均延迟** | **95.00% / 0.165 ms** (综合 0.191 ms) | **95.00% / 0.222 ms** (综合 0.219 ms) | 95.00% / 1.089 ms (综合 1.097 ms) | 95.00% / 3.190 ms (综合 3.272 ms) | 95.00% / 6.551 ms (综合 6.535 ms) |

### 5.2 不同 Batch Size (B = 1, 16, 64, 256) 稳态延迟、吞吐量与硬件计数器对比（独立进程实测）

| Batch (`B`) | 后端配置 | Mean 延迟 (ms) | P50 (ms) | P95 (ms) | P99 (ms) | 单样本延迟 (ms/sample) | 吞吐量 (samples/s) | 硬件计数器耗时 (ms/batch) | 相对 CPU 加速比 | 相对 GPU (Vulkan) 加速比 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **NPU (`RKNPU INT8`)** | **0.163** | **0.162** | **0.164** | **0.195** | **0.1631** | **6,132.4** | `0.0885` (RKNPU) | **6.02×** | **19.11×** |
| **1** | **NPU (`RKNPU FP16`)** | **0.184** | **0.183** | **0.186** | **0.216** | **0.1842** | **5,429.8** | `0.1131` (RKNPU) | **5.33×** | **16.92×** |
| **1** | **CPU (`Burn Flex`)** | 0.982 | 0.996 | 1.174 | 1.282 | 0.9822 | 1,018.2 | `0.0000` | 1.00× | 3.17× |
| **1** | **GPU (`WGPU Vulkan`)** | 3.116 | 3.012 | 3.421 | 6.383 | 3.1165 | 320.9 | `4.3863` (Panthor) | 0.32× | 1.00× |
| **1** | **GPU (`WGPU GLES`)** | 5.840 | 5.856 | 6.060 | 6.146 | 5.8395 | 171.2 | `1.8054` (Panthor) | 0.17× | 0.53× |
| **16** | **NPU (`RKNPU INT8`)** | **0.829** | **0.851** | **1.000** | **1.006** | **0.0518** | **19,307.0** | `1.0702` (3-Core Σ) | **4.87×** | **29.98×** |
| **16** | **NPU (`RKNPU FP16`)** | **1.167** | **1.174** | **1.300** | **1.319** | **0.0730** | **13,706.2** | `1.5592` (3-Core Σ) | **3.46×** | **21.28×** |
| **16** | **CPU (`Burn Flex`)** | 4.037 | 4.074 | 4.958 | 5.253 | 0.2523 | 3,963.5 | `0.0000` | 1.00× | 6.15× |
| **16** | **GPU (`WGPU Vulkan`)** | 24.845 | 23.572 | 28.795 | 29.094 | 1.5528 | 644.0 | `64.5112` (Panthor) | 0.16× | 1.00× |
| **16** | **GPU (`WGPU GLES`)** | 28.160 | 28.202 | 28.609 | 28.866 | 1.7600 | 568.2 | `23.1376` (Panthor) | 0.14× | 0.88× |
| **64** | **NPU (`RKNPU INT8`)** | **2.033** | **2.014** | **2.261** | **2.281** | **0.0318** | **31,472.9** | `3.0007` (3-Core Σ) | **6.79×** | **18.37×** |
| **64** | **NPU (`RKNPU FP16`)** | **2.904** | **2.911** | **2.989** | **2.994** | **0.0454** | **22,035.1** | `4.9963` (3-Core Σ) | **4.75×** | **12.86×** |
| **64** | **CPU (`Burn Flex`)** | 13.802 | 14.150 | 16.823 | 17.152 | 0.2157 | 4,637.0 | `0.0000` | 1.00× | 2.71× |
| **64** | **GPU (`WGPU Vulkan`)** | 37.354 | 36.412 | 39.993 | 40.127 | 0.5837 | 1,713.3 | `93.4342` (Panthor) | 0.37× | 1.00× |
| **64** | **GPU (`WGPU GLES`)** | 36.888 | 36.600 | 40.053 | 40.409 | 0.5764 | 1,735.0 | `31.0738` (Panthor) | 0.37× | 1.01× |
| **256** | **NPU (`RKNPU INT8`)** | **7.116** | **7.105** | **7.192** | **7.283** | **0.0278** | **35,975.4** | `10.8920` (3-Core Σ) | **6.21×** | **12.36×** |
| **256** | **NPU (`RKNPU FP16`)** | **12.406** | **12.420** | **12.468** | **12.471** | **0.0485** | **20,635.2** | `18.4163` (3-Core Σ) | **3.56×** | **7.09×** |
| **256** | **CPU (`Burn Flex`)** | 44.172 | 45.773 | 49.987 | 50.300 | 0.1725 | 5,795.5 | `0.0000` | 1.00× | 1.99× |
| **256** | **GPU (`WGPU Vulkan`)** | 87.968 | 87.658 | 89.155 | 89.383 | 0.3436 | 2,910.2 | `243.7145` (Panthor) | 0.50× | 1.00× |
| **256** | **GPU (`WGPU GLES`)** | 89.054 | 88.734 | 91.190 | 91.931 | 0.3479 | 2,874.6 | `81.8866` (Panthor) | 0.50× | 0.99× |

> 完整同进程对比表（`bench --backend all`）与 20 张 PNG 样本逐样本置信度分析详见 [`REPORT_RK3588_CPU_VS_GPU.md`](REPORT_RK3588_CPU_VS_GPU.md#5-实验结果与多维定量评估-quantitative-evaluation)。

---

## 6. 微架构机理深度剖析 (Microarchitectural & Roofline Analysis)

1. **CPU 何以击败移动 GPU**：`MnistCnn` 的总算术强度仅为 $\mathcal{I}(1) = 3.01\text{ FLOPs/Byte}$。RK3588 CPU 凭借 3 MB 共享 L3 缓存将 424.3 KB 权重完全锁定在片上高速缓存内，配合 NEON SIMD 与 Rayon 线程池实现了零驱动开销的核内计算；反观 Mali-G610 GPU，逐算子命令队列提交与末尾 Staging Buffer 映射同步引入了 2.5–5.5 ms 的固定延迟地板（Latency Floor）。
2. **NPU 何以实现数量级跃升**：RKNPU 通过离线整图编译将多层算子融合为单一硬件指令流，单次 `ioctl` 即可驱动片上脉动阵列与内部 SRAM 完成全部特征提取（`B = 1` 硬件耗时仅 88.5 μs），并在大批次下通过 3 核（`Core 0/1/2`）硬件并行与 `W = 32` 零拷贝突发 DMA 实现了高达 **35,975.4 samples/s** 的吞吐率。
3. **推理与图形渲染的系统级解耦**：板端 `digit-recognizer.service`（[`bsp-digit-recognizer/`](bsp-digit-recognizer)）默认运行 NPU 后端（`DIGIT_RECOGNIZER_BACKEND=npu`，状态栏实时显示 **`● NPU · RKNPU`**），进程常驻内存从 GPU 版的 196.1 MB 降至 **102.1 MB**，并将 Mali-G610 GPU 100% 释放给 Slint `LinuxKMS` / `FemtoVG` 进行无抖动的 60 FPS 全屏扫描输出。

---

## 7. 完整可复现性指南 (Artifact Evaluation & Step-by-Step Reproducibility Guide)

### 7.1 本地验证（单元测试、静态检查与 RKNB 容器校验）
```bash
git clone https://github.com/AnlangA/burn-mnist-rk3588.git
cd burn-mnist-rk3588

cargo fmt --check
cargo clippy --all-targets --features cpu,gpu,npu -- -D warnings
bash -n scripts/sync_to_ubuntu_bsp.sh
cargo test --features cpu,gpu,npu
```

### 7.2 训练与模型导出转换（Colab / 本地 Linux / macOS）
```bash
# 1. 运行 Burn Autodiff 训练（含 Google Drive 缓存自动探测与导出）
./scripts/colab_runtime_train.sh

# 2. 导出 ONNX 并编译 RK3588 RKNPU 多批次 FP16/INT8 容器
python3 scripts/export_and_build_rknn.py \
    --safetensors artifacts/mnist_cnn.safetensors \
    --data-dir data/mnist \
    --out-onnx artifacts/mnist_cnn.onnx \
    --out-rknn artifacts/mnist_cnn.rknn \
    --out-rknn-int8 artifacts/mnist_cnn_int8.rknn
```

### 7.3 交叉编译、板端零停机原子部署与全自动评测
```bash
# 自动同步至 Ubuntu 交叉编译服务器、构建 CPU/GPU/NPU 二进制、部署至 RK3588 并回收 4 份评测 JSON
./scripts/sync_to_ubuntu_bsp.sh --deploy-and-bench
```

### 7.4 板端单后端与全后端独立复现命令
在 RK3588 开发板上可直接执行以下命令复现全部评测数据：
```bash
# NPU 独立评测 (FP16 & INT8)
/opt/burn-mnist-rk3588/npu/mnist-npu --backend npu-all \
    --model /opt/burn-mnist-rk3588/model/mnist_cnn.rknn \
    --data-dir /opt/burn-mnist-rk3588/data/mnist \
    --images-dir /opt/burn-mnist-rk3588/images \
    --iterations 100 \
    --output-json /opt/burn-mnist-rk3588/results/rk3588_npu_benchmark.json

# CPU 独立评测 (Burn Flex NEON SIMD + Rayon)
/opt/burn-mnist-rk3588/cpu/mnist-cpu \
    --model /opt/burn-mnist-rk3588/model/mnist_cnn.bpk \
    --data-dir /opt/burn-mnist-rk3588/data/mnist \
    --images-dir /opt/burn-mnist-rk3588/images \
    --iterations 100 \
    --output-json /opt/burn-mnist-rk3588/results/rk3588_cpu_benchmark.json

# GPU 独立评测 (Burn WGPU Vulkan PanVK & OpenGL ES Panfrost)
/opt/burn-mnist-rk3588/gpu/mnist-gpu --backend gpu-all \
    --model /opt/burn-mnist-rk3588/model/mnist_cnn.bpk \
    --data-dir /opt/burn-mnist-rk3588/data/mnist \
    --images-dir /opt/burn-mnist-rk3588/images \
    --iterations 100 \
    --output-json /opt/burn-mnist-rk3588/results/rk3588_gpu_benchmark.json

# 五后端同进程综合对比评测
/opt/burn-mnist-rk3588/bench --backend all \
    --model /opt/burn-mnist-rk3588/model/mnist_cnn.bpk \
    --data-dir /opt/burn-mnist-rk3588/data/mnist \
    --images-dir /opt/burn-mnist-rk3588/images \
    --iterations 100 \
    --output-json /opt/burn-mnist-rk3588/results/rk3588_benchmark_results.json
```

---

## License

Dual-licensed under **MIT** OR **Apache-2.0** (the out-of-tree kernel driver [`drivers/rknpu.c`](drivers/rknpu.c) is licensed under **GPL-2.0**).
