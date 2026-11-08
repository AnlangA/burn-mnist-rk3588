# RK3588 Rust Burn 手写数字识别 (MNIST CNN) CPU / GPU / NPU 三版本部署及性能对比报告

## 1. 概述与架构设计

本项目基于 **Rust + Burn 0.22.1** 与 **Rockchip RKNPU2 (`librknnrt`)** 构建了端到端的手写数字图像识别（MNIST 28×28 CNN）训练、跨平台模型序列化与 **Rockchip RK3588** 边缘侧 **CPU / GPU / NPU 三版本**推理部署框架：

1. **模型结构 (`MnistCnn`)**：
   - `conv1`: `Conv2d(1 -> 16, kernel=3×3, padding=Valid)` + `ReLU` + `MaxPool2d(2×2, stride=2)`（输出 `16×13×13`）
   - `conv2`: `Conv2d(16 -> 32, kernel=3×3, padding=Valid)` + `ReLU` + `MaxPool2d(2×2, stride=2)`（输出 `32×5×5`）
   - `fc1`: `Linear(800 -> 128)` + `ReLU`
   - `fc2`: `Linear(128 -> 10)` + `Softmax`
   - **参数规模与多格式模型产物**：约 `107,786` 个参数，所有格式共享同一份训练权重：
     - `mnist_cnn.bpk`（Burnpack 权重）：`435,752 B` (`426 KB`)，SHA1 `71abbdd3c48860c9a4fd59d879c7328bc597ce9a`
     - `mnist_cnn.safetensors`（SafeTensors 权重）：`435,136 B` (`425 KB`)，SHA1 `1f0033c3067f60b085e1e19a66b044bf3c10dba6`
     - `mnist_cnn.onnx`（ONNX Opset 17 计算图）：`435,577 B` (`425 KB`)，SHA1 `099e59456458f49da862f318ed9569ca43e51214`
     - `mnist_cnn.rknn`（RK3588 RKNB 多批次 `batch=1/16/64/256` FP16 + INT8 离线编译包）：`6,581,460 B` (`6.28 MB`)，SHA1 `18d80802445362d7a09f9a3dac54f6535a1d0646`
     - `mnist_cnn_int8.rknn`（RK3588 单批次 INT8 量化模型）：`162,192 B` (`158 KB`)，SHA1 `7e0adae852d19d52b6601be357c9ac77f2704ddd`
2. **三版本编译与部署架构**：
   - **CPU 独立版本 (`mnist-cpu` / `digit-recognizer-cpu`)**：仅启用 `--no-default-features --features cpu`，底层采用 **Burn Flex CPU (`burn/flex` + `burn/simd` + `burn/rayon`)** 后端，利用 RK3588 的 4× Cortex-A76 + 4× Cortex-A55 八核 ARMv8.2 NEON SIMD 与多线程并行加速，不链接任何 GPU/WGPU 或 RKNPU 依赖（二进制体积仅 `13 MB`）。
   - **GPU 独立版本 (`mnist-gpu` / `digit-recognizer-gpu`)**：仅启用 `--no-default-features --features gpu`，底层采用 **Burn WGPU (`burn/wgpu` + `CubeCL`)** 后端，针对 RK3588 内置 **ARM Mali-G610 MP4 GPU**（主线 Linux `7.2.5-atk-dlrk3588+` 内核 `panthor` DRM 驱动，支持 **Vulkan (Mesa PanVK)** 与 **OpenGL ES 3.1 (Mesa Panfrost)** 双图形计算管线，二进制体积 `83 MB`）。
   - **NPU 独立版本 (`mnist-npu` / `digit-recognizer-npu`)**：仅启用 `--no-default-features --features npu`，底层通过 [`src/npu.rs`](file:///Users/a/workspace/ubuntu/burn-mnist-rk3588/src/npu.rs) 的 `libloading` 动态加载 `/usr/lib/librknnrt.so` (`v1.5.2`)，驱动 RK3588 内置 **6 TOPS 3-Core RKNPU**（主线内核桥接驱动 `rknpu.ko v0.9.6 @ 800 MHz`，支持 `batch=1` 低延迟单核模式与 `batch=16/64/256` 3 核 `RKNN_NPU_CORE_0_1_2` 硬件并发流水线，支持 **FP16** 与 **INT8** 双精度，推理二进制体积仅 `2.6 MB`）。
   - **综合对比评测与 UI 展示版本 (`bench` / `digit-recognizer`)**：`bench` 同时启用 `cpu,gpu,npu` 特性，支持同一进程内对比 **CPU (`Burn Flex`)**、**NPU FP16 (`RKNPU`)**、**NPU INT8 (`RKNPU INT8`)**、**GPU Vulkan (`PanVK`)**、**GPU OpenGL ES (`Panfrost`)** 五种模式的精度、预热耗时、稳态延迟与硬件计数器指标；板端 `digit-recognizer.service` 默认部署并运行 **NPU 版本 (`digit-recognizer-npu`，`DIGIT_RECOGNIZER_BACKEND=npu`)**，在物理屏幕顶部状态栏实时显示 **`● NPU · RKNPU`**。

---

## 2. 模型训练结果汇总 (Colab `d7f33ff`)

训练使用完整 MNIST 数据集（`60,000` 张训练图像 + `10,000` 张测试图像），超参数为 `epochs = 5, batch_size = 128, lr = 0.001 (Adam)`。训练指标记录于 [`artifacts/training_metrics.json`](file:///Users/a/workspace/ubuntu/burn-mnist-rk3588/artifacts/training_metrics.json)：

| Epoch | Train Loss | Train Accuracy | Test Loss (10k) | Test Accuracy (10k) | 耗时 (s) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.2245 | 93.65% | 0.0663 | 97.92% | 25.53 |
| 2 | 0.0637 | 98.00% | 0.0407 | 98.77% | 23.73 |
| 3 | 0.0431 | 98.64% | 0.0392 | 98.65% | 28.32 |
| 4 | 0.0316 | 99.04% | 0.0349 | 98.88% | 23.08 |
| **5** | **0.0253** | **99.22%** | **0.0391** | **98.66%** | **23.55** |

- **总训练耗时**：`124.21 s`
- **最终测试集准确率**：**`98.66%`**（10,000 张测试集正确识别 9,866 张）
- **产出模型文件**：
  - [`artifacts/mnist_cnn.bpk`](file:///Users/a/workspace/ubuntu/burn-mnist-rk3588/artifacts/mnist_cnn.bpk) (`426 KB`, SHA1: `71abbdd3c48860c9a4fd59d879c7328bc597ce9a`)
  - [`artifacts/mnist_cnn.safetensors`](file:///Users/a/workspace/ubuntu/burn-mnist-rk3588/artifacts/mnist_cnn.safetensors) (`425 KB`, SHA1: `1f0033c3067f60b085e1e19a66b044bf3c10dba6`)
  - [`artifacts/mnist_cnn.onnx`](file:///Users/a/workspace/ubuntu/burn-mnist-rk3588/artifacts/mnist_cnn.onnx) (`425 KB`, SHA1: `099e59456458f49da862f318ed9569ca43e51214`)
  - [`artifacts/mnist_cnn.rknn`](file:///Users/a/workspace/ubuntu/burn-mnist-rk3588/artifacts/mnist_cnn.rknn) (`6.28 MB`, SHA1: `18d80802445362d7a09f9a3dac54f6535a1d0646`)
  - [`artifacts/mnist_cnn_int8.rknn`](file:///Users/a/workspace/ubuntu/burn-mnist-rk3588/artifacts/mnist_cnn_int8.rknn) (`158 KB`, SHA1: `7e0adae852d19d52b6601be357c9ac77f2704ddd`)
  - [`artifacts/training_metrics.json`](file:///Users/a/workspace/ubuntu/burn-mnist-rk3588/artifacts/training_metrics.json) (`1.6 KB`)

---

## 3. RK3588 硬件环境与三版本部署验证

### 3.1 目标开发板硬件与内核环境
- **开发板**：ATK-DLRK3588 (`ubuntu@10.42.0.88`)
- **SoC**：Rockchip RK3588（4× Cortex-A76 @ 2.4GHz + 4× Cortex-A55 @ 1.8GHz，3MB 共享 DynamIQ L3 Cache）
- **GPU**：ARM Mali-G610 MP4（4 Shader Cores），主线 DRM `panthor` 驱动（计算节点 `/dev/dri/renderD128`，显示节点 `/dev/dri/card0`）
- **NPU**：Rockchip RKNPU2 三核 NPU（Core 0/1/2，综合算力 6 TOPS INT8 / 1.5 TFLOPS FP16），加载主线 `7.2.5-atk-dlrk3588+` 移植的 `rknpu.ko v0.9.6` 内核模块（设备节点 `/dev/rknpu`，运行频率 `800 MHz`，用户态运行时 `/usr/lib/librknnrt.so v1.5.2`）
- **硬件性能计数器**：
  - **GPU 计数器 (`panthor`)**：开启 `/sys/bus/platform/devices/fb000000.gpu/profiling = 3`，解析 `/proc/<pid>/fdinfo/*` 中的 `drm-engine-panthor: <ns> ns` 并按 `drm-client-id` 去重，精确采集 Mali-G610 硬件引擎执行时间。
  - **NPU 计数器 (`rknpu_perf_run`)**：通过 `rknn_query(ctx, RKNN_QUERY_PERF_RUN, &perf_run, sizeof(perf_run))` 直接读取 RKNPU 硬件级算子执行时间（微秒级精度 `run_duration`），记录于 `rknpu_hw_total_ms` 与 `rknpu_hw_per_batch_ms`。

### 3.2 板端三版本独立部署路径与默认 NPU 桌面服务
通过 [`scripts/sync_to_ubuntu_bsp.sh`](file:///Users/a/workspace/ubuntu/burn-mnist-rk3588/scripts/sync_to_ubuntu_bsp.sh) 将训练产物与源码同步至 Ubuntu 构建服务器 (`192.168.10.117:/home/a/workspace/rk3588/last-rk3588/atk-dlrk3588-bsp`) 交叉编译后，部署至 RK3588 开发板：

| 版本类别 | 板端部署路径 | 编译特性 (`Cargo Features`) | 二进制体积 | 运行时后端描述 |
| :--- | :--- | :--- | :---: | :--- |
| **CPU 独立推理版** | `/opt/burn-mnist-rk3588/cpu/mnist-cpu`<br>`/usr/local/bin/mnist-cpu` | `--no-default-features --features cpu` | 13 MB | `Burn Flex CPU (SIMD + Rayon)` |
| **GPU 独立推理版** | `/opt/burn-mnist-rk3588/gpu/mnist-gpu`<br>`/usr/local/bin/mnist-gpu` | `--no-default-features --features gpu` | 83 MB | `Burn WGPU (Vulkan)` & `Burn WGPU (OpenGL ES)` |
| **NPU 独立推理版** | `/opt/burn-mnist-rk3588/npu/mnist-npu`<br>`/usr/local/bin/mnist-npu` | `--no-default-features --features npu` | 2.6 MB | `RK3588 NPU 3-Core (FP16 & INT8, librknnrt 1.5.2)` |
| **UI 展示默认服务 (NPU 版)** | `/usr/local/bin/digit-recognizer` (`digit-recognizer.service`)<br>`/opt/burn-mnist-rk3588/npu/digit-recognizer-npu`<br>`/usr/local/bin/digit-recognizer-npu` | `--no-default-features --features npu` | 17 MB | `RK3588 NPU (RKNPU FP16/INT8)` + Slint LinuxKMS (`● NPU · RKNPU`) |
| **UI 展示 GPU 版** | `/opt/burn-mnist-rk3588/gpu/digit-recognizer-gpu`<br>`/usr/local/bin/digit-recognizer-gpu` | `--no-default-features --features gpu` | 97 MB | `Burn WGPU (Vulkan)` + Slint LinuxKMS (`● GPU · WGPU`) |
| **UI 展示 CPU 版** | `/opt/burn-mnist-rk3588/cpu/digit-recognizer-cpu`<br>`/usr/local/bin/digit-recognizer-cpu` | `--no-default-features --features cpu` | 27 MB | `Burn Flex CPU (SIMD + Rayon)` + Slint LinuxKMS (`● CPU · FLEX`) |
| **综合基准测试工具** | `/opt/burn-mnist-rk3588/bench` | `--features cpu,gpu,npu` | 84 MB | 同进程评测 `cpu` / `npu` / `npu-int8` / `vulkan` / `gl` |
| **模型与测试数据集** | `/opt/burn-mnist-rk3588/model/mnist_cnn.{bpk,safetensors,onnx,rknn}`<br>`/opt/burn-mnist-rk3588/model/mnist_cnn_int8.rknn`<br>`/usr/share/digit-recognizer/model/mnist_cnn.{bpk,safetensors,onnx,rknn}`<br>`/usr/share/digit-recognizer/model/mnist_cnn_int8.rknn`<br>`/opt/burn-mnist-rk3588/data/mnist/t10k-*` | — | 9.5 MB | Burnpack/RKNN 权重 + 10,000 张测试集 + 20 张板载 PNG 样本 |

- **显示服务 (`digit-recognizer.service`) 默认 NPU 后端与零长黑屏热更新机制**：
  1. **板端服务默认启动并运行 NPU 版本 (`RK3588 NPU (RKNPU FP16)`)**：`/etc/default/digit-recognizer` 配置 `DIGIT_RECOGNIZER_BACKEND=npu`，`/usr/local/bin/digit-recognizer` 默认部署并锁定为 NPU 版本 (`digit-recognizer-npu`)。服务启动后在屏幕上渲染 **`● NPU · RKNPU`** 徽标及 `RK3588 RKNPU (FP16) · 0.61 ms` 识别结果卡片；进程常驻内存从 GPU WGPU 版本的 `196.1 MB` 降低至 **`102.1 MB`**（二进制体积从 `97 MB` 降至 `17 MB`），并将 Mali-G610 GPU 100% 释放给 Slint LinuxKMS / FemtoVG 进行 60 FPS UI 渲染。
  2. **设备节点与沙箱权限配置**：通过 `/etc/udev/rules.d/99-rknpu.rules` 与 `board-service-guard.sh` 确保 `/dev/rknpu`（以及 `/dev/dri/renderD128`、`/dev/dri/renderD129`）权限为 `root:render 0666`、`/usr/lib/librknnrt.so` 权限为 `0755`，且 `digit-recognizer.service` 配置 `SupplementaryGroups=video input render`，确保服务沙箱可直接访问 NPU 与 DRM 节点。
  3. **最小化黑屏时间（切换实测仅 `~1.50 s`）与自动回滚保护**：若当前板端 `/usr/local/bin/digit-recognizer` 已处于活跃渲染状态且为 NPU 后端（通过 `is_npu_binary` 与 `journalctl` 中 `badge="NPU · RKNPU"` 校验），`board-service-guard.sh` 在同步时直接保留该运行实例（**0 ms 停机**）；仅当二进制或模型发生变化时才执行本地 `mv -f` 原子替换（实测切换仅 `1,500 ms`），并在异常时自动回滚至 `/var/backups/digit-recognizer-last/` 中的已知正常 NPU 版本。
  4. **基准测试期间无需停止显示服务**：`mnist-npu`、`mnist-gpu` 与 `bench` 分别通过 `/dev/rknpu` 与 `/dev/dri/renderD128` 提交计算任务，无需独占 DRM KMS Master (`/dev/dri/card0`)；默认在保持屏幕正常显示的同时完成全部 CPU / GPU / NPU 基准测试。

---

## 4. RK3588 CPU vs GPU vs NPU 性能对比评测结果

所有原始基准测试数据保存在：
- 独立 CPU 版本结果：[`artifacts/rk3588_cpu_benchmark.json`](file:///Users/a/workspace/ubuntu/burn-mnist-rk3588/artifacts/rk3588_cpu_benchmark.json)
- 独立 GPU 版本结果：[`artifacts/rk3588_gpu_benchmark.json`](file:///Users/a/workspace/ubuntu/burn-mnist-rk3588/artifacts/rk3588_gpu_benchmark.json)
- 独立 NPU 版本结果：[`artifacts/rk3588_npu_benchmark.json`](file:///Users/a/workspace/ubuntu/burn-mnist-rk3588/artifacts/rk3588_npu_benchmark.json)
- 全后端综合对比结果：[`artifacts/rk3588_benchmark_results.json`](file:///Users/a/workspace/ubuntu/burn-mnist-rk3588/artifacts/rk3588_benchmark_results.json)

### 4.1 整体准确率、预热开销与全量测试集 (10,000 张) 推理耗时对比

评测代码 ([`src/bench.rs`](file:///Users/a/workspace/ubuntu/burn-mnist-rk3588/src/bench.rs)) 将首次计算图/着色器/NPU 上下文预热耗时（`single_sample_warmup_ms` 与 `test_set_warmup_ms`）与稳态推理耗时（`test_set_eval_ms` 及 PNG 样本稳态延迟）进行了严格分离：

| 评测维度 | NPU INT8 (`mnist-npu`<br>RKNPU 3-Core INT8) | NPU FP16 (`mnist-npu`<br>RKNPU 3-Core FP16) | CPU (`mnist-cpu`<br>Burn Flex SIMD+Rayon) | GPU Vulkan (`mnist-gpu`<br>Burn WGPU / PanVK) | GPU OpenGL ES (`mnist-gpu`<br>Burn WGPU / Panfrost) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **模型加载耗时 (`model_load_ms`)** | 3.031 ms (综合 2.745 ms) | 13.080 ms (综合 10.703 ms) | 0.492 ms (综合 0.748 ms) | **0.188 ms** (综合 **0.176 ms**) | 0.204 ms (综合 0.225 ms) |
| **单样本 `[1,1,28,28]` 首次预热 (`single_sample_warmup_ms`)** | **0.407 ms** (综合 **0.281 ms**) | 0.458 ms (综合 0.409 ms) | 11.845 ms (综合 7.799 ms) | 123.927 ms (综合 120.806 ms) | 143.684 ms (综合 145.318 ms) |
| **MNIST 10,000 张测试集准确率** | **98.64% (9,864 / 10,000)** | **98.65% (9,865 / 10,000)** | **98.66% (9,866 / 10,000)** | **98.66% (9,866 / 10,000)** | **98.66% (9,866 / 10,000)** |
| **MNIST 10,000 张测试集预热 (`test_set_warmup_ms`)** | **33.19 ms** (综合 **29.75 ms**) | 38.31 ms (综合 36.67 ms) | 43.14 ms (综合 58.63 ms) | 497.49 ms (综合 498.03 ms) | 514.58 ms (综合 529.07 ms) |
| **MNIST 10,000 张稳态总耗时 (`test_set_eval_ms`)** | **283.64 ms** (综合 **280.12 ms**) | **485.77 ms** (综合 **388.41 ms**) | 1,647.67 ms (综合 1,651.92 ms) | 3,343.89 ms (综合 3,336.91 ms) | 3,549.23 ms (综合 3,524.72 ms) |
| **20 张板载手写数字 PNG 准确率** | **95.00% (19 / 20)** | **95.00% (19 / 20)** | **95.00% (19 / 20)** | **95.00% (19 / 20)** | **95.00% (19 / 20)** |
| **20 张 PNG 稳态平均单张延迟** | **0.165 ms** (综合 **0.189 ms**) | **0.222 ms** (综合 **0.215 ms**) | 1.089 ms (综合 1.090 ms) | 3.190 ms (综合 3.267 ms) | 6.531 ms (综合 6.299 ms) |

> **精度一致性验证**：
> - CPU (`Burn Flex`)、GPU Vulkan (`PanVK`)、GPU OpenGL ES (`Panfrost`) 三个 FP32 后端在 10,000 张 MNIST 测试集上均取得完全一致的 **98.66% (9,866/10,000)** 准确率；
> - **NPU FP16 (`RKNPU`)** 在 10,000 张测试集上准确率为 **98.65% (9,865/10,000)**（仅 1 张边界样本差异），**NPU INT8 (`RKNPU INT8`)** 准确率为 **98.64% (9,864/10,000)**（仅 2 张边界样本差异）；
> - 在 20 张板载 PNG 样本上，全部五个后端的预测类别 **100% 一致（均为 19/20 = 95.00%）**，且 Softmax 置信度高度吻合（例如 `d0_0003.png` 在 CPU/GPU 上为 `0.9999638`，在 NPU FP16 上为 `0.9999640`，在 NPU INT8 上为 `0.9999664`）。

---

### 4.2 不同 Batch Size (1, 16, 64, 256) 稳态推理延迟、吞吐量与硬件计数器对比

每个 Batch Size 在完成预热迭代（Warmup）后，连续测量稳态前向推理 + 结果同步的延迟分布，以及硬件级 `rknpu` (`RKNN_QUERY_PERF_RUN`) / `panthor` (`drm-engine-panthor`) 计数器：

#### 表 1：独立部署版本 (`mnist-npu` vs `mnist-cpu` vs `mnist-gpu`) 稳态性能指标

| Batch Size | 后端版本 | Mean 批次延迟 (ms) | P50 延迟 (ms) | P95 延迟 (ms) | P99 延迟 (ms) | 单样本均摊延迟 (ms/sample) | 吞吐量 (samples/sec) | NPU / GPU 硬件引擎耗时 (ms/batch) | 相对 CPU 加速比 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **NPU (`RKNPU INT8`)** | **0.163** | **0.162** | **0.164** | **0.195** | **0.1631** | **6,132.4** | `0.089` (RKNPU) | **6.02× (最快)** |
| **1** | **NPU (`RKNPU FP16`)** | **0.184** | **0.183** | **0.186** | **0.216** | **0.1842** | **5,429.8** | `0.113` (RKNPU) | **5.33×** |
| **1** | **CPU (`Burn Flex`)** | 0.982 | 0.996 | 1.174 | 1.282 | 0.9822 | 1,018.2 | `0.000` | **1.00× (基准)** |
| **1** | **GPU (`WGPU Vulkan`)** | 3.116 | 3.012 | 3.421 | 6.383 | 3.1165 | 320.9 | `4.386` (Panthor) | 0.32× (慢 3.17×) |
| **1** | **GPU (`WGPU OpenGL ES`)** | 5.840 | 5.856 | 6.060 | 6.146 | 5.8395 | 171.2 | `1.805` (Panthor) | 0.17× (慢 5.95×) |
| **16** | **NPU (`RKNPU INT8`)** | **0.829** | **0.851** | **1.000** | **1.006** | **0.0518** | **19,307.0** | `1.070` (3-Core Σ) | **4.87× (最快)** |
| **16** | **NPU (`RKNPU FP16`)** | **1.167** | **1.174** | **1.300** | **1.319** | **0.0730** | **13,706.2** | `1.559` (3-Core Σ) | **3.46×** |
| **16** | **CPU (`Burn Flex`)** | 4.037 | 4.074 | 4.958 | 5.253 | 0.2523 | 3,963.5 | `0.000` | **1.00× (基准)** |
| **16** | **GPU (`WGPU Vulkan`)** | 24.845 | 23.572 | 28.795 | 29.094 | 1.5528 | 644.0 | `64.511` (Panthor) | 0.16× (慢 6.15×) |
| **16** | **GPU (`WGPU OpenGL ES`)** | 28.160 | 28.202 | 28.609 | 28.866 | 1.7600 | 568.2 | `23.138` (Panthor) | 0.14× (慢 6.98×) |
| **64** | **NPU (`RKNPU INT8`)** | **2.033** | **2.014** | **2.261** | **2.281** | **0.0318** | **31,472.9** | `3.001` (3-Core Σ) | **6.79× (最快)** |
| **64** | **NPU (`RKNPU FP16`)** | **2.904** | **2.911** | **2.989** | **2.994** | **0.0454** | **22,035.1** | `4.996` (3-Core Σ) | **4.75×** |
| **64** | **CPU (`Burn Flex`)** | 13.802 | 14.150 | 16.823 | 17.152 | 0.2157 | 4,637.0 | `0.000` | **1.00× (基准)** |
| **64** | **GPU (`WGPU Vulkan`)** | 37.354 | 36.412 | 39.993 | 40.127 | 0.5837 | 1,713.3 | `93.434` (Panthor) | 0.37× (慢 2.71×) |
| **64** | **GPU (`WGPU OpenGL ES`)** | 36.888 | 36.600 | 40.053 | 40.409 | 0.5764 | 1,735.0 | `31.074` (Panthor) | 0.37× (慢 2.67×) |
| **256** | **NPU (`RKNPU INT8`)** | **7.116** | **7.105** | **7.192** | **7.283** | **0.0278** | **35,975.4** | `10.892` (3-Core Σ) | **6.21× (最快)** |
| **256** | **NPU (`RKNPU FP16`)** | **12.406** | **12.420** | **12.468** | **12.471** | **0.0485** | **20,635.2** | `18.416` (3-Core Σ) | **3.56×** |
| **256** | **CPU (`Burn Flex`)** | 44.172 | 45.773 | 49.987 | 50.300 | 0.1725 | 5,795.5 | `0.000` | **1.00× (基准)** |
| **256** | **GPU (`WGPU Vulkan`)** | 87.968 | 87.658 | 89.155 | 89.383 | 0.3436 | 2,910.2 | `243.714` (Panthor) | 0.50× (慢 1.99×) |
| **256** | **GPU (`WGPU OpenGL ES`)** | 89.054 | 88.734 | 91.190 | 91.931 | 0.3479 | 2,874.6 | `81.887` (Panthor) | 0.50× (慢 2.02×) |

#### 表 2：同进程综合评测 (`bench --backend all`) 稳态性能指标

| Batch Size | 后端版本 | Mean 批次延迟 (ms) | P50 延迟 (ms) | P95 延迟 (ms) | P99 延迟 (ms) | 单样本延迟 (ms/sample) | 吞吐量 (samples/sec) | NPU / GPU 硬件引擎耗时 (ms/batch) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **NPU (`RKNPU INT8`)** | **0.166** | **0.162** | **0.183** | **0.256** | **0.1662** | **6,016.8** | `0.089` (RKNPU) |
| **1** | **NPU (`RKNPU FP16`)** | **0.183** | **0.183** | **0.185** | **0.189** | **0.1832** | **5,459.1** | `0.113` (RKNPU) |
| **1** | **CPU (`Burn Flex`)** | 0.990 | 0.977 | 1.213 | 1.384 | 0.9897 | 1,010.4 | `0.000` |
| **1** | **GPU (`WGPU Vulkan`)** | 3.233 | 3.062 | 3.589 | 3.663 | 3.2333 | 309.3 | `4.398` (Panthor) |
| **1** | **GPU (`WGPU OpenGL ES`)** | 5.840 | 5.800 | 6.136 | 6.258 | 5.8399 | 171.2 | `1.786` (Panthor) |
| **16** | **NPU (`RKNPU INT8`)** | **0.805** | **0.790** | **1.011** | **1.022** | **0.0503** | **19,879.7** | `1.011` (3-Core Σ) |
| **16** | **NPU (`RKNPU FP16`)** | **1.175** | **1.142** | **1.354** | **1.418** | **0.0735** | **13,612.4** | `1.560` (3-Core Σ) |
| **16** | **CPU (`Burn Flex`)** | 3.809 | 3.792 | 4.600 | 5.129 | 0.2381 | 4,200.7 | `0.000` |
| **16** | **GPU (`WGPU Vulkan`)** | 23.421 | 23.444 | 23.691 | 23.817 | 1.4638 | 683.1 | `64.361` (Panthor) |
| **16** | **GPU (`WGPU OpenGL ES`)** | 28.385 | 28.182 | 30.452 | 31.818 | 1.7740 | 563.7 | `23.167` (Panthor) |
| **64** | **NPU (`RKNPU INT8`)** | **2.771** | **2.769** | **2.919** | **2.941** | **0.0433** | **23,093.2** | `2.981` (3-Core Σ) |
| **64** | **NPU (`RKNPU FP16`)** | **2.928** | **2.938** | **2.997** | **3.000** | **0.0457** | **21,859.0** | `5.009` (3-Core Σ) |
| **64** | **CPU (`Burn Flex`)** | 13.924 | 14.213 | 16.368 | 17.676 | 0.2176 | 4,596.3 | `0.000` |
| **64** | **GPU (`WGPU Vulkan`)** | 33.110 | 33.122 | 33.424 | 33.780 | 0.5173 | 1,932.9 | `93.164` (Panthor) |
| **64** | **GPU (`WGPU OpenGL ES`)** | 36.813 | 36.486 | 39.075 | 40.352 | 0.5752 | 1,738.5 | `31.010` (Panthor) |
| **256** | **NPU (`RKNPU INT8`)** | **7.139** | **7.150** | **7.197** | **7.215** | **0.0279** | **35,859.8** | `10.896` (3-Core Σ) |
| **256** | **NPU (`RKNPU FP16`)** | **9.915** | **9.931** | **9.959** | **10.003** | **0.0387** | **25,818.5** | `18.543` (3-Core Σ) |
| **256** | **CPU (`Burn Flex`)** | 42.677 | 44.377 | 47.871 | 49.332 | 0.1667 | 5,998.6 | `0.000` |
| **256** | **GPU (`WGPU Vulkan`)** | 83.276 | 83.333 | 83.592 | 83.722 | 0.3253 | 3,074.1 | `243.260` (Panthor) |
| **256** | **GPU (`WGPU OpenGL ES`)** | 89.788 | 89.848 | 93.347 | 93.636 | 0.3507 | 2,851.1 | `81.897` (Panthor) |

---

### 4.3 20 张板载手写数字 PNG 逐样本稳态推理详情

下表展示了 20 张板载测试图片（`/usr/share/digit-recognizer/images/d*.png`）在完成单样本预热后的预测数字、Softmax 置信度与单张端到端稳态推理延迟（数据来自综合评测 [`artifacts/rk3588_benchmark_results.json`](file:///Users/a/workspace/ubuntu/burn-mnist-rk3588/artifacts/rk3588_benchmark_results.json)）：

| 样本文件名 | 真实标签 | 预测结果 | Softmax 置信度 (FP32 / NPU FP16 / NPU INT8) | NPU INT8 延迟 (ms) | NPU FP16 延迟 (ms) | CPU 延迟 (ms) | GPU Vulkan 延迟 (ms) | GPU GLES 延迟 (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `d0_0003.png` | 0 | **0** ✓ | `0.999964` / `0.999964` / `0.999966` | 0.281 | 0.300 | 1.195 | 3.425 | 7.216 |
| `d0_0010.png` | 0 | **0** ✓ | `0.999988` / `0.999988` / `0.999989` | 0.194 | 0.279 | 1.048 | 3.297 | 6.405 |
| `d1_0002.png` | 1 | **1** ✓ | `0.999622` / `0.999622` / `0.999614` | 0.181 | 0.278 | 1.168 | 3.348 | 6.398 |
| `d1_0005.png` | 1 | **1** ✓ | `0.999867` / `0.999867` / `0.999868` | 0.246 | 0.217 | 1.137 | 3.674 | 6.411 |
| `d2_0001.png` | 2 | **2** ✓ | `0.999978` / `0.999978` / `0.999980` | 0.258 | 0.202 | 0.823 | 3.320 | 6.258 |
| `d2_0035.png` | 2 | **2** ✓ | `0.999994` / `0.999994` / `0.999994` | 0.195 | 0.268 | 1.111 | 3.219 | 6.218 |
| `d3_0018.png` | 3 | **8** ✗ | `0.943886` / `0.944040` / `0.941026` | 0.184 | 0.217 | 1.027 | 3.244 | 6.245 |
| `d3_0030.png` | 3 | **3** ✓ | `1.000000` / `1.000000` / `0.999999` | 0.177 | 0.200 | 0.942 | 3.200 | 6.194 |
| `d4_0004.png` | 4 | **4** ✓ | `0.999973` / `0.999973` / `0.999977` | 0.176 | 0.202 | 0.916 | 3.140 | 6.201 |
| `d4_0006.png` | 4 | **4** ✓ | `0.993437` / `0.993478` / `0.993098` | 0.172 | 0.195 | 1.087 | 3.176 | 6.239 |
| `d5_0008.png` | 5 | **5** ✓ | `0.846724` / `0.847006` / `0.846673` | 0.170 | 0.193 | 0.911 | 3.203 | 6.454 |
| `d5_0015.png` | 5 | **5** ✓ | `0.999218` / `0.999215` / `0.999213` | 0.179 | 0.193 | 1.130 | 3.198 | 6.260 |
| `d6_0011.png` | 6 | **6** ✓ | `0.999985` / `0.999985` / `0.999987` | 0.178 | 0.198 | 1.285 | 3.100 | 6.177 |
| `d6_0021.png` | 6 | **6** ✓ | `0.999964` / `0.999965` / `0.999965` | 0.176 | 0.195 | 1.140 | 3.637 | 6.177 |
| `d7_0000.png` | 7 | **7** ✓ | `0.999999` / `0.999999` / `0.999999` | 0.173 | 0.195 | 1.162 | 3.277 | 6.285 |
| `d7_0017.png` | 7 | **7** ✓ | `0.999993` / `0.999993` / `0.999993` | 0.178 | 0.193 | 1.101 | 3.223 | 6.064 |
| `d8_0061.png` | 8 | **8** ✓ | `1.000000` / `1.000000` / `1.000000` | 0.176 | 0.193 | 1.217 | 3.232 | 6.163 |
| `d8_0084.png` | 8 | **8** ✓ | `0.999995` / `0.999995` / `0.999995` | 0.172 | 0.193 | 1.198 | 3.176 | 6.220 |
| `d9_0007.png` | 9 | **9** ✓ | `0.999517` / `0.999517` / `0.999492` | 0.171 | 0.194 | 1.157 | 3.148 | 6.207 |
| `d9_0009.png` | 9 | **9** ✓ | `0.999399` / `0.999397` / `0.999353` | 0.176 | 0.193 | 1.053 | 3.195 | 6.204 |

---

## 5. 深入性能差异分析与工程结论

### 5.1 为什么 RK3588 NPU (`RKNPU`) 在所有 Batch Size 下均大幅领先 CPU 与 GPU？
1. **专用 MAC 阵列与离线整图编译 (`RKNN-Toolkit2`)**：
   -不同于 GPU (`Burn WGPU`) 在运行时按算子逐层派发通用 Compute Shader，RKNN 编译器将 `Conv2d + ReLU + MaxPool2d + Linear` 离线融合并直接映射至 RKNPU 的专用矩阵乘加（MAC）脉动阵列与片上 SRAM。
   - 在 `batch=1` 时，RKNPU 硬件纯计算耗时仅 **`0.089 ms` (INT8)** / **`0.113 ms` (FP16)**，包含 Rust 端输入/输出拷贝与 Softmax 后的端到端延迟仅 **`0.163 ms` (INT8)** / **`0.184 ms` (FP16)**，比 8 核 Cortex-A76/A55 CPU (`0.982 ms`) 快 **5.3× ~ 6.0×**，比 Mali-G610 GPU Vulkan (`3.116 ms`) 快 **16.9× ~ 19.1×**。
2. **三核 NPU (`Core 0/1/2`) 硬件级并行扩展性**：
   - RK3588 拥有 3 个独立的 NPU 核心（Core 0、Core 1、Core 2）。本项目通过 **RKNB 多批次离线模型容器** 结合 Rayon 线程池将大 Batch 切片并行分发至 3 个 NPU 物理核心同时执行：
   - 在 `batch=256` 下，NPU INT8 吞吐量高达 **`35,975.4 samples/s`**（单张均摊仅 **`0.0278 ms / 27.8 µs`**），NPU FP16 吞吐量达 **`25,818.5 samples/s`**（综合评测），分别是 8 核 CPU 峰值吞吐量 (`5,998.6 samples/s`) 的 **6.00 倍**与 **4.30 倍**，是 Mali-G610 GPU Vulkan (`3,074.1 samples/s`) 的 **11.70 倍**与 **8.40 倍**。
   - 10,000 张 MNIST 全量测试集在 NPU INT8 上仅需 **`280.12 ms`** 完成全部推理，在 NPU FP16 上仅需 **`388.41 ms`**，而 CPU 需 `1,647.67 ms`、GPU Vulkan 需 `3,336.91 ms`。

### 5.2 为什么在轻量级 MNIST CNN 上，CPU (`Burn Flex`) 比 GPU (`Burn WGPU`) 更快？
1. **模型计算密度与 Cache 局部性**：
   - `MnistCnn` 输入仅 `1×28×28`，全部权重仅 `426 KB`，单次前向推理计算量约 `1.4 MFLOPs`。
   - RK3588 拥有 **4× Cortex-A76（每核 512 KB L2）+ 3 MB DynamIQ L3 共享缓存**，`426 KB` 模型权重与中间激活张量完全常驻于 CPU L2/L3 高速缓存中。`Burn Flex` 通过 ARMv8.2 NEON SIMD 与 `Rayon` 线程池直接在缓存内完成计算，单张推理仅需 **`0.982 ms`**。
2. **GPU (`Burn WGPU`) 的驱动调度与 Staging Buffer 回读开销**：
   - `CubeCL` / `Burn WGPU` 首次运行需动态生成并编译 WGSL/SPIR-V Compute Shader（单样本预热 `~121 ms`）；在 `batch=1` 稳态下，多层算子命令队列提交与 `wgpu` Staging Buffer 映射回读开销（约 `2~3 ms`）大于算子本身的计算时间。

### 5.3 部署建议与当前板端默认配置
1. **板端生产与触控 UI 服务 (`digit-recognizer.service`) — 默认启用 NPU 版本 (`digit-recognizer-npu`)**：
   - 兼具 **最低延迟 (`~0.18 ms` 基准 / `~0.61 ms` 含图像预处理)**、**最小内存占用 (`102.1 MB` vs GPU 版 `196.1 MB`)**、**最小二进制体积 (`17 MB` vs GPU 版 `97 MB`)** 以及 **零 CPU/GPU 算力争用**（CPU 与 Mali-G610 GPU 完全留给 Slint LinuxKMS / FemtoVG 进行 60 FPS UI 渲染）。
2. **高吞吐量批量图像流处理 (`batch=16 ~ 256`)**：
   - 优先使用 **NPU INT8 (`--backend npu-int8`)** 或 **NPU FP16 (`--backend npu`)**，可获得 **`20,000 ~ 35,975 samples/sec`** 的三核并发吞吐能力。
3. **无 NPU 驱动的通用 ARM64 Linux 环境**：
   - 推荐使用 **CPU 版本 (`mnist-cpu` / `Burn Flex SIMD + Rayon`)**，无需任何专有驱动即可获得 `< 1 ms` 的单张实时响应与 `~6,000 samples/sec` 的批量吞吐。
