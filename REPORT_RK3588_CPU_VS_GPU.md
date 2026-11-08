# RK3588 Rust Burn 手写数字识别 (MNIST CNN) CPU 与 GPU 双版本部署及性能对比报告

## 1. 概述与架构设计

本项目基于 **Rust + Burn 0.22.1** 构建了端到端的手写数字图像识别（MNIST 28×28 CNN）训练、跨平台模型序列化与 **Rockchip RK3588** 边缘侧 CPU / GPU 双版本推理部署框架：

1. **模型结构 (`MnistCnn`)**：
   - `conv1`: `Conv2d(1 -> 16, kernel=3×3, padding=Valid)` + `ReLU` + `MaxPool2d(2×2, stride=2)`（输出 `16×13×13`）
   - `conv2`: `Conv2d(16 -> 32, kernel=3×3, padding=Valid)` + `ReLU` + `MaxPool2d(2×2, stride=2)`（输出 `32×5×5`）
   - `fc1`: `Linear(800 -> 128)` + `ReLU`
   - `fc2`: `Linear(128 -> 10)` + `Softmax`
   - 参数规模：约 `107,786` 个 `f32` 参数（`mnist_cnn.bpk` 权重文件大小 `435,752 B` / `426 KB`，SHA1 `71abbdd3c48860c9a4fd59d879c7328bc597ce9a`；`mnist_cnn.safetensors` 大小 `435,136 B` / `425 KB`，SHA1 `1f0033c3067f60b085e1e19a66b044bf3c10dba6`）。
2. **双版本编译与部署架构**：
   - **CPU 独立版本 (`mnist-cpu` / `digit-recognizer-cpu`)**：仅启用 `--no-default-features --features cpu`，底层采用 **Burn Flex CPU (`burn/flex` + `burn/simd` + `burn/rayon`)** 后端，利用 RK3588 的 4× Cortex-A76 + 4× Cortex-A55 八核 ARMv8.2 NEON SIMD 与多线程并行加速，不链接任何 GPU/WGPU 依赖（二进制体积仅 `13 MB`）。
   - **GPU 独立版本 (`mnist-gpu` / `digit-recognizer-gpu`)**：仅启用 `--no-default-features --features gpu`，底层采用 **Burn WGPU (`burn/wgpu` + `CubeCL`)** 后端，针对 RK3588 内置 **ARM Mali-G610 MP4 GPU**（主线 Linux `7.2.5-atk-dlrk3588+` 内核 `panthor` DRM 驱动，支持 **Vulkan (Mesa PanVK)** 与 **OpenGL ES 3.1 (Mesa Panfrost)** 双图形计算管线，二进制体积 `83 MB`）。
   - **综合对比评测与 UI 展示版本 (`bench` / `digit-recognizer`)**：同时启用 `cpu` 与 `gpu` 特性，支持同一进程内对比 CPU、GPU (Vulkan)、GPU (OpenGL ES) 三个后端的精度、预热耗时、稳态延迟与硬件计数器指标，并将训练好的真实模型接入 BSP 的 `app/digit-recognizer` LinuxKMS/FemtoVG 展示界面（`model_live() == true`）。

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
  - `artifacts/mnist_cnn.bpk` (`426 KB`, SHA1: `71abbdd3c48860c9a4fd59d879c7328bc597ce9a`)
  - `artifacts/mnist_cnn.safetensors` (`425 KB`, SHA1: `1f0033c3067f60b085e1e19a66b044bf3c10dba6`)
  - `artifacts/training_metrics.json` (`1.6 KB`)

---

## 3. RK3588 硬件环境与双版本部署验证

### 3.1 目标开发板硬件与内核环境
- **开发板**：ATK-DLRK3588 (`ubuntu@10.42.0.88`)
- **SoC**：Rockchip RK3588（4× Cortex-A76 @ 2.4GHz + 4× Cortex-A55 @ 1.8GHz，3MB 共享 DynamIQ L3 Cache）
- **GPU**：ARM Mali-G610 MP4（4 Shader Cores）
- **内核与驱动**：`Linux atk-rk3588 7.2.5-atk-dlrk3588+ #5 SMP PREEMPT aarch64`，主线 DRM `panthor` 驱动（计算节点 `/dev/dri/renderD128`，显示节点 `/dev/dri/card*`）
- **GPU 硬件性能计数器**：已开启 `/sys/bus/platform/devices/fb000000.gpu/profiling = 3`，通过解析 `/proc/<pid>/fdinfo/*` 中的 `drm-engine-panthor: <ns> ns` 并按 `drm-client-id` 去重（避免 Mesa `PanVK`/`Panfrost` 进程内复制 render node 文件描述符导致重复累加），精确采集 Mali-G610 硬件引擎执行时间。

### 3.2 板端双版本独立部署路径与零长黑屏热更新机制
通过 [`scripts/sync_to_ubuntu_bsp.sh`](file:///Users/a/workspace/ubuntu/burn-mnist-rk3588/scripts/sync_to_ubuntu_bsp.sh) 将训练产物与源码同步至 Ubuntu 构建服务器 (`192.168.10.117:/home/a/workspace/rk3588/last-rk3588/atk-dlrk3588-bsp`) 交叉编译后，部署至 RK3588 开发板：

| 版本类别 | 板端部署路径 | 编译特性 (`Cargo Features`) | 二进制体积 | 运行时后端描述 |
| :--- | :--- | :--- | :---: | :--- |
| **CPU 独立推理版** | `/opt/burn-mnist-rk3588/cpu/mnist-cpu`<br>`/usr/local/bin/mnist-cpu` | `--no-default-features --features cpu` | 13 MB | `Burn Flex CPU (SIMD + Rayon)` |
| **GPU 独立推理版** | `/opt/burn-mnist-rk3588/gpu/mnist-gpu`<br>`/usr/local/bin/mnist-gpu` | `--no-default-features --features gpu` | 83 MB | `Burn WGPU (Vulkan)` & `Burn WGPU (OpenGL ES)` |
| **UI 展示 CPU 版** | `/opt/burn-mnist-rk3588/cpu/digit-recognizer-cpu`<br>`/usr/local/bin/digit-recognizer-cpu` | `--no-default-features --features cpu` | 21 MB | `Burn Flex CPU (SIMD + Rayon)` + Slint LinuxKMS |
| **UI 展示 GPU 版** | `/opt/burn-mnist-rk3588/gpu/digit-recognizer-gpu`<br>`/usr/local/bin/digit-recognizer-gpu` | `--no-default-features --features gpu` | 91 MB | `Burn WGPU (Mali-G610)` + Slint LinuxKMS |
| **综合基准测试工具** | `/opt/burn-mnist-rk3588/bench` | `--features cpu,gpu` | 92 MB | 同进程评测 `cpu` / `vulkan` / `gl` |
| **模型与测试数据集** | `/opt/burn-mnist-rk3588/model/mnist_cnn.bpk`<br>`/usr/share/digit-recognizer/model/mnist_cnn.bpk`<br>`/opt/burn-mnist-rk3588/data/mnist/t10k-*` | — | 2.5 MB | 权重 + 10,000 张测试集 + 20 张板载 PNG 样本 |

- **显示服务 (`digit-recognizer.service`) 保护与 GPU 并发说明**：
  1. **最小化黑屏时间（实测仅 `~1.43 s`）**：交叉编译（Phase 1）与文件网络传输预暂存（Phase 2，上传至 `/opt/burn-mnist-rk3588/staging`）全程保持 `digit-recognizer.service` 运行；仅在板端本地执行 `mv -f` 原子替换二进制与模型（Phase 3）的瞬间停止并立即重启服务，黑屏切换耗时从原来的数分钟缩短至 **`1,430 ms`**（同时修复了 `app/digit-recognizer/src/main.rs` 中 `slint::Timer` 临时对象提前析构导致 `SIGTERM` 10 秒超时的 bug，使服务停机在 `< 250 ms` 内正常退出）。
  2. **自动回滚与多层 `trap` 兜底**：替换前自动备份已知正常的二进制与模型至 `/var/backups/digit-recognizer-last/`；若新版本启动失败或未在 `journalctl` 输出 `Rendering at 1080x1920` / `GPU_SCANOUT:`，脚本在 `~2 s` 内自动回滚至备份版本并恢复显示。本地脚本、服务器端与板端 `board-service-guard.sh` 均注册了 `EXIT/INT/TERM/HUP` `trap`，中途失败或被中断时均自动恢复 `digit-recognizer.service`。
  3. **基准测试期间无需停止显示服务**：`mnist-gpu` 与 `bench` 通过 DRM Render Node (`/dev/dri/renderD128`) 提交 `PanVK` (Vulkan) 与 Surfaceless EGL `Panfrost` (OpenGL ES) 计算队列，无需独占 DRM KMS Master (`/dev/dri/card*`)；且 `digit-recognizer.service` 在无触控交互时完全静默（0% CPU、0% GPU 占用），因此默认在保持屏幕正常显示的同时完成全部基准测试（若传入可选参数 `--stop-display-for-bench` 独占测试，测试结束后也会自动恢复并校验渲染日志）。

---

## 4. RK3588 CPU vs GPU 性能对比评测结果

所有原始基准测试数据保存在：
- 独立 CPU 版本结果：[`artifacts/rk3588_cpu_benchmark.json`](file:///Users/a/workspace/ubuntu/burn-mnist-rk3588/artifacts/rk3588_cpu_benchmark.json)
- 独立 GPU 版本结果：[`artifacts/rk3588_gpu_benchmark.json`](file:///Users/a/workspace/ubuntu/burn-mnist-rk3588/artifacts/rk3588_gpu_benchmark.json)
- 全后端综合对比结果：[`artifacts/rk3588_benchmark_results.json`](file:///Users/a/workspace/ubuntu/burn-mnist-rk3588/artifacts/rk3588_benchmark_results.json)

### 4.1 整体准确率、预热开销与全量测试集 (10,000 张) 推理耗时对比

评测代码 ([`src/bench.rs`](file:///Users/a/workspace/ubuntu/burn-mnist-rk3588/src/bench.rs)) 将首次计算图/着色器编译预热耗时（`single_sample_warmup_ms` 与 `test_set_warmup_ms`）与稳态推理耗时（`test_set_eval_ms` 及 PNG 样本稳态延迟）进行了严格分离：

| 评测维度 | CPU 版本 (`mnist-cpu`<br>Burn Flex SIMD+Rayon) | GPU Vulkan 版本 (`mnist-gpu`<br>Burn WGPU Vulkan / PanVK) | GPU OpenGL ES 版本 (`mnist-gpu`<br>Burn WGPU GLES / Panfrost) |
| :--- | :---: | :---: | :---: |
| **模型加载耗时 (`model_load_ms`)** | 0.561 ms (综合 0.323 ms) | **0.186 ms** (综合 **0.159 ms**) | 0.190 ms (综合 0.205 ms) |
| **单样本 `[1,1,28,28]` 首次预热耗时 (`single_sample_warmup_ms`)** | **7.198 ms** (综合 18.393 ms) | 115.838 ms (综合 115.528 ms) | 139.342 ms (综合 144.024 ms) |
| **MNIST 10,000 张测试集准确率** | **98.66% (9,866 / 10,000)** | **98.66% (9,866 / 10,000)** | **98.66% (9,866 / 10,000)** |
| **MNIST 10,000 张测试集预热耗时 (`test_set_warmup_ms`)** | **55.78 ms** (综合 52.91 ms) | 485.55 ms (综合 489.64 ms) | 502.17 ms (综合 540.29 ms) |
| **MNIST 10,000 张测试集稳态总耗时 (`test_set_eval_ms`)** | **1,667.72 ms** (综合 1,760.91 ms) | 3,365.39 ms (综合 3,358.21 ms) | 3,545.77 ms (综合 3,519.93 ms) |
| **20 张板载手写数字 PNG 准确率** | **95.00% (19 / 20)** | **95.00% (19 / 20)** | **95.00% (19 / 20)** |
| **20 张 PNG 稳态平均单张延迟** | **1.030 ms** (综合 1.039 ms) | 3.271 ms (综合 3.191 ms) | 6.781 ms (综合 6.428 ms) |

> **精度一致性验证**：CPU (`Burn Flex`)、GPU Vulkan (`PanVK`)、GPU OpenGL ES (`Panfrost`) 三个后端在 10,000 张 MNIST 测试集上均取得完全一致的 **98.66% (9,866/10,000)** 准确率（与 Colab 训练指标 `final_test_accuracy = 0.9866` 完全吻合），在 20 张板载 PNG 样本上预测结果与每张图片的 Softmax 置信度在小数点后 6 位完全一致（例如 `d0_0003.png` 在三者上均为 `0.9999638`）。

---

### 4.2 不同 Batch Size (1, 16, 64, 256) 稳态推理延迟、吞吐量与硬件 GPU 计数器对比

每个 Batch Size 在完成预热迭代（Warmup，消除首次管线编译开销）后，连续测量稳态前向推理 + 结果同步 (`forward_probabilities` + `to_data`) 的延迟分布与去重后的硬件级 `drm-engine-panthor` 计数器：

#### 表 1：独立部署版本 (`mnist-cpu` vs `mnist-gpu`) 稳态性能指标

| Batch Size | 后端版本 | Mean 批次延迟 (ms) | P50 延迟 (ms) | P95 延迟 (ms) | P99 延迟 (ms) | 单样本均摊延迟 (ms/sample) | 吞吐量 (samples/sec) | `panthor` GPU 硬件耗时 (ms/batch) | CPU 相对加速比 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **CPU (`Burn Flex`)** | **0.972** | **1.027** | **1.215** | **1.390** | **0.9724** | **1,028.4** | 0.000 | **1.00× (基准最快)** |
| **1** | **GPU (`WGPU Vulkan`)** | 2.991 | 2.966 | 3.114 | 3.452 | 2.9912 | 334.3 | 4.465 | 0.33× (慢 3.08×) |
| **1** | **GPU (`WGPU OpenGL ES`)** | 5.784 | 5.764 | 6.082 | 6.265 | 5.7839 | 172.9 | 1.781 | 0.17× (慢 5.95×) |
| **16** | **CPU (`Burn Flex`)** | **3.875** | **3.845** | **4.368** | **4.667** | **0.2422** | **4,128.9** | 0.000 | **1.00× (基准最快)** |
| **16** | **GPU (`WGPU Vulkan`)** | 23.351 | 23.293 | 23.897 | 24.058 | 1.4595 | 685.2 | 64.357 | 0.17× (慢 6.03×) |
| **16** | **GPU (`WGPU OpenGL ES`)** | 28.169 | 28.164 | 28.597 | 28.725 | 1.7606 | 568.0 | 23.100 | 0.14× (慢 7.27×) |
| **64** | **CPU (`Burn Flex`)** | **13.793** | **14.166** | **16.543** | **16.782** | **0.2155** | **4,640.1** | 0.000 | **1.00× (基准最快)** |
| **64** | **GPU (`WGPU Vulkan`)** | 33.015 | 32.999 | 33.629 | 33.766 | 0.5159 | 1,938.5 | 93.212 | 0.42× (慢 2.39×) |
| **64** | **GPU (`WGPU OpenGL ES`)** | 36.346 | 36.306 | 36.880 | 37.121 | 0.5679 | 1,760.8 | 30.935 | 0.38× (慢 2.64×) |
| **256** | **CPU (`Burn Flex`)** | **44.128** | **45.191** | **49.066** | **51.954** | **0.1724** | **5,801.2** | 0.000 | **1.00× (基准最快)** |
| **256** | **GPU (`WGPU Vulkan`)** | 83.915 | 83.623 | 85.316 | 85.346 | 0.3278 | 3,050.7 | 245.396 | 0.53× (慢 1.90×) |
| **256** | **GPU (`WGPU OpenGL ES`)** | 89.228 | 88.914 | 90.461 | 90.547 | 0.3485 | 2,869.1 | 81.688 | 0.49× (慢 2.02×) |

#### 表 2：同进程综合评测 (`bench --backend all`) 稳态性能指标

| Batch Size | 后端版本 | Mean 批次延迟 (ms) | P50 延迟 (ms) | P95 延迟 (ms) | P99 延迟 (ms) | 单样本延迟 (ms/sample) | 吞吐量 (samples/sec) | `panthor` GPU 硬件耗时 (ms/batch) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **CPU (`Burn Flex`)** | **1.030** | **1.039** | **1.289** | **1.374** | **1.0303** | **970.6** | 0.000 |
| **1** | **GPU (`WGPU Vulkan`)** | 3.141 | 3.035 | 3.212 | 6.066 | 3.1414 | 318.3 | 4.470 |
| **1** | **GPU (`WGPU OpenGL ES`)** | 5.816 | 5.802 | 6.064 | 6.202 | 5.8162 | 171.9 | 1.803 |
| **16** | **CPU (`Burn Flex`)** | **3.836** | **3.800** | **4.454** | **4.878** | **0.2398** | **4,170.9** | 0.000 |
| **16** | **GPU (`WGPU Vulkan`)** | 23.463 | 23.389 | 24.004 | 24.201 | 1.4665 | 681.9 | 64.512 |
| **16** | **GPU (`WGPU OpenGL ES`)** | 28.094 | 28.059 | 28.715 | 28.767 | 1.7559 | 569.5 | 23.004 |
| **64** | **CPU (`Burn Flex`)** | **15.980** | **15.649** | **19.432** | **20.411** | **0.2497** | **4,005.0** | 0.000 |
| **64** | **GPU (`WGPU Vulkan`)** | 33.154 | 33.126 | 33.803 | 34.030 | 0.5180 | 1,930.4 | 93.394 |
| **64** | **GPU (`WGPU OpenGL ES`)** | 36.588 | 36.581 | 37.062 | 37.184 | 0.5717 | 1,749.2 | 31.011 |
| **256** | **CPU (`Burn Flex`)** | **46.847** | **47.351** | **51.691** | **57.663** | **0.1830** | **5,464.6** | 0.000 |
| **256** | **GPU (`WGPU Vulkan`)** | 83.668 | 83.629 | 84.249 | 84.255 | 0.3268 | 3,059.7 | 244.389 |
| **256** | **GPU (`WGPU OpenGL ES`)** | 89.163 | 89.414 | 89.993 | 90.676 | 0.3483 | 2,871.1 | 82.160 |

---

### 4.3 20 张板载手写数字 PNG 逐样本稳态推理详情

下表展示了 20 张板载测试图片（`/usr/share/digit-recognizer/images/d*.png`）在完成单样本预热后的预测数字、Softmax 置信度与单张端到端稳态推理延迟（数据来自综合评测 `rk3588_benchmark_results.json`）：

| 样本文件名 | 真实标签 | 预测结果 | Softmax 置信度 | CPU 稳态延迟 (ms) | GPU Vulkan 稳态延迟 (ms) | GPU OpenGL ES 稳态延迟 (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `d0_0003.png` | 0 | **0** ✓ | 100.00% (`0.999964`) | 1.135 | 3.515 | 6.791 |
| `d0_0010.png` | 0 | **0** ✓ | 100.00% (`0.999988`) | 1.135 | 3.283 | 6.368 |
| `d1_0002.png` | 1 | **1** ✓ | 99.96% (`0.999622`) | 0.967 | 3.028 | 6.551 |
| `d1_0005.png` | 1 | **1** ✓ | 99.99% (`0.999867`) | 0.940 | 3.128 | 6.434 |
| `d2_0001.png` | 2 | **2** ✓ | 100.00% (`0.999978`) | 0.967 | 3.246 | 6.518 |
| `d2_0035.png` | 2 | **2** ✓ | 100.00% (`0.999994`) | 0.646 | 3.286 | 6.251 |
| `d3_0018.png` | 3 | **8** ✗ | 94.39% (`0.943886`) | 1.138 | 3.175 | 6.428 |
| `d3_0030.png` | 3 | **3** ✓ | 100.00% (`1.000000`) | 0.951 | 3.120 | 6.224 |
| `d4_0004.png` | 4 | **4** ✓ | 100.00% (`0.999973`) | 1.030 | 3.088 | 6.404 |
| `d4_0006.png` | 4 | **4** ✓ | 99.34% (`0.993437`) | 1.101 | 3.694 | 6.378 |
| `d5_0008.png` | 5 | **5** ✓ | 84.67% (`0.846724`) | 1.198 | 3.166 | 6.229 |
| `d5_0015.png` | 5 | **5** ✓ | 99.92% (`0.999218`) | 1.168 | 3.006 | 6.514 |
| `d6_0011.png` | 6 | **6** ✓ | 100.00% (`0.999985`) | 1.120 | 3.558 | 6.430 |
| `d6_0021.png` | 6 | **6** ✓ | 100.00% (`0.999964`) | 0.641 | 3.325 | 6.262 |
| `d7_0000.png` | 7 | **7** ✓ | 100.00% (`0.999999`) | 0.977 | 2.973 | 6.562 |
| `d7_0017.png` | 7 | **7** ✓ | 100.00% (`0.999993`) | 1.008 | 3.035 | 6.314 |
| `d8_0061.png` | 8 | **8** ✓ | 100.00% (`1.000000`) | 0.971 | 3.048 | 6.498 |
| `d8_0084.png` | 8 | **8** ✓ | 100.00% (`0.999995`) | 1.181 | 3.051 | 6.517 |
| `d9_0007.png` | 9 | **9** ✓ | 99.95% (`0.999517`) | 1.315 | 3.047 | 6.409 |
| `d9_0009.png` | 9 | **9** ✓ | 99.94% (`0.999399`) | 1.199 | 3.054 | 6.483 |

---

## 5. 深入性能差异分析与工程结论

### 5.1 为什么在轻量级 MNIST CNN 上，RK3588 CPU (`Burn Flex`) 比 GPU (`Burn WGPU`) 更快？
1. **模型计算密度与 Cache 局部性**：
   - `MnistCnn` 输入仅 `1×28×28`，全部权重仅 `426 KB`，单次前向推理计算量约 `1.4 MFLOPs`。
   - RK3588 拥有 **4× Cortex-A76（每核 512 KB L2）+ 3 MB DynamIQ L3 共享缓存**，整个 `426 KB` 模型权重与中间激活张量完全常驻于 CPU L2/L3 高速缓存中。`Burn Flex` 通过 `gemm`（ARMv8.2 NEON SIMD 向量化矩阵乘）与 `Rayon` 线程池直接在缓存内完成计算，单张推理仅需 **`0.972 ms`**（吞吐量 **`1,028.4 samples/s`**）。
2. **GPU (`Burn WGPU`) 的调度、同步与着色器编译开销**：
   - **首次管线编译开销（Warmup）**：`CubeCL` / `Burn WGPU` 会根据算子与张量维度动态生成 WGSL/SPIR-V Compute Shader 并交由 Mesa `PanVK` (Vulkan) 或 `Panfrost` (OpenGL ES) 编译。单样本 `[1,1,28,28]` 首次预热耗时为 `115.8 ms` (Vulkan) / `139.3 ms` (GLES)，10k 测试集首批次 `[256,1,28,28]` + 尾批次 `[16,1,28,28]` 预热耗时为 `485.5 ms` (Vulkan) / `502.2 ms` (GLES)。一旦预热完成，10,000 张测试集的纯推理耗时立即从冷态的 `~35 s` 降至 **`3,365.4 ms` (Vulkan)** 与 **`3,545.8 ms` (GLES)**，单张 PNG 推理稳定在 **`~3.2 ms` (Vulkan)** 和 **`~6.4 ms` (GLES)**。
   - **小 Batch 下驱动提交与 Staging Buffer 回读开销**：一次 CNN 前向传播包含多层算子（2× Conv2d、2× MaxPool2d、2× Linear、3× ReLU、Softmax），每次 `to_data()` 需通过 `wgpu` Staging Buffer 将 GPU 显存映射回 CPU 读取 10 维概率。在 `batch=1` 时，CPU↔GPU 命令队列提交与同步开销（约 `2~3 ms`）远大于算子本身的计算时间。
3. **大 Batch 下 GPU 吞吐量扩展性显著高于 CPU**：
   - 当 Batch Size 从 `1` 增大到 `256` 时：
     - **CPU (`Burn Flex`)** 吞吐量从 `1,028.4 img/s` 提升至 `5,801.2 img/s`（扩展 **5.64×**）；
     - **GPU (`WGPU Vulkan`)** 吞吐量从 `334.3 img/s` 提升至 `3,050.7 img/s`（扩展 **9.13×**）；
     - **GPU (`WGPU OpenGL ES`)** 吞吐量从 `172.9 img/s` 提升至 `2,869.1 img/s`（扩展 **16.59×**）。
   - 同时，GPU 稳态尾延迟抖动极小：在 `batch=256` 时，Vulkan 的 `P50 = 83.62 ms, P99 = 85.35 ms`（P99/P50 仅相差 `2.1%`），而 CPU 的 `P50 = 45.19 ms, P99 = 51.95 ms`（受大小核调度影响相差 `15.0%`）。

### 5.2 Vulkan (`PanVK`) vs OpenGL ES (`Panfrost`) 在 Mali-G610 上的对比
- 在 `batch=1` 实时交互场景下，**Vulkan (`2.99 ms`) 比 OpenGL ES (`5.78 ms`) 快 1.93 倍**，因为 Vulkan 显式命令缓冲提交比 OpenGL ES 状态机驱动同步开销更低。
- 在 `batch=256` 大批量吞吐场景下，Vulkan (`3,050.7 img/s`) 比 OpenGL ES (`2,869.1 img/s`) 快约 `6.3%`。
- 按 `drm-client-id` 去重后的硬件计数器 `drm-engine-panthor` 显示两个 GPU 后端均真实驱动了 Mali-G610 CSF 硬件队列（CPU 模式下 `panthor_gpu_total_ms = 0.0 ms`，GPU 模式下 `panthor_gpu_per_batch_ms` 随 Batch Size 增长）。

### 5.3 部署建议
1. **单张实时交互式识别（如 `digit-recognizer` 触控界面，`batch=1`）**：推荐使用 **CPU 版本 (`Burn Flex SIMD + Rayon`)**，可获得 `< 1 ms` 的极致响应且无首次 Shader 编译延迟，同时将 Mali-G610 GPU 完全留给 Slint LinuxKMS/FemtoVG 进行 60 FPS UI 渲染。
2. **CPU 负载受限或需要确定性尾延迟（P99）的批量流水线任务**：可使用 **GPU Vulkan 版本 (`Burn WGPU Vulkan`)**，在预热后具备极低延迟抖动（P99 与 P50 偏差 `< 2.5%`）并卸载 CPU 算力。
