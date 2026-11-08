# burn-mnist-rk3588

Rust [Burn](https://burn.dev) (`0.22.1`) MNIST handwritten digit recognition training, runtime caching, and RK3588 CPU (`Flex` NEON SIMD + Rayon), GPU (`WGPU` Vulkan & OpenGL ES compute on Mali-G610 `panthor`), and NPU (`RKNN` 3-Core FP16 & INT8 on `/dev/rknpu`) deployment & benchmark framework. On the RK3588 board, `/usr/local/bin/digit-recognizer` (`digit-recognizer.service`) runs the **NPU version (`digit-recognizer-npu`, `DIGIT_RECOGNIZER_BACKEND=npu`, `NPU · RKNPU`)** by default.

## Repository Layout

- `src/model.rs`: `MnistCnn` 2D Convolutional Neural Network (`Conv2d -> ReLU -> MaxPool2d -> Conv2d -> ReLU -> MaxPool2d -> Linear -> ReLU -> Linear`), runtime device resolver (`cpu`, `gpu`, `vulkan`, `gl`, `npu`, `npu-int8`), `BurnpackStore` / `SafetensorsStore` weight serialization, and RK3588 `panthor` DRM GPU hardware timer integration.
- `src/npu.rs`: Zero-copy dynamic `librknnrt.so` FFI bindings and 3-core concurrent NPU execution engine (`RknnEngine`) supporting multi-batch `RKNB` bundles (`batch=1, 6, 22, 86` across NPU Core 0/1/2) in both FP16 and INT8 modes.
- `drivers/rknpu.c`: Out-of-tree mainline Linux 7.x `rknpu.ko` kernel driver (`v0.9.6`) enabling all 3 RK3588 NPU cores (`npu@fdab0000`, `npu@fdac0000`, `npu@fdad0000` at 800 MHz), exposing `/dev/rknpu` (`0666`) and `/sys/module/rknpu/parameters/hw_busy_ns` hardware busy-time telemetry.
- `src/data.rs`: MNIST IDX gzipped dataset loader and pure-Rust PNG 28×28 box-downsampling preprocessor.
- `src/bin/train.rs`: Training binary with Adam optimizer, cross-entropy loss, epoch validation on the 10,000-sample MNIST test split, and automatic export of `mnist_cnn.bpk`, `mnist_cnn.safetensors`, and `training_metrics.json`.
- `src/bin/mnist-cpu.rs`: Dedicated RK3588 CPU inference & benchmark binary (`--no-default-features --features cpu`, `Burn Flex` SIMD + Rayon).
- `src/bin/mnist-gpu.rs`: Dedicated RK3588 GPU inference & benchmark binary (`--no-default-features --features gpu`, `Burn WGPU` Vulkan & OpenGL ES on Mali-G610 `panthor`).
- `src/bin/mnist-npu.rs`: Dedicated RK3588 NPU inference & benchmark binary (`--no-default-features --features npu`, 3-Core RKNN FP16 & INT8).
- `src/bin/bench.rs`: Unified hardware benchmark binary comparing CPU (`Flex`), GPU (`WGPU` Vulkan & OpenGL ES), and NPU (`RKNN` FP16 & INT8) latency, throughput across batch sizes (`1, 16, 64, 256`), accuracy, `/proc/<pid>/fdinfo` `drm-engine-panthor` GPU nanoseconds, and `/sys/module/rknpu/parameters/hw_busy_ns` NPU nanoseconds.
- `notebooks/colab_burn_mnist_train.ipynb` & `scripts/colab_runtime_train.py` & `scripts/colab_runtime_train.sh`: Runtime scripts that check Google Drive for cached Rust Burn environment and MNIST dataset, migrate from Drive (or download and cache back to Drive), clone this repository, train the model, and export the trained artifacts.
- `scripts/sync_to_ubuntu_bsp.sh`: Transfers trained model artifacts and crate sources to the Ubuntu server's `last-rk3588/atk-dlrk3588-bsp` repository, cross-compiles CPU/GPU/NPU binaries, deploys to the RK3588 board while keeping `/usr/local/bin/digit-recognizer` (`digit-recognizer.service`) on the **NPU version (`digit-recognizer-npu`, `DIGIT_RECOGNIZER_BACKEND=npu`, `NPU · RKNPU`)**, and runs 3-way benchmarks.
- `REPORT_RK3588_CPU_VS_GPU.md` & `artifacts/rk3588_benchmark_results.json`: Detailed RK3588 CPU vs GPU vs NPU deployment verification and 3-way benchmark comparison report.

## Quick Start

### 1. Run Colab / Runtime Training Script
```bash
./scripts/colab_runtime_train.sh
```

### 2. Sync Trained Artifacts to Ubuntu Server BSP Repository & Deploy to RK3588 Board
```bash
./scripts/sync_to_ubuntu_bsp.sh --deploy-and-bench
```
