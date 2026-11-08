# burn-mnist-rk3588

Rust [Burn](https://burn.dev) (`0.22.1`) MNIST handwritten digit recognition training, runtime caching, and RK3588 CPU (`Flex` NEON SIMD + Rayon) vs. GPU (`WGPU` Vulkan & OpenGL ES compute on Mali-G610 `panthor`) deployment & benchmark framework.

## Repository Layout

- `src/model.rs`: `MnistCnn` 2D Convolutional Neural Network (`Conv2d -> ReLU -> MaxPool2d -> Conv2d -> ReLU -> MaxPool2d -> Linear -> ReLU -> Linear`), runtime device resolver (`cpu`, `gpu`, `vulkan`, `gl`), `BurnpackStore` / `SafetensorsStore` weight serialization, and RK3588 `panthor` DRM GPU hardware timer integration.
- `src/data.rs`: MNIST IDX gzipped dataset loader and pure-Rust PNG 28×28 box-downsampling preprocessor.
- `src/bin/train.rs`: Training binary with Adam optimizer, cross-entropy loss, epoch validation on the 10,000-sample MNIST test split, and automatic export of `mnist_cnn.bpk`, `mnist_cnn.safetensors`, and `training_metrics.json`.
- `src/bin/bench.rs`: Hardware benchmark binary comparing CPU (`Flex`) and GPU (`WGPU` Vulkan & OpenGL ES) latency, throughput across batch sizes (`1, 16, 64, 256`), accuracy, and `/proc/<pid>/fdinfo` `drm-engine-panthor` GPU nanoseconds.
- `notebooks/colab_burn_mnist_train.ipynb` & `scripts/colab_runtime_train.py` & `scripts/colab_runtime_train.sh`: Runtime scripts that check Google Drive for cached Rust Burn environment and MNIST dataset, migrate from Drive (or download and cache back to Drive), clone this repository, train the model, and export the trained artifacts.
- `scripts/sync_to_ubuntu_bsp.sh`: Transfers trained model artifacts and crate sources to the Ubuntu server's `last-rk3588/atk-dlrk3588-bsp` repository.

## Quick Start

### 1. Run Colab / Runtime Training Script
```bash
./scripts/colab_runtime_train.sh
```

### 2. Sync Trained Artifacts to Ubuntu Server BSP Repository
```bash
./scripts/sync_to_ubuntu_bsp.sh
```
