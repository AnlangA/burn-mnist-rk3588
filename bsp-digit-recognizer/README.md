# digit-recognizer

手写数字识别展示应用:LinuxKMS + FemtoVG(OpenGL ES 3)直接在 DRM 输出上渲染,
无需桌面环境;推理使用 `../burn-mnist-rk3588` 训练好的 Burn CNN 与 RKNN 模型
(支持 **RK3588 RKNPU FP16/INT8**、**CPU Flex SIMD+Rayon** 或 **GPU WGPU Mali-G610 Vulkan/OpenGL ES** 三种硬件后端,板端 `digit-recognizer.service` 默认启用 **NPU** 后端)。

## 界面

布局随输出方向自适应(`root.height >= root.width` 视为竖屏):

- **竖屏**(板载 DSI-1 1080×1920,`SLINT_SCALE_FACTOR=2` → 540×960 逻辑像素):
  自上而下为标题栏(模型/后端徽标,如 `● NPU · RKNPU` / `● GPU · Burn WGPU` / `● CPU · Burn Flex`)、输入图像 + 预测结果(大数字、置信度条、
  与文件名标注是否一致、推理耗时)、0–9 概率分布、样本库网格、操作栏、状态栏。
- **横屏**(HDMI 等):左侧标题栏 + 输入图像/预测结果 + 操作栏,右侧概率分布 + 样本库。

交互:

- 点击样本库缩略图切换样本并立即识别;缩略图下方的标签条是文件名中的标注数字
  (单独占一条,不遮挡图片)。
- `‹ 上一张` / `下一张 ›` 循环切换(跳过无法解码的图片);`重新识别` 对当前样本再跑一次推理。
- `▶ 自动轮播` 每 2.5 s 切换到下一张(默认关闭;关闭时界面无定时推理,
  不会干扰板上的基准测试)。按钮宽度按两种文字中较宽的一种计算,切换时按钮行不移动。
- 概率条原地更新并带 260 ms 过渡动画。
- 样本库网格按行数搜索最大的单元格(含标签条),使全部样本尽量大并居中;区域过小时才滚动,
  且至少完整显示一行。
- 触摸点击不会抢占键盘焦点,避免最后点过的按钮一直带高亮边框。
- 低于 540×960 逻辑像素的竖屏(如 720×1280@2)自动收紧间距、隐藏重复的耗时行、
  缩小按钮字号,保证状态栏不被挤出屏幕;宽度不足 440 逻辑像素时标题栏改为两行
  (标题 + 徽标 / 副标题),文件名移到输入图像下方,避免截断和挤在一起。

字体:板上只有 DejaVu Sans 与文泉驿正黑(仅常规体),而 FemtoVG 不做伪粗体,
中文 `font-weight: 700` 会显示为常规体。标题、小节标题、置信度和主按钮因此使用
内嵌的 `ui/fonts/DRHeadingSC-Bold.ttf`:Noto Sans CJK SC Bold(SIL OFL 1.1,见
`ui/fonts/OFL.txt`)按界面实际用到的字符裁剪并改名,约 54 KB,随二进制编译进去,
无需在镜像里安装字体包。修改界面文字后重新生成(未覆盖的字会回退为常规体):

```sh
pip install fonttools
python3 tools/make_heading_font.py   # 默认读取 /usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc
```

截图(用于远程核对屏幕内容):

```sh
# 运行中的服务:发送 SIGUSR1,截图写到 /run/digit-recognizer/screenshot.png
sudo systemctl kill -s SIGUSR1 digit-recognizer.service
# 或指定路径:DIGIT_RECOGNIZER_CAPTURE=/path/shot.png
# 单次运行(需先停止占用屏幕的服务):启动 1.5 s 后截图并退出
digit-recognizer --capture /tmp/shot.png
```

## 样例图片

`assets/images/` 内置 20 张 MNIST 测试集图片(每个数字 2 张,
28×28 最近邻放大到 224×224 保留像素质感,文件名 `d<数字>_<序号>.png`)。重新生成:

```sh
# 数据: https://ossci-datasets.s3.amazonaws.com/mnist/ (t10k-images-idx3 / t10k-labels-idx1)
python3 tools/gen_mnist_samples.py assets/images
```

## 构建与运行

直接在当前仓库的 `bsp-digit-recognizer/` 目录下即可独立编译与运行测试(已内置 `vendor/i-slint-backend-linuxkms`,并通过 `path = ".."` 直接引用父目录的 `burn-mnist-rk3588`):

```sh
cargo test --manifest-path bsp-digit-recognizer/Cargo.toml
```

在 `atk-dlrk3588-bsp` 工程目录下(`app/digit-recognizer` 与 `app/burn-mnist-rk3588` 同级,将 `Cargo.toml` 中 `burn-mnist-rk3588` 的 `path` 设为 `"../burn-mnist-rk3588"`):

```sh
make display            # 交叉编译(与 slint-dashboard 相同 sysroot)
make display CARGO_BUILD_TARGET=x86_64-unknown-linux-gnu   # 主机调试构建
cargo test              # 单元测试(主机)
```

独立构建 CPU / GPU / NPU 单后端裁剪二进制:

```sh
# NPU 独立版 (仅 17 MB, 板端默认服务版本)
cargo build --release --target aarch64-unknown-linux-gnu --no-default-features --features npu
# CPU 独立版 (27 MB)
cargo build --release --target aarch64-unknown-linux-gnu --no-default-features --features cpu
# GPU 独立版 (97 MB)
cargo build --release --target aarch64-unknown-linux-gnu --no-default-features --features gpu
```

板上运行(图片目录默认按顺序探测 `/usr/share/digit-recognizer/images`、
`/opt/burn-mnist-rk3588/images`、`assets/images`):

```sh
digit-recognizer [--images-dir DIR] [--backend npu|npu-int8|cpu|gpu|vulkan|gl] [--benchmark] [--capture FILE.png]
```

`--benchmark` 不打开显示,逐张推理并打印准确率与平均延迟。

## 模型与板端服务配置 (`board/`)

- **NPU 模型加载顺序**:依次探测 `DIGIT_RECOGNIZER_RKNN_MODEL`、`/usr/share/digit-recognizer/model/mnist_cnn.rknn`、`/opt/burn-mnist-rk3588/model/mnist_cnn.rknn`、`assets/model/mnist_cnn.rknn`(INT8 模式优先探测 `mnist_cnn_int8.rknn` 或 RKNB 多配置包中的 `int8_b1` 切片)。
- **CPU / GPU 模型加载顺序**:依次探测 `DIGIT_RECOGNIZER_MODEL`、`/usr/share/digit-recognizer/model/mnist_cnn.bpk`、`/opt/burn-mnist-rk3588/model/mnist_cnn.bpk`、`assets/model/mnist_cnn.bpk`,最后回退到编译时内嵌的权重。
- **Systemd 服务与环境配置**:`board/digit-recognizer.service` 与 `board/digit-recognizer.default` 默认配置 `DIGIT_RECOGNIZER_BACKEND=npu`,以 `slint-display` 系统用户(`SupplementaryGroups=video render`)直接驱动 DRM KMS 与 `/dev/rknpu`。
- 后端初始化失败时 `model_live()` 为 false,界面徽标显示"演示数据 · 模型未接入",预测结果为基于文件名的确定性假数据。
