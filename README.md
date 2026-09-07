# YiImageBig

基于 **RealESRGAN** 模型的本地图片超分辨率放大工具。纯本地推理，无网络依赖。

## 功能

- 4 倍超分放大
- 多硬件自动适配：
  - Intel CPU / 核显 / Arc 独显 / NPU —— 通过 **OpenVINO**
  - NVIDIA GPU —— 通过 **DirectML** (ONNX Runtime)
- 本地运行，无需联网
- Intel NPU支持
- 支持多任务批处理
- 大图分块推理 + 重叠融合，拼接无接缝
- Win32 GUI 图形界面 + 命令行模式

## 系统要求

- Windows 10 / 11 (x64)
- Intel CPU/GPU/NPU或Nvidia GPU（暂不支持AMD平台）（详见：支持的硬件）
- CMake ≥ 3.24
- MinGW-w64 (UCRT64 GCC) 或 MSVC

## 构建步骤

### 1. 克隆仓库

```bash
git clone https://github.com/yourname/YiImageBig.git
cd YiImageBig
```

### 2. 下载第三方依赖

```powershell
powershell -ExecutionPolicy Bypass -File scripts/fetch_sdks.ps1
```

这会自动从 NuGet / GitHub 下载以下依赖：

| 依赖 | 版本 | 用途 |
|------|------|------|
| OpenVINO Runtime | 2025.4.0 | Intel CPU/GPU/NPU 推理 |
| ONNX Runtime DirectML | 1.24.4 | NVIDIA GPU 推理 |
| Microsoft.AI.DirectML | 1.15.4 | DirectML 运行时 |
| stb_image | - | 图片读写（仅头文件） |

### 3. 构建

```bash
# MinGW-w64 (推荐)
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

# 或 MSVC
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

构建产物在 `build/bin/YiImageBig.exe`。

### 4. 部署（打包为可分发程序）

```powershell
powershell -ExecutionPolicy Bypass -File scripts/deploy.ps1
```

会自动将 exe、运行时 DLL、模型文件打包到 `YiImageBig(C++)` 文件夹。

## 目录结构

```
SourceCode(C++)/
├── CMakeLists.txt          构建配置
├── .gitignore
├── README.md
├── src/
│   ├── main.cpp            入口 (GUI/CLI 双模式)
│   ├── gui/
│   │   ├── app.h/cpp       GUI 主窗口 (纯 Win32)
│   │   └── widgets.cpp     自定义控件
│   ├── core/
│   │   ├── common.h/cpp    日志、工具函数、并行
│   │   ├── image.h/cpp     图片 I/O (stb) + 图像处理
│   │   ├── ov_engine.h/cpp OpenVINO 推理引擎
│   │   ├── ort_engine.h/cpp ONNX Runtime 推理引擎
│   │   ├── devices.h/cpp   硬件检测与设备枚举
│   │   └── upscaler.h/cpp  超分核心流程 (分块/融合/后处理)
│   └── res/
│       ├── app.ico         图标
│       ├── app.rc          资源脚本
│       └── app.manifest    应用清单
├── scripts/
│   ├── fetch_sdks.ps1      下载第三方依赖
│   ├── deploy.ps1          构建 + 部署
│   ├── autotest.cpp        端到端自动化回归 (g++ 编译)
│   └── ...                 辅助脚本
└── third_party/
    └── stb/                stb 头文件 (提交到 git)
        ├── stb_image.h
        └── stb_image_write.h
```

> `third_party/ov/`、`third_party/ort/`、`third_party/dml/` 由 `fetch_sdks.ps1` 自动下载，已加入 `.gitignore`，不提交到仓库。

## 模型文件

RealESRGAN 模型需要单独准备（可以从https://github.com/YiImageBig/YiImageBig/releases下载），放到 `models` 目录（与 exe 同级）：

| 文件 | 用途 |
|------|------|
| `RealESRGAN_x4plus.xml` / `.bin` | OpenVINO IR 模型 (CPU/GPU, tile=256) |
| `RealESRGAN_x4plus_npu_128.xml` / `.bin` | NPU 专用静态 shape (tile=128) |
| `RealESRGAN_x4plus.onnx` | DirectML / ONNX Runtime (NVIDIA GPU) |

## 命令行用法

```bash
# 基本用法（自动选设备）
YiImageBig.exe --upscale input.png

# 指定 NPU、输出目录
YiImageBig.exe --upscale input.png --device npu --out ./output

# 指定倍数
YiImageBig.exe --upscale input.png --scale 4
```

## 许可证

本项目源代码采用 [MIT License](LICENSE)。

RealESRGAN 模型采用BSD-3-Clause license许可（https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE）。
