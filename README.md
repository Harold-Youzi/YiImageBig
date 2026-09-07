# YiImageBig 

<p align="center">
  <img src="docs/icon.png" width="128" alt="YiImageBig Icon" />
</p>

<p align="center">
  <b>基于 RealESRGAN_x4plus 的 AI 图片放大工具，支持 NVIDIA GPU / Intel GPU / Intel NPU / CPU 多硬件加速</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-0.0.3--beta-blue" alt="Version" />
  <img src="https://img.shields.io/badge/python-3.10+-green" alt="Python" />
  <img src="https://img.shields.io/badge/license-MIT-orange" alt="License" />
  <img src="https://img.shields.io/badge/platform-Windows-lightgrey" alt="Platform" />
</p>

---

## 功能特点

- **4 倍 AI 超分辨率放大**：基于 RealESRGAN_x4plus 模型，画质优秀
- **隐私安全**：无需联网，本地运行
- **多硬件智能加速**：自动检测并选择最优硬件
- **NPU支持**：支持Intel NPU
- **多推理后端**：支持 ONNX Runtime (CUDA / DirectML) 和 OpenVINO (GPU / NPU / CPU)
- **分块推理**：智能分块 + 余弦融合，支持超大图片不爆显存
- **ETA 预估**：基于 Benchmark + EMA 算法，实时显示预计剩余时间
- **中文路径**：完整支持中文文件名和路径


## 支持的硬件

| 硬件 | 推理后端 | 评分 | 分块大小 | 备注 |
|------|----------|------|----------|------|
| NVIDIA GPU (RTX/GTX) | ONNX Runtime CUDA | 100 | 512×512 | 需安装 CUDA 驱动 |
| NVIDIA GPU (Fallback) | ONNX Runtime DirectML | 70 | 512×512 | 无 CUDA 时自动启用 |
| Intel NPU (Lunar Lake+) | OpenVINO | 90 | 128×128 | 固定形状模型 |
| Intel GPU (Arc/集显) | OpenVINO | 80 | 512×512 | FP16 加速 |
| CPU (OpenVINO) | OpenVINO | 40 | 256×256 | Intel CPU 优化 |
| CPU (通用) | ONNX Runtime | 30 | 256×256 | 最慢但兼容性最好 |

### 回退链

```
CUDA → DirectML → OpenVINO CPU → ONNX Runtime CPU
```

## 快速开始

### 方式一：直接运行（推荐）

1. 从 [Releases](https://github.com/YiImageBig/YiImageBig/releases) 下载最新版本
2. 解压到任意目录
3. 双击 `YiImageBig.exe` 或 `Launch.bat` 启动

### 方式二：从源码运行

```bash
# 克隆仓库
git clone https://github.com/YiImageBig/YiImageBig.git
cd YiImageBig

# 安装依赖
pip install -r requirements.txt

# 下载模型文件（参见下方"模型说明"）

# 运行
python run.py
```

## 模型文件


| 文件 | 大小 | 用途 |
|------|------|------|
| `RealESRGAN_x4plus.onnx` | ~64MB | ONNX Runtime 推理（NVIDIA GPU / CPU） |
| `RealESRGAN_x4plus.xml` | ~3MB | OpenVINO IR 头文件（Intel GPU/CPU） |
| `RealESRGAN_x4plus.bin` | ~32MB | OpenVINO IR 权重 |
| `RealESRGAN_x4plus_npu_128.xml` | ~3MB | NPU 专用模型头文件 |
| `RealESRGAN_x4plus_npu_128.bin` | ~32MB | NPU 专用模型权重 |
| `RealESRGAN_x4plus.pth` | ~64MB | PyTorch 原始权重（运行时不需要） |

将所有模型文件放在项目根目录（与 `run.py` 同级），或放在打包后 `YiImageBig.exe` 同级目录。

### 模型转换

如果你只有 `.pth` 文件，可以用以下工具转换：

```bash
# PyTorch → 动态 ONNX + OpenVINO IR
python tools/export_dynamic_onnx.py

# 导出 NPU 固定 128×128 模型
python tools/export_npu_model.py
```


### 打包目录结构

```
YiImageBig/
├── YiImageBig.exe                    # 主程序 (17MB)
├── _internal/                        # 运行时依赖 (无模型文件)
│   ├── src/                          # 源代码
│   ├── PySide6/
│   ├── openvino/
│   ├── onnxruntime/
│   └── ...
├── RealESRGAN_x4plus.xml             # OpenVINO 动态模型
├── RealESRGAN_x4plus.bin
├── RealESRGAN_x4plus.onnx            # ONNX 动态模型
├── RealESRGAN_x4plus_npu_128.xml     # NPU 专用模型
├── RealESRGAN_x4plus_npu_128.bin
└── Launch.bat                        # 启动脚本
```

## 项目结构

```
YiImageBig/
├── run.py                      # 程序入口
├── build_lite.spec             # PyInstaller 打包配置
├── cleanup_package.py          # 打包后清理脚本
├── requirements.txt            # Python 依赖
├── pyproject.toml              # 项目元数据
├── LICENSE                     # MIT 许可证
├── README.md
│
├── src/                        # 源代码
│   ├── __init__.py
│   ├── main.py                 # 模块入口
│   ├── core/
│   │   ├── __init__.py
│   │   ├── model_manager.py    # 多后端推理引擎
│   │   ├── image_upscaler.py   # 图片放大
│   │   ├── video_upscaler.py   # 视频放大（开发中）
│   │   ├── optical_flow.py     # 光流时序平滑（开发中）
│   │   └── rrdbnet_arch.py     # RRDBNet 架构定义
│   ├── gui/
│   │   ├── __init__.py
│   │   └── main_window.py      # PySide6 GUI
│   └── utils/
│       ├── __init__.py
│       ├── hardware_detector.py  # 硬件检测与智能选择
│       └── system_info.py        # 系统资源检测
│
├── tools/                      # 模型转换工具
│   ├── export_dynamic_onnx.py  # PyTorch → 动态 ONNX/OpenVINO IR
│   ├── export_npu_model.py     # 导出 NPU 固定形状模型
│   └── convert_to_ov.py        # ONNX → OpenVINO IR
│
├── tests/
│   └── test_npu.py             # NPU 推理测试
│
└── docs/                       # 文档
    └── openvino-fix.md
```

### 推理引擎

- **RRDBNet**：23 个残差密集块，64 通道特征，4 倍放大
- **分块推理**：GPU 512×512 / NPU 128×128 / CPU 256×256
- **余弦融合**：`0.5 * (1 - cos(π * t))` 平滑过渡，消除分块接缝
- **多后端**：OpenVINO (Intel GPU/NPU/CPU) + ONNX Runtime (CUDA/DirectML/CPU)

### 硬件检测原则

- `is_available = True` 表示后端**真正可用**，不仅仅是硬件被检测到
- `device_type` 精确匹配推理后端（"cuda"、"directml"、"openvino_gpu" 等）
- nvidia-smi 作为**主要**检测手段，CUDAExecutionProvider 作为能力验证
- OpenVINO 自动跳过非 Intel GPU（NVIDIA、AMD），避免误识别

### ETA 预估算法

1. **Benchmark 预热**：首次启动或切换设备时，生成 256×256 随机张量，运行 warmup + 计时
2. **EMA 滑动平均**：`T_new = α × T_actual + (1-α) × T_old`（α=0.25）
3. **显示**："预计剩余 2分30秒"，每完成一个分块更新

## 依赖

```
PySide6>=6.5.0
opencv-python>=4.8.0
openvino>=2025.0.0
onnxruntime-directml>=1.24.0
numpy>=1.24.0
psutil>=5.9.0
```

## 系统要求

- **操作系统**：Windows 10/11 (64-bit)（暂不支持MacOS、Linux、HarmonyOS）
- **Python**：3.10+（从源码运行时）
- **硬件**：Intel CPU/GPU/NPU或Nvidia GPU（暂不支持AMD平台）（详见：支持的硬件）
- **内存**：≥4GB
- **显存**：≥2GB（使用 GPU 加速时）

欢迎提交 Issue ！

## 许可证

本项目采用 [MIT 许可证](LICENSE)。

## 致谢

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) — 基础超分辨率模型
- [OpenVINO](https://github.com/openvinotoolkit/openvino) — Intel 硬件推理加速
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) — 跨平台推理引擎
- [PySide6](https://doc.qt.io/qtforpython-6/) — GUI 框架