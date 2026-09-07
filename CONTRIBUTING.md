# 贡献指南

感谢您对 YiImageBig 的关注！本文档将帮助您参与项目开发。

## 环境准备

### 系统要求
- Windows 10/11 (64-bit)
- Python 3.10+

### 安装依赖

```bash
# 克隆仓库
git clone https://github.com/YiImageBig/YiImageBig.git
cd YiImageBig

# 创建虚拟环境（推荐）
python -m venv venv
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 额外开发依赖（可选）
pip install pyinstaller
```

### 模型文件

将以下模型文件放在项目根目录：
- `RealESRGAN_x4plus.onnx` — ONNX Runtime 推理
- `RealESRGAN_x4plus.xml` + `.bin` — OpenVINO IR 模型
- `RealESRGAN_x4plus_npu_128.xml` + `.bin` — NPU 专用模型
- `RealESRGAN_x4plus.pth` — PyTorch 原始权重（仅用于模型转换）

## 项目结构

```
SourceCode/
├── run.py                      # 主入口（PyInstaller 和开发模式均使用）
├── src/                        # 源代码
│   ├── core/                   # 推理引擎
│   │   ├── model_manager.py    # 多后端推理（OpenVINO / ONNX Runtime）
│   │   ├── image_upscaler.py   # 图片放大逻辑
│   │   ├── video_upscaler.py   # 视频放大（开发中）
│   │   ├── optical_flow.py     # 光流平滑（开发中）
│   │   └── rrdbnet_arch.py     # RRDBNet 网络架构
│   ├── gui/
│   │   └── main_window.py      # PySide6 GUI
│   └── utils/
│       ├── hardware_detector.py # 硬件检测与选择
│       └── system_info.py      # 系统资源检测
├── tools/                      # 模型转换工具
├── tests/                      # 测试
└── docs/                       # 文档
```

## 开发规范

### 代码风格
- 使用 Python 3.10+ 语法
- 字符串统一使用**单引号**（避免中文引号导致 SyntaxError）
- 遵循 PEP 8 命名规范
- 每个模块添加模块级 docstring

### 硬件检测原则
- `is_available = True` 表示后端**真正可用**，不仅仅是硬件被检测到
- `device_type` 精确匹配推理后端名称
- nvidia-smi 作为 NVIDIA GPU 主要检测手段
- OpenVINO 自动跳过非 Intel GPU 设备

### 推理后端回退链
```
CUDA → DirectML → OpenVINO CPU → ONNX Runtime CPU
```

### 提交规范
- `feat:` 新功能
- `fix:` 修复 Bug
- `perf:` 性能优化
- `docs:` 文档更新
- `refactor:` 代码重构
- `test:` 测试相关

## 构建与打包

```bash
# PyInstaller 打包
pyinstaller build_lite.spec --noconfirm --clean

# 清理无用文件
python cleanup_package.py
```

## 提交 Issue

- 使用 Issue 模板
- 附上 `app.log` 日志文件
- 说明硬件配置（CPU/GPU/NPU 型号）
- 描述问题复现步骤

## 提交 Pull Request

1. Fork 本仓库
2. 创建功能分支：`git checkout -b feature/your-feature`
3. 确保代码通过基本测试
4. 提交更改并附上有意义的 commit message
5. 推送分支：`git push origin feature/your-feature`
6. 创建 Pull Request，描述修改内容

## 许可证

贡献的代码将采用 MIT 许可证发布。
