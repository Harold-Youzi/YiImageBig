# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller 打包配置 - YiImageBig (精简版)
OpenVINO (Intel) + ONNX Runtime (NVIDIA/CPU) - 不含 PyTorch
"""
import os
import sys

block_cipher = None
ROOT_DIR = os.path.dirname(os.path.abspath(SPEC))
SRC_DIR = os.path.join(ROOT_DIR, 'src')

# ============================================================
#  收集项目文件
# ============================================================
project_datas = []
excluded_filenames = ['video_upscaler.py', 'optical_flow.py']
for root, dirs, files in os.walk(SRC_DIR):
    for f in files:
        if f.endswith('.py'):
            if f in excluded_filenames:
                print(f"  跳过视频模块: {f}")
                continue
            src_path = os.path.join(root, f)
            rel_dir = os.path.relpath(root, ROOT_DIR)
            project_datas.append((src_path, rel_dir))

# 模型文件不复制到 _internal (避免与根目录重复)
# 运行时从 exe 同目录搜索: YiImageBig/RealESRGAN_x4plus.*

datas = project_datas

# ============================================================
#  隐藏导入
# ============================================================
from PyInstaller.utils.hooks import collect_submodules

src_modules = collect_submodules('src')
# 排除视频处理模块（成品不含视频功能）
src_modules = [m for m in src_modules if 'video_upscaler' not in m and 'optical_flow' not in m]

hidden_imports = [
    'PySide6', 'PySide6.QtCore', 'PySide6.QtWidgets', 'PySide6.QtGui',
    'cv2', 'cv2.data',
    'numpy',
    'openvino',
    'onnxruntime', 'onnxruntime.capi',
] + src_modules

# 收集 openvino
from PyInstaller.utils.hooks import collect_all
try:
    ov_datas, ov_binaries, ov_hidden = collect_all('openvino')
    datas += ov_datas
    binaries = ov_binaries
    hidden_imports += ov_hidden
except Exception:
    binaries = []

# 收集 onnxruntime
try:
    ort_datas, ort_binaries, ort_hidden = collect_all('onnxruntime')
    datas += ort_datas
    binaries += ort_binaries
    hidden_imports += ort_hidden
except Exception:
    pass

# ============================================================
#  排除
# ============================================================
excludes = [
    # PyTorch / 训练框架
    'torch', 'torchvision', 'torchaudio',
    # 数据科学
    'matplotlib', 'scipy', 'pandas',
    # GUI
    'tkinter',
    # 网络/Web
    'xmlrpc', 'pydoc', 'http', 'h2',
    # 开发工具
    'IPython', 'jupyter', 'notebook', 'setuptools', 'pkg_resources',
    # 旧依赖
    'basicsr', 'realesrgan',
    # 未使用的包
    'PIL', 'PIL.Image', 'PIL.ImageFilter', 'PIL.ImageDraw', 'PIL.ImageFont',
    'cryptography', 'cffi', '_cffi_backend',
    'lxml', 'lxml.etree', 'lxml._elementpath',
    'onnx', 'onnx.onnx_ml_pb2', 'onnx.onnx_data_pb2', 'onnx.onnx_pb',
    'ml_dtypes',
    'certifi',
    'yaml', 'yaml.loader',
    'pywin32_system32',
    'google', 'google.protobuf',
    'tqdm',
    'psutil',
    'multiprocessing.resource_tracker',
]

a = Analysis(
    [os.path.join(ROOT_DIR, 'run.py')],
    pathex=[ROOT_DIR],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='YiImageBig',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='YiImageBig',
)
