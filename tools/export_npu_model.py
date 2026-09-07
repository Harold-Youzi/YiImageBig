"""
为 NPU 导出固定形状 ONNX 模型 (128x128)
NPU 不支持动态形状, 必须用固定输入尺寸
"""
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

import torch
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.rrdbnet_arch import RRDBNet

WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pth_path = os.path.join(WORK_DIR, 'RealESRGAN_x4plus.pth')

# NPU 支持的最大安全尺寸 (Intel NPU 通常上限 1024x1024 但内存有限)
NPU_TILE = 128

print(f"Loading model...")
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

state_dict = torch.load(pth_path, map_location='cpu')
if 'params_ema' in state_dict:
    state_dict = state_dict['params_ema']
elif 'params' in state_dict:
    state_dict = state_dict['params']
if any(k.startswith('module.') for k in state_dict.keys()):
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

model.load_state_dict(state_dict, strict=True)
model.eval()

# 导出固定形状 ONNX (NPU 专用)
onnx_npu_path = os.path.join(WORK_DIR, f'RealESRGAN_x4plus_npu_{NPU_TILE}.onnx')
dummy = torch.randn(1, 3, NPU_TILE, NPU_TILE)

print(f"Exporting fixed-shape ONNX ({NPU_TILE}x{NPU_TILE})...")
torch.onnx.export(
    model, dummy, onnx_npu_path,
    opset_version=14,
    dynamo=False,
    input_names=['input'],
    output_names=['output'],
)
print(f"ONNX saved: {onnx_npu_path}")

# 转换为 OpenVINO IR (NPU 专用)
try:
    from openvino import Core, save_model
    core = Core()
    ov_model = core.read_model(onnx_npu_path)
    xml_npu = os.path.join(WORK_DIR, f'RealESRGAN_x4plus_npu_{NPU_TILE}.xml')
    save_model(ov_model, xml_npu)
    print(f"OpenVINO IR saved: {xml_npu}")
except Exception as e:
    print(f"OpenVINO conversion: {e}")

# 测试 NPU 编译
print()
print("=== NPU Compile Test ===")
try:
    compiled = core.compile_model(xml_npu, 'NPU')
    print("NPU compile SUCCESS!")

    inp = np.random.randn(1, 3, NPU_TILE, NPU_TILE).astype(np.float32)
    result = compiled({0: inp})
    out = result[0]
    print(f"NPU inference SUCCESS: {inp.shape} -> {out.shape}")

    # 第二次推理 (验证稳定性)
    inp2 = np.random.randn(1, 3, NPU_TILE, NPU_TILE).astype(np.float32)
    result2 = compiled({0: inp2})
    print(f"NPU inference x2 OK: {result2[0].shape}")

    del compiled
except Exception as e:
    print(f"NPU FAILED: {e}")

# 文件大小
print()
for f in [onnx_npu_path]:
    if os.path.exists(f):
        size = os.path.getsize(f) / 1024 / 1024
        print(f"{os.path.basename(f)}: {size:.1f} MB")

xml_npu = os.path.join(WORK_DIR, f'RealESRGAN_x4plus_npu_{NPU_TILE}.xml')
bin_npu = os.path.join(WORK_DIR, f'RealESRGAN_x4plus_npu_{NPU_TILE}.bin')
if os.path.exists(xml_npu):
    print(f"IR: {os.path.getsize(xml_npu)/1024/1024:.1f} MB + {os.path.getsize(bin_npu)/1024/1024:.1f} MB")

print()
print("Done!")
