"""
重新导出 ONNX 模型 (动态形状)
- 动态 batch, height, width
- 支持任意分辨率输入
"""
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
import torch
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.rrdbnet_arch import RRDBNet

# 输出到项目根目录（与 run.py 同级），而非 tools/ 目录
WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pth_path = os.path.join(WORK_DIR, 'RealESRGAN_x4plus.pth')
onnx_path = os.path.join(WORK_DIR, 'RealESRGAN_x4plus.onnx')

print("Loading model...")
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

# 导出 ONNX (动态形状)
dummy = torch.randn(1, 3, 64, 64)
torch.onnx.export(
    model, dummy, onnx_path,
    opset_version=18,
    dynamo=False,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch', 2: 'height', 3: 'width'},
        'output': {0: 'batch', 2: 'height', 3: 'width'}
    }
)
print(f"ONNX exported (dynamic): {onnx_path}")

# 同时重新导出 OpenVINO IR (从新 ONNX)
try:
    from openvino import Core, save_model
    core = Core()
    ov_model = core.read_model(onnx_path)
    xml_path = os.path.join(WORK_DIR, 'RealESRGAN_x4plus.xml')
    save_model(ov_model, xml_path)
    print(f"OpenVINO IR saved: {xml_path}")
except Exception as e:
    print(f"OpenVINO conversion skipped: {e}")

# 文件大小
pth_size = os.path.getsize(pth_path) / 1024 / 1024
onnx_size = os.path.getsize(onnx_path) / 1024 / 1024
xml_path = os.path.join(WORK_DIR, 'RealESRGAN_x4plus.xml')
bin_path = os.path.join(WORK_DIR, 'RealESRGAN_x4plus.bin')
if os.path.exists(xml_path):
    xml_size = os.path.getsize(xml_path) / 1024 / 1024
    bin_size = os.path.getsize(bin_path) / 1024 / 1024
    print(f"PTH: {pth_size:.1f}MB, ONNX: {onnx_size:.1f}MB, IR: {xml_size+bin_size:.1f}MB")
else:
    print(f"PTH: {pth_size:.1f}MB, ONNX: {onnx_size:.1f}MB")
