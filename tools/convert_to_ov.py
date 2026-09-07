"""将 ONNX 模型转换为 OpenVINO IR 格式"""
import os

WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
onnx_path = os.path.join(WORK_DIR, 'RealESRGAN_x4plus.onnx')
xml_path = os.path.join(WORK_DIR, 'RealESRGAN_x4plus.xml')

print(f"Loading ONNX: {onnx_path}")

from openvino import Core, save_model

core = Core()
model = core.read_model(onnx_path)
print(f"Input: {model.input().shape}, Output: {model.output().shape}")

save_model(model, xml_path)
print(f"Saved: {xml_path}")

# 检查结果
pth_size = os.path.getsize(os.path.join(WORK_DIR, 'RealESRGAN_x4plus.pth')) / 1024 / 1024
xml_size = os.path.getsize(xml_path) / 1024 / 1024
bin_path = xml_path.replace('.xml', '.bin')
bin_size = os.path.getsize(bin_path) / 1024 / 1024 if os.path.exists(bin_path) else 0
print(f"\n=== File Sizes ===")
print(f"PTH:  {pth_size:.1f} MB")
print(f"IR:   {xml_size + bin_size:.1f} MB (XML {xml_size:.1f} + BIN {bin_size:.1f})")
print(f"Saved: {pth_size - xml_size - bin_size:.1f} MB")
