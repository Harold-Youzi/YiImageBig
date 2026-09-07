"""
NPU 兼容性测试脚本
"""
import os, sys, time
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openvino import Core
import numpy as np

core = Core()

# 1. NPU 信息
print('=== NPU Info ===')
try:
    name = core.get_property('NPU', 'FULL_DEVICE_NAME')
    print(f'Name: {name}')
except Exception as e:
    print(f'Name: {e}')

# 2. 用固定的 ONNX 模型测试
print()
print('=== NPU Compile Test (128x128) ===')
t0 = time.time()
try:
    model = core.read_model('RealESRGAN_x4plus.onnx')
    compiled = core.compile_model(model, 'NPU')
    t1 = time.time()
    print(f'Compile OK ({t1-t0:.1f}s)')

    inp = np.random.randn(1, 3, 128, 128).astype(np.float32)
    t2 = time.time()
    result = compiled({0: inp})
    t3 = time.time()
    out = result[0]
    print(f'Inference OK ({t3-t2:.1f}s): input={inp.shape} -> output={out.shape}')
    del compiled
except Exception as e:
    t1 = time.time()
    print(f'FAILED ({t1-t0:.1f}s): {e}')

# 3. 256x256
print()
print('=== NPU Test (256x256) ===')
t0 = time.time()
try:
    model2 = core.read_model('RealESRGAN_x4plus.onnx')
    compiled2 = core.compile_model(model2, 'NPU')
    inp2 = np.random.randn(1, 3, 256, 256).astype(np.float32)
    result2 = compiled2({0: inp2})
    t1 = time.time()
    print(f'OK ({t1-t0:.1f}s): input={inp2.shape} -> output={result2[0].shape}')
    del compiled2
except Exception as e:
    t1 = time.time()
    print(f'FAILED ({t1-t0:.1f}s): {e}')

# 4. 小尺寸 64x64
print()
print('=== NPU Test (64x64) ===')
t0 = time.time()
try:
    model3 = core.read_model('RealESRGAN_x4plus.onnx')
    compiled3 = core.compile_model(model3, 'NPU')
    inp3 = np.random.randn(1, 3, 64, 64).astype(np.float32)
    result3 = compiled3({0: inp3})
    t1 = time.time()
    print(f'OK ({t1-t0:.1f}s): input={inp3.shape} -> output={result3[0].shape}')
    del compiled3
except Exception as e:
    t1 = time.time()
    print(f'FAILED ({t1-t0:.1f}s): {e}')

print()
print('=== Done ===')
