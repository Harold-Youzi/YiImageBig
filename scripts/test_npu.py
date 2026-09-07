import numpy as np
import openvino as ov
import time

print("== NPU 最小推理测试 (对照实验) ==")
core = ov.Core()
print("devices:", core.available_devices)

try:
    import os
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "RealESRGAN")
    model_path = os.path.join(model_dir, "RealESRGAN_x4plus_npu_128.xml")
    model = core.read_model(model_path)
    t0 = time.time()
    compiled = core.compile_model(model, "NPU", {"PERFORMANCE_HINT": "LATENCY"})
    print(f"compile: {time.time()-t0:.1f}s")

    req = compiled.create_infer_request()
    inp = np.random.rand(1, 3, 128, 128).astype(np.float32)
    t0 = time.time()
    res = req.infer({0: inp})
    out = res[compiled.output(0)]
    print(f"infer: {time.time()-t0:.2f}s, out shape: {out.shape}, dtype {out.dtype}")
    print("NPU OK")
except Exception as e:
    print(f"NPU FAILED: {type(e).__name__}: {e}")
