# OpenVINO 2026 快速参考

> 已修复 DLL 路径 bug（sitecustomize.py 自动生效）。本机设备: CPU/GPU/NPU 全可用。

## 1. 导入（已自动修复，直接可用）

```python
import openvino._pyopenvino as ov
core = ov.Core()
print(core.available_devices)  # ['CPU','GPU','NPU']
```

若未自动修复（其他机器），先执行：
```python
import os
os.add_dll_directory(r"<site-packages>\openvino\libs")
```

## 2. 常用 API

| 功能 | 旧版(runtime) | 新版(_pyopenvino) |
|---|---|---|
| 创建核心 | `ov.Core()` | 同左 |
| 设备列表 | `core.available_devices` | 同左 |
| 设备名 | `get_property(d,"FULL_DEVICE_NAME")` | 同左 |
| 参数节点 | `ov.Parameter` | `from openvino._pyopenvino import op as ov_op; ov_op.Parameter` |
| 类型 | `ov.element.Type.f32` | `ov.Type.f32` |
| 编译模型 | `compile_model(m, d)` | `compile_model(m, d, {})` 必传第3参 |
| 推理 | `req.infer([t])` | `req.infer({0:t}, False, False)` 必传3参 |
| 输入转换 | 自动 | 必须 `ov.Tensor(np_arr)` |

## 3. 完整示例

```python
import numpy as np
import openvino._pyopenvino as ov
from openvino._pyopenvino import op as ov_op

core = ov.Core()
sh = ov.Shape([1,3,224,224])
a = ov_op.Parameter(ov.Type.f32, sh)
b = ov_op.Parameter(ov.Type.f32, sh)
model = ov.Model([a+b], [a, b])

for dev in core.available_devices:
    cm = core.compile_model(model, dev, {})
    t = ov.Tensor(np.ones([1,3,224,224], np.float32))
    cm.create_infer_request().infer({0:t, 1:t}, False, False)
    print(dev, "OK")
```

## 4. 相关文件

| 文件 | 位置 | 作用 |
|---|---|---|
| sitecustomize.py | `Lib\site-packages\` | 自动修复 DLL 路径（无需手动） |
| openvino_fix.py | `Lib\site-packages\` | 手动修复模块（备用） |
| check_devices.py | 工作目录 | 设备检测脚本 |

## 5. 已知坑

- `Parameter` 构造只有 `(Type, Shape)` 两参，无 name
- `infer` 输入必须是 `ov.Tensor`，不能用 numpy
- `compile_model` 第3参传空 dict `{}`
- Python 路径含中文（`C:\Users\铀子\`）曾致 numpy 源码编译失败 → 用 `--only-binary=:all:` 装预编译版
