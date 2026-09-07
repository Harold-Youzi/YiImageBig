"""
硬件检测与智能选择算法
支持: CPU / NVIDIA GPU (ONNX Runtime CUDA/DirectML) / Intel GPU (OpenVINO) / Intel NPU (OpenVINO)

设计原则:
  - is_available: 后端真正可用, 引擎可以调用
  - device_type: 精确匹配推理后端 ("cuda", "directml", "openvino_gpu", "openvino_npu", "openvino_cpu", "cpu")
  - 不混淆 "硬件存在" 与 "后端可用"
"""
import os
import platform
import subprocess
import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class HardwareDevice:
    """硬件设备信息
    is_available = True 时, device_type 对应的推理后端必须能直接使用
    is_available = False 时表示检测到硬件但后端不可用 (不传给引擎)
    """
    name: str
    device_type: str  # 'cpu', 'cuda', 'directml', 'openvino_gpu', 'openvino_npu', 'openvino_cpu'
    is_available: bool
    full_name: Optional[str] = None
    score: int = 0
    backend_info: str = ""  # 后端详情: "CUDA 12.6" / "DirectML" / "OpenVINO 2025.4" 等


SCORES = {
    "cuda": 100,
    "directml": 70,        # DirectML 性能低于 CUDA, 但比 OpenVINO CPU 好
    "openvino_npu": 90,    # NPU 实测比 iGPU 更快 (8.8s vs 19.1s for 1080p)
    "openvino_gpu": 80,
    "openvino_cpu": 40,
    "cpu": 30,
}


class HardwareDetector:

    def __init__(self):
        self.devices: List[HardwareDevice] = []
        self._nvidia_gpu_names: List[str] = []
        self._nvidia_vram_gb: float = 0.0
        self._detect_all()

    def _detect_all(self):
        self.devices.clear()

        # 1. NVIDIA GPU (nvidia-smi 检测硬件 + ONNX Runtime 检测后端)
        self._detect_nvidia_gpu()

        # 2. CPU (始终可用)
        cpu_name = platform.processor() or "CPU"
        self.devices.append(HardwareDevice(
            name="CPU", device_type="cpu", is_available=True,
            full_name=cpu_name, score=SCORES["cpu"]
        ))

        # 3. OpenVINO 设备 (Intel GPU/NPU/CPU)
        self._detect_openvino()

        logger.info(f"检测到 {len(self.devices)} 个可用设备")
        for dev in self.devices:
            logger.info(f"  [{dev.name}] {dev.full_name} type={dev.device_type} score={dev.score}")

    # ================================================================
    #  NVIDIA GPU 检测
    # ================================================================

    def _get_nvidia_gpu_names(self) -> List[str]:
        """通过 nvidia-smi 获取 NVIDIA GPU 名称列表, 多路径尝试"""
        if self._nvidia_gpu_names:
            return self._nvidia_gpu_names

        smi_paths = ['nvidia-smi']
        for env_var in ['ProgramFiles', 'ProgramW6432', 'CUDA_PATH']:
            base = os.environ.get(env_var, '')
            if base:
                smi_paths.append(os.path.join(base, 'NVIDIA Corporation', 'NVSMI', 'nvidia-smi.exe'))
                smi_paths.append(os.path.join(base, 'NVIDIA', 'NVSMI', 'nvidia-smi.exe'))
        system_root = os.environ.get('SystemRoot', r'C:\Windows')
        smi_paths.append(os.path.join(system_root, 'System32', 'nvidia-smi.exe'))

        flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        for smi in smi_paths:
            try:
                result = subprocess.run(
                    [smi, '--query-gpu=name', '--format=csv,noheader'],
                    capture_output=True, text=True, timeout=5, creationflags=flags
                )
                if result.returncode == 0 and result.stdout.strip():
                    self._nvidia_gpu_names = [n.strip() for n in result.stdout.strip().split('\n') if n.strip()]
                    return self._nvidia_gpu_names
            except Exception:
                continue

        # 兜底: WMI 查询
        try:
            result = subprocess.run(
                ['wmic', 'path', 'win32_videocontroller', 'get', 'name', '/format:list'],
                capture_output=True, text=True, timeout=5, creationflags=flags
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if line.startswith('Name=') and 'nvidia' in line.lower():
                        self._nvidia_gpu_names.append(line.split('=', 1)[1].strip())
        except Exception:
            pass

        return self._nvidia_gpu_names

    def _get_nvidia_vram(self) -> float:
        """通过 nvidia-smi 获取 NVIDIA GPU 显存 (GB)"""
        if self._nvidia_vram_gb > 0:
            return self._nvidia_vram_gb

        smi_paths = ['nvidia-smi']
        for env_var in ['ProgramFiles', 'ProgramW6432', 'CUDA_PATH']:
            base = os.environ.get(env_var, '')
            if base:
                smi_paths.append(os.path.join(base, 'NVIDIA Corporation', 'NVSMI', 'nvidia-smi.exe'))
        system_root = os.environ.get('SystemRoot', r'C:\Windows')
        smi_paths.append(os.path.join(system_root, 'System32', 'nvidia-smi.exe'))

        flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        for smi in smi_paths:
            try:
                result = subprocess.run(
                    [smi, '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=5, creationflags=flags
                )
                if result.returncode == 0 and result.stdout.strip():
                    # 取第一块 GPU 的显存 (MB → GB)
                    first_line = result.stdout.strip().split('\n')[0].strip()
                    self._nvidia_vram_gb = float(first_line) / 1024.0
                    return self._nvidia_vram_gb
            except Exception:
                continue
        return self._nvidia_vram_gb

    def _is_nvidia_gpu_device(self, full_name: str) -> bool:
        """判断 OpenVINO 检测到的 GPU 是否是 NVIDIA 设备 (纯关键字匹配)"""
        lower = full_name.lower()
        nvidia_keywords = ['nvidia', 'geforce', 'rtx', 'gtx', 'quadro', 'tesla', 'titan',
                           'nvda', 'nv_', 'cuve']
        return any(kw in lower for kw in nvidia_keywords)

    def _detect_nvidia_gpu(self):
        """检测 NVIDIA GPU:
        1. nvidia-smi 检测硬件 + 显存
        2. ONNX Runtime 检测后端: CUDA > DirectML
        3. 只有后端真正可用时才 is_available=True
        """
        # 第一步: nvidia-smi 检测硬件
        nvidia_names = self._get_nvidia_gpu_names()
        if not nvidia_names:
            logger.info("nvidia-smi 未检测到 NVIDIA GPU")
            return

        gpu_name = f"NVIDIA {nvidia_names[0]}"
        if len(nvidia_names) > 1:
            gpu_name += f" (x{len(nvidia_names)})"

        # 获取显存
        vram_gb = self._get_nvidia_vram()
        if vram_gb > 0:
            gpu_name += f" {vram_gb:.0f}GB"

        # 第二步: 检测 ONNX Runtime 后端
        cuda_ready = False
        dml_ready = False
        ort_version = ""
        try:
            import onnxruntime as ort
            ort_version = ort.__version__
            providers = ort.get_available_providers()
            cuda_ready = 'CUDAExecutionProvider' in providers
            dml_ready = 'DmlExecutionProvider' in providers
            logger.info(f"ONNX Runtime {ort_version} providers: {providers}")
        except Exception as e:
            logger.warning(f"ONNX Runtime 检测异常: {e}")

        # 第三步: 按优先级添加 (只添加后端真正可用的设备)
        if cuda_ready:
            self.devices.append(HardwareDevice(
                name="NVIDIA GPU", device_type="cuda", is_available=True,
                full_name=gpu_name, score=SCORES["cuda"],
                backend_info=f"CUDA (ONNX Runtime {ort_version})"
            ))
            logger.info(f"添加 NVIDIA GPU (CUDA): {gpu_name}")
        elif dml_ready:
            self.devices.append(HardwareDevice(
                name="NVIDIA GPU (DirectML)", device_type="directml", is_available=True,
                full_name=gpu_name, score=SCORES["directml"],
                backend_info=f"DirectML (ONNX Runtime {ort_version})"
            ))
            logger.info(f"添加 NVIDIA GPU (DirectML): {gpu_name}")
        else:
            # 硬件存在但后端不可用: 不添加, 也不设 is_available=True
            logger.warning(
                f"NVIDIA GPU 已检测到 ({gpu_name}) 但无可用后端 "
                f"(需要 CUDA Runtime 或 DirectML 运行时)"
            )

    # ================================================================
    #  OpenVINO 设备检测 (Intel GPU/NPU/CPU)
    # ================================================================

    def _detect_openvino(self):
        """检测 OpenVINO 设备, 带超时保护"""
        try:
            from openvino import Core
            core = Core()

            try:
                devices = core.available_devices
            except Exception as e:
                logger.warning(f"OpenVINO 获取设备列表失败: {e}")
                return

            for dev_name in devices:
                try:
                    full_name = core.get_property(dev_name, "FULL_DEVICE_NAME")
                except Exception:
                    full_name = dev_name

                if dev_name == "CPU":
                    self.devices.append(HardwareDevice(
                        name="OpenVINO CPU", device_type="openvino_cpu",
                        is_available=True, full_name=full_name,
                        score=SCORES["openvino_cpu"], backend_info="OpenVINO"
                    ))

                elif dev_name.startswith("GPU"):
                    # 跳过非 Intel GPU (如 NVIDIA 被 SYCL 检测到)
                    if self._is_nvidia_gpu_device(full_name):
                        logger.info(f"跳过 OpenVINO GPU (非 Intel): {full_name}")
                        continue
                    # 跳过 AMD GPU
                    lower = full_name.lower()
                    if 'amd' in lower or 'radeon' in lower or 'rx ' in lower:
                        logger.info(f"跳过 OpenVINO GPU (AMD): {full_name}")
                        continue
                    # 判断是 Arc 还是集显
                    is_arc = "Arc" in full_name or "DG" in full_name
                    display = "Intel Arc GPU" if is_arc else "Intel GPU"
                    self.devices.append(HardwareDevice(
                        name=display, device_type="openvino_gpu", is_available=True,
                        full_name=full_name, score=SCORES["openvino_gpu"],
                        backend_info="OpenVINO"
                    ))

                elif dev_name == "NPU":
                    self.devices.append(HardwareDevice(
                        name="Intel NPU", device_type="openvino_npu",
                        is_available=True, full_name=full_name,
                        score=SCORES["openvino_npu"], backend_info="OpenVINO"
                    ))

        except ImportError:
            logger.info("OpenVINO 未安装")
        except Exception as e:
            logger.warning(f"OpenVINO 检测失败: {e}")

    # ================================================================
    #  设备查询接口
    # ================================================================

    def get_available_devices(self) -> List[HardwareDevice]:
        """只返回 is_available=True 的设备 (后端真正可用)"""
        return [d for d in self.devices if d.is_available]

    def get_device_by_type(self, device_type: str) -> Optional[HardwareDevice]:
        for dev in self.devices:
            if dev.device_type == device_type and dev.is_available:
                return dev
        return None

    def select_optimal_device(self) -> HardwareDevice:
        """智能选择最优设备 (按评分降序, 只从可用设备中选)"""
        available = self.get_available_devices()
        if not available:
            return self.devices[0]
        return sorted(available, key=lambda d: d.score, reverse=True)[0]

    def get_device_display_name(self, device: HardwareDevice) -> str:
        display_map = {
            "cuda": "NVIDIA GPU",
            "directml": "NVIDIA GPU (DirectML)",
            "openvino_gpu": "Intel GPU",
            "openvino_npu": "Intel NPU",
            "openvino_cpu": "OpenVINO CPU",
            "cpu": "CPU",
        }
        return display_map.get(device.device_type, device.name)

    def get_device_summary(self) -> str:
        lines = ["=== Hardware Detection ==="]
        for dev in self.devices:
            tag = "[OK]" if dev.is_available else "[--]"
            backend = f" ({dev.backend_info})" if dev.backend_info else ""
            lines.append(f"  {tag} {dev.name}: {dev.full_name}{backend} score={dev.score}")
        return "\n".join(lines)


_hardware_detector: Optional[HardwareDetector] = None

def get_hardware_detector() -> HardwareDetector:
    global _hardware_detector
    if _hardware_detector is None:
        _hardware_detector = HardwareDetector()
    return _hardware_detector

hardware_detector = get_hardware_detector()
