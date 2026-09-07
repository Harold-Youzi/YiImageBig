"""
系统资源检测与限制
根据硬件配置自动调整处理参数
"""
import os
import platform
import subprocess
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SystemResources:
    """系统资源信息"""
    total_ram_gb: float = 8.0
    available_ram_gb: float = 4.0
    cpu_cores: int = 4
    cpu_threads: int = 8
    gpu_vram_gb: float = 0.0
    gpu_name: str = "None"
    npu_available: bool = False

    # 推荐处理参数
    max_tile_size: int = 256
    max_concurrent_frames: int = 1
    max_output_resolution: tuple = (4096, 4096)  # 最大输出分辨率 (H, W)
    use_optical_flow: bool = True


def _detect_nvidia_gpu_vram() -> tuple:
    """通过 nvidia-smi 检测 NVIDIA GPU 名称和显存 (GB), 失败返回 (None, 0)"""
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
                [smi, '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5, creationflags=flags
            )
            if result.returncode == 0 and result.stdout.strip():
                # 第一行: "NVIDIA GeForce RTX 4090, 24564"
                parts = result.stdout.strip().split('\n')[0].split(',')
                name = parts[0].strip()
                vram_mb = float(parts[1].strip())
                return name, vram_mb / 1024.0
        except Exception:
            continue
    return None, 0.0


def detect_system_resources() -> SystemResources:
    """检测系统资源并推荐处理参数"""
    res = SystemResources()

    # CPU
    try:
        import multiprocessing
        res.cpu_cores = multiprocessing.cpu_count()
        res.cpu_threads = min(res.cpu_cores * 2, 32)
    except Exception:
        pass

    # RAM
    try:
        import psutil
        mem = psutil.virtual_memory()
        res.total_ram_gb = mem.total / (1024**3)
        res.available_ram_gb = mem.available / (1024**3)
    except ImportError:
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            c_ulonglong = ctypes.c_uint64
            mem_info = c_ulonglong()
            kernel32.GetPhysicallyInstalledMemory(ctypes.byref(mem_info))
            res.total_ram_gb = mem_info.value / (1024**3)
            res.available_ram_gb = res.total_ram_gb * 0.5
        except Exception:
            res.total_ram_gb = 8.0
            res.available_ram_gb = 4.0

    # NVIDIA GPU (通过 nvidia-smi 检测, 最可靠)
    nv_name, nv_vram = _detect_nvidia_gpu_vram()
    if nv_name:
        res.gpu_name = nv_name
        res.gpu_vram_gb = nv_vram
        logger.info(f"NVIDIA GPU: {nv_name} ({nv_vram:.1f}GB)")

    # OpenVINO 设备 (Intel GPU/NPU)
    try:
        from openvino import Core
        core = Core()
        for dev in core.available_devices:
            if dev.startswith("GPU"):
                try:
                    full_name = core.get_property(dev, "FULL_DEVICE_NAME")
                except Exception:
                    full_name = dev
                # 如果 NVIDIA 已检测到, OpenVINO GPU 覆盖仅在 Intel 时
                lower = full_name.lower()
                is_nvidia = any(kw in lower for kw in ['nvidia', 'geforce', 'rtx', 'gtx'])
                is_amd = any(kw in lower for kw in ['amd', 'radeon'])
                if is_nvidia or is_amd:
                    continue
                res.gpu_name = full_name
                # Intel 集显: 估算显存为系统内存一半
                if res.gpu_vram_gb == 0:
                    res.gpu_vram_gb = min(res.total_ram_gb * 0.5, 8.0)
            elif dev == "NPU":
                res.npu_available = True
    except ImportError:
        logger.info("OpenVINO 未安装")
    except Exception as e:
        logger.warning(f"OpenVINO 检测失败: {e}")

    # ============================================================
    #  根据资源推荐处理参数
    # ============================================================
    # 优先用实际显存, 没有显存则用可用内存
    vram = res.gpu_vram_gb if res.gpu_vram_gb > 0 else res.available_ram_gb

    if vram >= 8:
        res.max_tile_size = 512
        res.max_concurrent_frames = 1
        res.max_output_resolution = (4320, 7680)  # 8K
    elif vram >= 4:
        res.max_tile_size = 256
        res.max_concurrent_frames = 1
        res.max_output_resolution = (2160, 3840)  # 4K
    elif vram >= 2:
        res.max_tile_size = 128
        res.max_concurrent_frames = 1
        res.max_output_resolution = (1080, 1920)  # 1080p
    else:
        res.max_tile_size = 64
        res.max_concurrent_frames = 1
        res.max_output_resolution = (720, 1280)

    if res.available_ram_gb < 2:
        res.use_optical_flow = False

    logger.info(
        f"系统资源: RAM={res.total_ram_gb:.1f}GB, "
        f"GPU={res.gpu_name}({res.gpu_vram_gb:.1f}GB), "
        f"CPU={res.cpu_cores}核 | "
        f"推荐: tile={res.max_tile_size}, 最大输出={res.max_output_resolution}"
    )

    return res


_system_resources: Optional[SystemResources] = None

def get_system_resources() -> SystemResources:
    global _system_resources
    if _system_resources is None:
        _system_resources = detect_system_resources()
    return _system_resources
