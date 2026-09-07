"""
视频放大器 - 流水线版
架构: 3 阶段流水线 (Reader → Infer+Smooth → Writer) 实现 I/O 与 GPU 并行
修复: 暂停/停止功能, 中文路径, 光流优化
"""
import os
import queue
import threading
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Callable
import time

from .model_manager import RealESRGANInference
from .optical_flow import OpticalFlowSmoother

logger = logging.getLogger(__name__)

# 流水线队列缓冲大小 (帧数)
_PIPELINE_QUEUE_SIZE = 3
# 哨兵对象, 标记队列结束
_SENTINEL = None


def imread_unicode(path: str) -> np.ndarray:
    """读取视频/图片，兼容中文路径"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass
    return cv2.imread(path, cv2.IMREAD_COLOR)


class VideoUpscaler:
    """视频放大器 - 3阶段流水线架构"""

    def __init__(self, model_path: str, device_type: str = "cpu",
                 use_optical_flow: bool = True, alpha: float = 0.2):
        self.model_path = model_path
        self.device_type = device_type
        self.use_optical_flow = use_optical_flow
        self.alpha = alpha
        self.inference_engine: Optional[RealESRGANInference] = None
        self._should_stop = False  # 关键: 这个标志被外部控制

        self._init_engine()

    def _init_engine(self):
        try:
            self.inference_engine = RealESRGANInference(
                model_path=self.model_path,
                device_type=self.device_type
            )
            logger.info(f"视频放大器初始化成功 (后端: {self.inference_engine.backend})")
        except Exception as e:
            logger.error(f"视频放大器初始化失败: {e}")
            raise

    def stop(self):
        """外部调用此方法停止处理"""
        self._should_stop = True
        logger.info("视频处理停止请求已接收")

    # ============================================================
    #  3 阶段流水线: Reader → Infer+Smooth → Writer
    # ============================================================
    def _reader_thread(
        self,
        cap: cv2.VideoCapture,
        frame_queue: queue.Queue,
        total_frames: int,
    ):
        """
        Stage 1: 读帧线程
        将 VideoCapture 的读取与主线程的 GPU 推理并行化.
        """
        frame_idx = 0
        while not self._should_stop:
            ret, frame = cap.read()
            if not ret:
                frame_queue.put(_SENTINEL)
                break
            frame_queue.put((frame_idx, frame))
            frame_idx += 1
            # 背压控制: 如果推理跟不上, 读帧会自动阻塞在 queue.put 上

    def _writer_thread(
        self,
        out: cv2.VideoWriter,
        result_queue: queue.Queue,
    ):
        """
        Stage 3: 写帧线程
        将视频编码的写入与主线程的 GPU 推理并行化.
        """
        while True:
            item = result_queue.get()
            if item is _SENTINEL:
                result_queue.task_done()
                break
            out.write(item)
            result_queue.task_done()

    def _reader_thread_with_prefetch(
        self,
        cap: cv2.VideoCapture,
        frame_queue: queue.Queue,
    ):
        """
        带预读的读帧线程 (备用, 用于需要精确控制帧序的场景).
        一次性读取并缓存多帧, 减少 I/O 延迟.
        """
        while not self._should_stop:
            ret, frame = cap.read()
            if not ret:
                frame_queue.put(_SENTINEL)
                break
            frame_queue.put(frame)

    def upscale_video(
        self,
        input_path: str,
        output_path: str,
        scale_factor: int = 4,
        progress_callback: Optional[Callable[[float, int, int], None]] = None
    ) -> bool:
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise ValueError(f"无法打开视频: {input_path}")

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            logger.info(f"输入视频: {width}x{height}, {fps:.2f} FPS, {total_frames} 帧")

            output_width = width * scale_factor
            output_height = height * scale_factor

            # 资源检查: 如果输出分辨率超过系统能力, 降级处理
            try:
                from ..utils.system_info import get_system_resources
                res = get_system_resources()
                max_h, max_w = res.max_output_resolution
                if output_height > max_h or output_width > max_w:
                    max_scale_h = max_h // height
                    max_scale_w = max_w // width
                    safe_scale = max(min(max_scale_h, max_scale_w), 1)
                    if safe_scale < scale_factor:
                        logger.warning(
                            f"输出分辨率 {output_width}x{output_height} 超出系统能力 "
                            f"({max_w}x{max_h}), 自动降级到 {safe_scale}x"
                        )
                        scale_factor = safe_scale
                        output_width = width * scale_factor
                        output_height = height * scale_factor
            except Exception:
                pass

            fourcc = self._get_fourcc(output_path)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
            if not out.isOpened():
                alt_fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(output_path, alt_fourcc, fps, (output_width, output_height))
                if not out.isOpened():
                    raise ValueError(f"无法创建输出视频: {output_path}")

            # 光流平滑
            smoother = None
            if self.use_optical_flow:
                smoother = OpticalFlowSmoother(alpha=self.alpha, flow_scale=scale_factor)
                logger.info(f"光流时域平滑已启用 (alpha={self.alpha})")

            self._should_stop = False

            # ---- 创建流水线队列 ----
            frame_queue: queue.Queue = queue.Queue(maxsize=_PIPELINE_QUEUE_SIZE)
            result_queue: queue.Queue = queue.Queue(maxsize=_PIPELINE_QUEUE_SIZE)

            # ---- 启动 Stage 1 (读帧) 和 Stage 3 (写帧) 线程 ----
            reader = threading.Thread(
                target=self._reader_thread,
                args=(cap, frame_queue, total_frames),
                daemon=True,
            )
            writer = threading.Thread(
                target=self._writer_thread,
                args=(out, result_queue),
                daemon=True,
            )
            reader.start()
            writer.start()

            # ---- 主线程: Stage 2 (推理 + 光流) ----
            frame_count = 0
            start_time = time.time()
            last_log_time = start_time

            while not self._should_stop:
                # 从读帧队列获取下一帧 (阻塞直到有数据)
                try:
                    item = frame_queue.get(timeout=5.0)
                except queue.Empty:
                    # 超时: 可能读帧线程已结束但队列为空
                    break

                if item is _SENTINEL:
                    frame_queue.task_done()
                    break

                frame_idx, frame = item

                # 推理
                try:
                    super_frame = self.inference_engine.upscale_image(frame)
                except Exception as e:
                    logger.warning(f"帧 {frame_idx} 超分失败: {e}")
                    super_frame = cv2.resize(
                        frame, (output_width, output_height),
                        interpolation=cv2.INTER_LANCZOS4
                    )

                # 尺寸修正
                current_h, current_w = super_frame.shape[:2]
                if current_h != output_height or current_w != output_width:
                    super_frame = cv2.resize(
                        super_frame, (output_width, output_height),
                        interpolation=cv2.INTER_LANCZOS4
                    )

                # 光流时域平滑
                if smoother is not None:
                    super_frame = smoother.process_frame(frame, super_frame)

                # 放入写帧队列 (背压: 如果写帧跟不上, 这里会阻塞)
                result_queue.put(super_frame)

                frame_count += 1
                frame_queue.task_done()

                if progress_callback and total_frames > 0:
                    progress = (frame_count / total_frames) * 100
                    progress_callback(progress, frame_count, total_frames)

                now = time.time()
                if frame_count % 50 == 0 or (now - last_log_time) > 3.0:
                    elapsed = now - start_time
                    fps_proc = frame_count / elapsed if elapsed > 0 else 0
                    eta = (total_frames - frame_count) / fps_proc if fps_proc > 0 else 0
                    logger.info(
                        f"进度: {frame_count}/{total_frames} "
                        f"({frame_count/total_frames*100:.1f}%) | "
                        f"速度: {fps_proc:.2f} FPS | "
                        f"预计剩余: {eta:.0f}秒"
                    )
                    last_log_time = now

            # ---- 等待写帧线程完成 ----
            result_queue.put(_SENTINEL)  # 通知 writer 退出
            writer.join(timeout=30)
            reader.join(timeout=5)

            cap.release()
            out.release()

            total_time = time.time() - start_time
            avg_fps = frame_count / total_time if total_time > 0 else 0

            if self._should_stop:
                logger.info(f"视频处理已停止: 已处理 {frame_count} 帧")
                return False

            logger.info(
                f"视频放大完成: {output_path} | "
                f"{frame_count} 帧, {total_time:.1f}秒, {avg_fps:.2f} FPS"
            )
            return True

        except Exception as e:
            logger.error(f"视频放大失败: {e}", exc_info=True)
            return False

    def _get_fourcc(self, output_path: str) -> cv2.VideoWriter_fourcc:
        ext = Path(output_path).suffix.lower()
        codec_map = {
            '.mp4': cv2.VideoWriter_fourcc(*'mp4v'),
            '.avi': cv2.VideoWriter_fourcc(*'XVID'),
            '.mkv': cv2.VideoWriter_fourcc(*'mp4v'),
            '.mov': cv2.VideoWriter_fourcc(*'mp4v'),
        }
        return codec_map.get(ext, cv2.VideoWriter_fourcc(*'mp4v'))

    def get_video_info(self, video_path: str) -> dict:
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {}
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            info = {
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': fps,
                'total_frames': total_frames,
                'duration': total_frames / fps if fps > 0 else 0,
                'file_size': os.path.getsize(video_path) / (1024 * 1024)
            }
            cap.release()
            return info
        except Exception as e:
            logger.error(f"获取视频信息失败: {e}")
            return {}

    def __del__(self):
        self.inference_engine = None
