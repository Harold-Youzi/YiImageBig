"""
图片放大器 - 支持中文路径
"""
import os
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from .model_manager import RealESRGANInference

logger = logging.getLogger(__name__)


def imread_unicode(path: str) -> np.ndarray:
    """读取图片，兼容中文路径"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass
    return cv2.imread(path, cv2.IMREAD_COLOR)


def imwrite_unicode(path: str, img: np.ndarray, params=None) -> bool:
    """保存图片，兼容中文路径"""
    try:
        ext = os.path.splitext(path)[1]
        result, buf = cv2.imencode(ext, img, params or [])
        if result:
            buf.tofile(path)
            return True
    except Exception:
        pass
    return cv2.imwrite(path, img, params or [])


class ImageUpscaler:
    """图片放大器"""

    def __init__(self, model_path: str, device_type: str = "cpu"):
        self.model_path = model_path
        self.device_type = device_type
        self.inference_engine = None
        self._init_engine()

    def _init_engine(self):
        try:
            self.inference_engine = RealESRGANInference(
                model_path=self.model_path,
                device_type=self.device_type
            )
            logger.info(f"图片放大器初始化成功 (后端: {self.inference_engine.backend})")
        except Exception as e:
            logger.error(f"图片放大器初始化失败: {e}")
            raise

    def upscale_image(self, input_path: str, output_path: str,
                     scale_factor: int = 4,
                     output_format: str = "png",
                     on_tile_done=None) -> bool:
        """放大单张图片"""
        try:
            image = imread_unicode(input_path)
            if image is None:
                raise ValueError(f"无法读取图片: {input_path}")

            logger.info(f"读取图片: {input_path}, 尺寸: {image.shape}")

            upscaled_image = self.inference_engine.upscale_image(image, on_tile_done=on_tile_done)

            if scale_factor != 4:
                target_h = image.shape[0] * scale_factor
                target_w = image.shape[1] * scale_factor
                upscaled_image = cv2.resize(upscaled_image, (target_w, target_h),
                                          interpolation=cv2.INTER_LANCZOS4)

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            save_params = []
            if output_format.lower() in ("jpg", "jpeg"):
                save_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
            elif output_format.lower() == "png":
                save_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

            success = imwrite_unicode(output_path, upscaled_image, save_params)

            if success:
                logger.info(f"图片放大完成: {output_path}, 输出尺寸: {upscaled_image.shape}")
            else:
                logger.error(f"保存图片失败: {output_path}")

            return success

        except Exception as e:
            logger.error(f"图片放大失败: {e}", exc_info=True)
            return False

    def upscale_image_batch(self, input_dir: str, output_dir: str,
                           scale_factor: int = 4,
                           output_format: str = "png") -> Tuple[int, int]:
        """批量放大图片"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)

        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        for ext in supported_formats:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))

        total = len(image_files)
        success_count = 0

        for i, img_file in enumerate(image_files, 1):
            logger.info(f"处理进度: {i}/{total} - {img_file.name}")
            rel_path = img_file.relative_to(input_path)
            out_file = output_path / rel_path.with_suffix(f".{output_format}")
            if self.upscale_image(str(img_file), str(out_file), scale_factor, output_format):
                success_count += 1

        return success_count, total
