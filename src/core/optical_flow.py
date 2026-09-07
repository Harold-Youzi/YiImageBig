"""
光流时域平滑引擎
基于 Farneback 光流的双缓冲区对齐融合算法
解决 RealESRGAN 逐帧放大的画面闪烁问题
"""
import logging
import numpy as np
import cv2
from typing import Optional

logger = logging.getLogger(__name__)


class OpticalFlowSmoother:
    """
    光流时域平滑器

    算法 (参考 txt 文件):
    1. 输入: 当前低清帧 L_t, 上一低清帧 L_{t-1}, 上一超分输出 S_{t-1}
    2. 光流: F = CalcFlow(L_{t-1}, L_t), 放大 4 倍
    3. 变形: W = Warp(S_{t-1}, F_up)
    4. 遮罩: M (运动剧烈->0, 静止->1)
    5. 输出: S_t = M * (a*W + (1-a)*U_t) + (1-M) * U_t
    """

    def __init__(self, alpha: float = 0.2, flow_scale: int = 4):
        """
        Args:
            alpha: 融合权重 (0=完全当前帧, 1=完全上一帧), 建议 0.15~0.25
            flow_scale: 光流放大倍数 (应与超分倍数一致)
        """
        self.alpha = alpha
        self.flow_scale = flow_scale
        self.prev_low_res: Optional[np.ndarray] = None   # 上一低清帧 (灰度)
        self.prev_super_res: Optional[np.ndarray] = None  # 上一超分输出 (BGR)
        self.frame_count = 0

    def reset(self):
        """重置状态 (处理新视频时调用)"""
        self.prev_low_res = None
        self.prev_super_res = None
        self.frame_count = 0

    def process_frame(self, current_low_res: np.ndarray,
                      current_super_res: np.ndarray) -> np.ndarray:
        """
        处理单帧: 将当前超分输出与时域融合

        Args:
            current_low_res: 当前低清帧 (BGR uint8)
            current_super_res: 当前超分输出 (BGR uint8, 已放大 4 倍)

        Returns:
            平滑后的超分帧 (BGR uint8)
        """
        self.frame_count += 1

        # 第一帧直接返回
        if self.prev_low_res is None or self.prev_super_res is None:
            self._update_buffer(current_low_res, current_super_res)
            return current_super_res

        try:
            # 1. 计算低清帧间光流
            prev_gray = cv2.cvtColor(self.prev_low_res, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(current_low_res, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )

            h, w = flow.shape[:2]

            # 2. 将光流放大 flow_scale 倍
            flow_up = cv2.resize(flow, (w * self.flow_scale, h * self.flow_scale),
                                 interpolation=cv2.INTER_LINEAR)
            flow_up[:, :, 0] *= self.flow_scale
            flow_up[:, :, 1] *= self.flow_scale

            # 3. 生成网格坐标
            sh, sw = self.prev_super_res.shape[:2]
            yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)

            # 4. 将上一帧超分输出 warp 到当前帧
            map_x = xx + flow_up[:, :, 0]
            map_y = yy + flow_up[:, :, 1]

            warped = cv2.remap(
                self.prev_super_res, map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101
            )

            # 5. 计算遮罩 (运动剧烈区域->0, 静止区域->1)
            flow_magnitude = np.sqrt(flow_up[:, :, 0]**2 + flow_up[:, :, 1]**2)

            # 归一化到 [0, 1], 运动越大越接近 0
            max_motion = max(flow_magnitude.max(), 1.0)
            motion_normalized = np.clip(flow_magnitude / (max_motion * 0.5), 0, 1)

            # 遮罩: 运动小->1 (信任 warped), 运动大->0 (信任当前帧)
            mask = 1.0 - motion_normalized

            # 平滑遮罩 (避免硬边)
            mask = cv2.GaussianBlur(mask, (5, 5), 1.5)

            # 扩展到 3 通道
            mask_3ch = np.stack([mask] * 3, axis=-1)

            # 6. 融合输出
            blended = (mask_3ch * (self.alpha * warped + (1 - self.alpha) * current_super_res)
                       + (1 - mask_3ch) * current_super_res)

            result = np.clip(blended, 0, 255).astype(np.uint8)

            # 更新缓冲区
            self._update_buffer(current_low_res, current_super_res)

            return result

        except Exception as e:
            logger.warning(f"光流平滑失败 (帧 {self.frame_count}): {e}, 使用原始超分帧")
            self._update_buffer(current_low_res, current_super_res)
            return current_super_res

    def _update_buffer(self, low_res: np.ndarray, super_res: np.ndarray):
        """更新双缓冲区"""
        self.prev_low_res = low_res.copy()
        self.prev_super_res = super_res.copy()

    def __del__(self):
        self.prev_low_res = None
        self.prev_super_res = None
