"""
YiImageBig - 启动脚本
基于 RealESRGAN_x4plus 模型的图片&视频放大工具
支持硬件: CPU / NVIDIA GPU / Intel ARC GPU / Intel NPU
"""
import sys
import os
import logging

# ============================================================
#  PyInstaller 兼容：获取正确的基础路径
# ============================================================
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
    BUNDLE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BUNDLE_DIR = BASE_DIR

sys.path.insert(0, BUNDLE_DIR)

# ============================================================
#  Windows DLL 搜索路径修复
# ============================================================
def _setup_dll_paths():
    if not getattr(sys, 'frozen', False):
        return
    dirs_to_add = [
        BUNDLE_DIR,
        os.path.join(BUNDLE_DIR, 'torch', 'lib'),
        os.path.join(BUNDLE_DIR, 'openvino', 'libs'),
        os.path.join(BUNDLE_DIR, 'openvino'),
    ]
    for d in dirs_to_add:
        d = os.path.abspath(d)
        if os.path.isdir(d):
            try:
                if sys.version_info >= (3, 8) and hasattr(os, 'add_dll_directory'):
                    os.add_dll_directory(d)
            except Exception:
                pass
    path_parts = []
    for d in dirs_to_add:
        d = os.path.abspath(d)
        if os.path.isdir(d) and d not in os.environ.get('PATH', ''):
            path_parts.append(d)
    if path_parts:
        os.environ['PATH'] = ';'.join(path_parts) + ';' + os.environ.get('PATH', '')

_setup_dll_paths()

# ============================================================
#  顶层导入 - 确保 PyInstaller 静态分析能找到
# ============================================================
from src.utils.hardware_detector import get_hardware_detector
from src.gui.main_window import MainWindow

# ============================================================
#  主函数
# ============================================================
def main():
    log_path = os.path.join(BASE_DIR, "app.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding='utf-8')
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("YiImageBig 启动中...")
    logger.info(f"运行模式: {'打包模式' if getattr(sys, 'frozen', False) else '开发模式'}")
    logger.info(f"BASE_DIR: {BASE_DIR}")
    logger.info(f"BUNDLE_DIR: {BUNDLE_DIR}")
    logger.info("=" * 60)

    # 检测硬件
    detector = get_hardware_detector()
    logger.info(detector.get_device_summary())
    optimal = detector.select_optimal_device()
    logger.info(f"推荐设备: {detector.get_device_display_name(optimal)}")

    # 启动 GUI
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("YiImageBig")
    app.setOrganizationName("AIVideoUpscaler")
    app.setStyle("Fusion")

    window = MainWindow()
    window.base_dir = BASE_DIR
    window._update_model_path()
    window.show()

    logger.info("GUI 窗口已显示")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
