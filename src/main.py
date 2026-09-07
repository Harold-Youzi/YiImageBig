"""
YiImageBig - 模块入口
"""
import sys
import logging
from pathlib import Path

# 确保项目根目录在 Python 路径中
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from src.gui.main_window import MainWindow

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(project_root / "app.log", encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)


def main():
    """主函数"""
    try:
        app = QApplication(sys.argv)
        app.setApplicationName("YiImageBig")
        app.setOrganizationName("AIVideoUpscaler")
        app.setStyle("Fusion")

        window = MainWindow()
        window.show()

        logger.info("应用程序启动成功")
        sys.exit(app.exec())

    except Exception as e:
        logger.error(f"应用程序启动失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()