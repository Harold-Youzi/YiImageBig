import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QPushButton, QFileDialog, QProgressBar,
    QStatusBar, QGroupBox, QRadioButton, QButtonGroup, QMessageBox,
    QLineEdit, QFrame, QTextEdit, QSplitter, QSizePolicy, QScrollArea
)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QFont, QIcon, QTextCursor

from ..utils.hardware_detector import get_hardware_detector, HardwareDevice
from ..core.image_upscaler import ImageUpscaler

logger = logging.getLogger(__name__)


# ============================================================
#  时间估算器
# ============================================================
class TimeEstimator:
    """基准测试 + EMA 滑动平均 预计时间估算

    - 首次运行/设备切换时跑 benchmark 得到 T_tile
    - 运行中每完成一个 tile, 用 EMA 更新: T_new = α * T_actual + (1-α) * T_old
    - 剩余时间 = 剩余 tile 数 × T_new
    """

    def __init__(self, t_tile: float = 0.2, alpha: float = 0.25):
        self.t_tile = t_tile          # 当前估算的单 tile 耗时 (秒)
        self.alpha = alpha            # EMA 平滑系数 (0.2~0.3)
        self.total_tiles = 0
        self.completed_tiles = 0
        self._start_time = 0.0

    def start(self, total_tiles: int):
        """开始新一轮处理"""
        self.total_tiles = total_tiles
        self.completed_tiles = 0
        self._start_time = time.perf_counter()

    def on_tile_done(self, completed: int, total: int):
        """每完成一个 tile 调用, 用 EMA 更新 T_tile"""
        self.total_tiles = total
        now = time.perf_counter()

        if self.completed_tiles > 0 and self.completed_tiles <= total:
            # 该 tile 的实际耗时 (用总经过时间 / 已完成数 近似)
            elapsed = now - self._start_time
            t_actual = elapsed / self.completed_tiles
            # EMA 修正
            self.t_tile = self.alpha * t_actual + (1 - self.alpha) * self.t_tile

        self.completed_tiles = completed

    def remaining_seconds(self) -> float:
        """剩余时间 (秒)"""
        remaining_tiles = max(0, self.total_tiles - self.completed_tiles)
        return remaining_tiles * self.t_tile

    def elapsed_seconds(self) -> float:
        """已用时间 (秒)"""
        if self._start_time == 0:
            return 0
        return time.perf_counter() - self._start_time

    def format_remaining(self) -> str:
        """格式化剩余时间: '预计剩余 2分30秒'"""
        remaining = self.remaining_seconds()
        elapsed = self.elapsed_seconds()

        if remaining < 1:
            return "即将完成..."
        elif remaining < 60:
            return f"预计剩余 {remaining:.0f}秒"
        elif remaining < 3600:
            m = int(remaining // 60)
            s = int(remaining % 60)
            return f"预计剩余 {m}分{s}秒"
        else:
            h = int(remaining // 3600)
            m = int((remaining % 3600) // 60)
            return f"预计剩余 {h}时{m}分"


# ============================================================
#  后台处理线程
# ============================================================
class UpscaleWorker(QThread):
    """后台放大处理线程"""
    progress_updated = Signal(float, int, int)  # 进度百分比, 当前帧, 总帧数
    processing_finished = Signal(bool, str)      # 是否成功, 消息
    status_updated = Signal(str)                 # 状态文本
    log_message = Signal(str)                    # 日志消息
    eta_updated = Signal(str)                    # 预计时间文本

    def __init__(self, mode: str, input_path: str, output_path: str,
                 device_type: str, scale_factor: int, model_path: str,
                 t_tile: float = 0.2):
        super().__init__()
        self.mode = mode
        self.input_path = input_path
        self.output_path = output_path
        self.device_type = device_type
        self.scale_factor = scale_factor
        self.model_path = model_path
        self.t_tile = t_tile  # 基准测试的 T_tile
        self._should_stop = False
        self._upscaler = None

    def run(self):
        """执行放大任务"""
        try:
            if self.mode == "image":
                self._process_image()
            elif self.mode == "video":
                self._process_video()
            else:
                self.processing_finished.emit(False, f"未知模式: {self.mode}")
        except Exception as e:
            logger.error(f"处理失败: {e}", exc_info=True)
            self.processing_finished.emit(False, f"处理失败: {str(e)}")

    def _process_image(self):
        """处理图片"""
        import time
        start_time = time.time()

        self.status_updated.emit("正在初始化图片放大器...")
        self.log_message.emit(f"═══════════════════════════════════════")
        self.log_message.emit(f"设备类型: {self.device_type}")
        self.log_message.emit(f"模型路径: {self.model_path}")
        self.log_message.emit(f"基准T_tile: {self.t_tile*1000:.0f}ms")

        upscaler = ImageUpscaler(self.model_path, self.device_type)
        self._upscaler = upscaler

        # 输出实际使用的硬件后端信息 (防止 fallback 用户未知)
        engine = upscaler.inference_engine
        actual_backend = getattr(engine, 'backend', self.device_type)
        actual_device = getattr(engine, 'device_type', self.device_type)
        self.log_message.emit(f"实际后端: {actual_backend}")
        self.log_message.emit(f"实际设备: {actual_device}")
        self.log_message.emit(f"Tile尺寸: {getattr(engine, 'tile_size', '?')} | "
                              f"Overlap: {getattr(engine, 'tile_overlap', '?')}")
        self.log_message.emit(f"═══════════════════════════════════════")

        # 预估 tile 数量 (用于 ETA)
        try:
            from ..core.image_upscaler import imread_unicode
            image = imread_unicode(self.input_path)
            h, w = image.shape[:2]
            tile_sz = getattr(engine, 'tile_size', 256)
            ol = getattr(engine, 'tile_overlap', 32)
            step = tile_sz - ol
            tiles_y = max(1, (h + step - 1) // step)
            tiles_x = max(1, (w + step - 1) // step)
            total_tiles = tiles_y * tiles_x
        except Exception:
            total_tiles = 1

        # 创建 EMA 估算器
        estimator = TimeEstimator(t_tile=self.t_tile)
        estimator.start(total_tiles)
        self.eta_updated.emit(estimator.format_remaining())

        self.status_updated.emit("正在放大图片...")
        self.log_message.emit(f"输入: {self.input_path} ({w}x{h}, ~{total_tiles}个tile)")

        # 完成回调: 更新进度 + ETA
        def on_tile_done(completed, total):
            estimator.on_tile_done(completed, total)
            pct = completed / total * 100 if total > 0 else 0
            self.progress_updated.emit(pct, completed, total)
            self.eta_updated.emit(estimator.format_remaining())

        success = upscaler.upscale_image(
            self.input_path,
            self.output_path,
            self.scale_factor,
            on_tile_done=on_tile_done
        )

        elapsed = time.time() - start_time

        if success:
            self.progress_updated.emit(100, 1, 1)
            self.log_message.emit(f"输出: {self.output_path}")
            self.log_message.emit(f"耗时统计: 总耗时 {elapsed:.2f}s | "
                                  f"平均 {estimator.t_tile*1000:.0f}ms/tile | "
                                  f"{total_tiles} tiles")
            self.eta_updated.emit(f"✅ 完成! 耗时 {elapsed:.1f}秒")
            self.processing_finished.emit(True, f"✅ 图片放大完成!\n输出文件: {self.output_path}\n耗时: {elapsed:.1f}秒")
        else:
            self.eta_updated.emit("❌ 失败")
            self.processing_finished.emit(False, "❌ 图片放大失败，请查看日志了解详情")

    def _process_video(self):
        """视频放大功能尚在开发中"""
        self.processing_finished.emit(
            False,
            "视频放大功能尚在开发中，当前版本仅支持图片放大。\n"
            "请切换到图片模式后重试。"
        )

    def stop(self):
        """请求停止处理 - 同时传递给 upscaler"""
        self._should_stop = True
        if self._upscaler is not None:
            self._upscaler.stop()


# ============================================================
#  主窗口
# ============================================================
class MainWindow(QMainWindow):
    """YiImageBig主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("YiImageBig - RealESRGAN x4 Plus")
        self.setMinimumSize(640, 480)
        self.resize(800, 650)

        # 模型路径 (默认，会被 run.py 覆盖)
        self.base_dir = str(Path(__file__).parent.parent.parent)
        self.model_path = Path(self.base_dir) / "models" / "RealESRGAN_x4plus.pth"

        # 处理线程
        self.worker_thread: Optional[UpscaleWorker] = None
        self._t_tile = 0.2  # 基准测试的 T_tile, 默认 0.2s

        # 硬件检测器
        self.hw_detector = get_hardware_detector()

        # 初始化UI
        self._init_ui()

        # 搜索模型文件
        self._update_model_path()

        # 检测硬件 (会自动触发基准测试)
        self._detect_hardware()

        # 连接信号
        self._connect_signals()

        # 初始模式
        self._on_mode_changed()

    # ----------------------------------------------------------
    #  UI 初始化
    # ----------------------------------------------------------
    def _init_ui(self):
        """初始化用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 外层布局: 内容区可滚动
        outer_layout = QVBoxLayout(central_widget)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        scroll_content = QWidget()
        main_layout = QVBoxLayout(scroll_content)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(20, 10, 20, 10)

        # ---- 标题 ----
        title = QLabel("🔬 YiImageBig - AI 多媒体放大工具")
        title.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #2c3e50; padding: 2px;")
        main_layout.addWidget(title)

        subtitle = QLabel("基于 RealESRGAN x4plus | CPU / NVIDIA GPU / Intel GPU / Intel NPU / DirectML")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #7f8c8d; font-size: 11px; padding: 0 0 4px 0;")
        main_layout.addWidget(subtitle)

        # ---- 模式选择 ----
        mode_group = QGroupBox("📋 处理模式")
        mode_layout = QHBoxLayout(mode_group)

        self.image_mode_radio = QRadioButton("🖼️  图片放大")
        self.video_mode_radio = QRadioButton("🎬 视频放大")
        self.image_mode_radio.setChecked(True)
        self.image_mode_radio.setFont(QFont("Microsoft YaHei", 11))
        self.video_mode_radio.setFont(QFont("Microsoft YaHei", 11))

        mode_layout.addWidget(self.image_mode_radio)
        mode_layout.addWidget(self.video_mode_radio)

        self.mode_button_group = QButtonGroup()
        self.mode_button_group.addButton(self.image_mode_radio, 0)
        self.mode_button_group.addButton(self.video_mode_radio, 1)

        main_layout.addWidget(mode_group)

        # ---- 文件选择 ----
        file_group = QGroupBox("📁 文件选择")
        file_layout = QGridLayout(file_group)
        file_layout.setColumnStretch(1, 1)

        # 输入文件
        file_layout.addWidget(QLabel("输入文件:"), 0, 0)
        self.input_path_edit = QLineEdit()
        self.input_path_edit.setPlaceholderText('点击"浏览"选择输入文件...')
        self.input_path_edit.setReadOnly(True)
        file_layout.addWidget(self.input_path_edit, 0, 1)

        self.input_browse_btn = QPushButton("📂 浏览...")
        self.input_browse_btn.setFixedWidth(100)
        file_layout.addWidget(self.input_browse_btn, 0, 2)

        # 输出文件
        file_layout.addWidget(QLabel("输出文件:"), 1, 0)
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText('点击"浏览"选择输出位置...')
        self.output_path_edit.setReadOnly(True)
        file_layout.addWidget(self.output_path_edit, 1, 1)

        self.output_browse_btn = QPushButton("📂 浏览...")
        self.output_browse_btn.setFixedWidth(100)
        file_layout.addWidget(self.output_browse_btn, 1, 2)

        main_layout.addWidget(file_group)

        # ---- 设置 ----
        settings_group = QGroupBox("⚙️ 放大设置")
        settings_layout = QGridLayout(settings_group)

        # 硬件选择
        settings_layout.addWidget(QLabel("硬件模式:"), 0, 0)
        self.hardware_combo = QComboBox()
        self.hardware_combo.setFont(QFont("Microsoft YaHei", 10))
        settings_layout.addWidget(self.hardware_combo, 0, 1)

        self.hardware_info_label = QLabel("")
        self.hardware_info_label.setStyleSheet("color: #7f8c8d; font-size: 10px;")
        settings_layout.addWidget(self.hardware_info_label, 0, 2)

        # 放大倍数
        settings_layout.addWidget(QLabel("放大倍数:"), 1, 0)
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(["2x", "3x", "4x"])
        self.scale_combo.setCurrentIndex(2)  # 默认4x
        self.scale_combo.setFont(QFont("Microsoft YaHei", 10))
        settings_layout.addWidget(self.scale_combo, 1, 1)

        main_layout.addWidget(settings_group)

        # ---- 控制按钮 ----
        btn_layout = QHBoxLayout()

        self.start_btn = QPushButton("🚀 开始处理")
        self.start_btn.setMinimumHeight(36)
        self.start_btn.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 6px 24px;
            }
            QPushButton:hover { background-color: #2ecc71; }
            QPushButton:pressed { background-color: #1e8449; }
            QPushButton:disabled { background-color: #bdc3c7; color: #7f8c8d; }
        """)

        self.stop_btn = QPushButton("⏹  停止处理")
        self.stop_btn.setMinimumHeight(36)
        self.stop_btn.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 30px;
            }
            QPushButton:hover { background-color: #c0392b; }
            QPushButton:pressed { background-color: #a93226; }
            QPushButton:disabled { background-color: #bdc3c7; color: #7f8c8d; }
        """)

        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        main_layout.addLayout(btn_layout)

        # ---- 进度条 ----
        progress_group = QGroupBox("📊 处理进度")
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMinimumHeight(28)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                text-align: center;
                font-size: 12px;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 4px;
            }
        """)
        progress_layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("✅ 准备就绪 - 请选择输入文件")
        self.progress_label.setFont(QFont("Microsoft YaHei", 10))
        progress_layout.addWidget(self.progress_label)

        # ---- 预计时间 ----
        self.eta_label = QLabel("")
        self.eta_label.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        self.eta_label.setAlignment(Qt.AlignCenter)
        self.eta_label.setStyleSheet("color: #e67e22; font-size: 12px; padding: 2px;")
        self.eta_label.setVisible(False)
        progress_layout.addWidget(self.eta_label)

        main_layout.addWidget(progress_group)

        # ---- 日志输出 ----
        log_group = QGroupBox("📝 运行日志")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(100)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4;")
        log_layout.addWidget(self.log_text)

        main_layout.addWidget(log_group)

        # 添加弹簧让内容紧凑
        main_layout.addStretch(1)

        scroll.setWidget(scroll_content)
        outer_layout.addWidget(scroll)

        # ---- 状态栏 ----
        self.statusBar().showMessage("就绪")
        self.statusBar().setStyleSheet("font-size: 11px;")

        # ---- 全局样式 ----
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 12px;
                border: 1px solid #dcdde1;
                border-radius: 6px;
                margin-top: 8px;
                padding: 8px 6px 6px 6px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }
            QLineEdit {
                padding: 6px 8px;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                font-size: 11px;
            }
            QComboBox {
                padding: 6px 8px;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                font-size: 11px;
                min-width: 200px;
            }
            QRadioButton {
                spacing: 8px;
            }
        """)

    # ----------------------------------------------------------
    #  硬件检测
    # ----------------------------------------------------------
    def _detect_hardware(self):
        """检测硬件并填充下拉框"""
        self.hardware_combo.clear()

        available_devices = self.hw_detector.get_available_devices()

        for device in available_devices:
            display_name = self.hw_detector.get_device_display_name(device)
            # 在下拉框中显示设备类型和全名
            combo_text = f"{display_name} - {device.full_name}"
            self.hardware_combo.addItem(combo_text, device.device_type)

        # 智能选择最优设备
        optimal = self.hw_detector.select_optimal_device()
        idx = self.hardware_combo.findData(optimal.device_type)
        if idx >= 0:
            self.hardware_combo.setCurrentIndex(idx)

        # 显示硬件信息
        self.hardware_info_label.setText(
            f"检测到 {len(available_devices)} 个设备 | 推荐: {self.hw_detector.get_device_display_name(optimal)}"
        )

        # 如果硬件combo改变，更新信息
        self.hardware_combo.currentIndexChanged.connect(self._on_hardware_changed)

        logger.info(f"硬件检测完成，共 {len(available_devices)} 个可用设备")

    def _on_hardware_changed(self, index: int):
        """硬件选择改变 → 运行基准测试"""
        device_type = self.hardware_combo.currentData()
        if device_type:
            for dev in self.hw_detector.get_available_devices():
                if dev.device_type == device_type:
                    self.hardware_info_label.setText(
                        f"当前: {dev.full_name} | 评分: {dev.score}"
                    )
                    break
            # 触发基准测试
            self._run_benchmark(device_type)

    def _run_benchmark(self, device_type: str):
        """后台线程运行基准测试, 得到 T_tile"""
        class BenchmarkThread(QThread):
            finished = Signal(float)
            log = Signal(str)

            def __init__(self, model_path, device_type):
                super().__init__()
                self.model_path = model_path
                self.device_type = device_type

            def run(self):
                try:
                    from ..core.model_manager import RealESRGANInference
                    engine = RealESRGANInference(self.model_path, self.device_type)
                    t_tile = engine.benchmark(warmup=2)
                    self.finished.emit(t_tile)
                except Exception as e:
                    self.log.emit(f"基准测试失败: {e}, 使用默认值 0.2s")
                    self.finished.emit(0.2)

        self._bench_thread = BenchmarkThread(str(self.model_path), device_type)
        self._bench_thread.log.connect(self._append_log)
        self._bench_thread.finished.connect(self._on_benchmark_done)
        self._bench_thread.start()

    @Slot(float)
    def _on_benchmark_done(self, t_tile: float):
        """基准测试完成"""
        self._t_tile = t_tile
        self._append_log(f"基准测试完成: T_tile = {t_tile*1000:.0f}ms")

    # ----------------------------------------------------------
    #  动态路径
    # ----------------------------------------------------------
    def _update_model_path(self):
        """搜索模型文件 (.xml > .onnx > .pth), 优先 models/ 目录"""
        base = Path(self.base_dir)
        exe_dir = Path(os.path.dirname(os.path.abspath(
            sys.executable if getattr(sys, 'frozen', False) else __file__
        )))
        # 搜索顺序: models/ -> 项目根目录 -> exe 同目录
        search_dirs = [base / 'models', base, exe_dir]
        for search_dir in search_dirs:
            for name in ['RealESRGAN_x4plus']:
                for ext in ['.xml', '.onnx', '.pth']:
                    candidate = search_dir / f'{name}{ext}'
                    if candidate.exists():
                        self.model_path = candidate
                        logger.info(f"模型路径更新为: {self.model_path}")
                        return
        logger.warning(f"未找到模型文件 (搜索目录: {[str(d) for d in search_dirs]})")

    # ----------------------------------------------------------
    #  信号连接
    # ----------------------------------------------------------
    def _connect_signals(self):
        """连接信号和槽"""
        self.input_browse_btn.clicked.connect(self._browse_input_file)
        self.output_browse_btn.clicked.connect(self._browse_output_file)
        self.mode_button_group.buttonClicked.connect(self._on_mode_changed)
        self.start_btn.clicked.connect(self._start_processing)
        self.stop_btn.clicked.connect(self._stop_processing)
        self.input_path_edit.textChanged.connect(self._update_output_path)

    # ----------------------------------------------------------
    #  文件浏览
    # ----------------------------------------------------------
    def _browse_input_file(self):
        """浏览输入文件"""
        if self.image_mode_radio.isChecked():
            file_filter = (
                "图片文件 (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp);;"
                "所有文件 (*)"
            )
        else:
            file_filter = (
                "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm);;"
                "所有文件 (*)"
            )

        file_path, _ = QFileDialog.getOpenFileName(self, "选择输入文件", "", file_filter)
        if file_path:
            self.input_path_edit.setText(file_path)
            self._append_log(f"已选择输入文件: {file_path}")

    def _browse_output_file(self):
        """浏览输出文件"""
        if self.image_mode_radio.isChecked():
            file_filter = "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;所有文件 (*)"
            default_ext = ".png"
        else:
            file_filter = "MP4 (*.mp4);;AVI (*.avi);;MKV (*.mkv);;所有文件 (*)"
            default_ext = ".mp4"

        # 默认输出文件名
        input_path = self.input_path_edit.text()
        if input_path:
            default_dir = str(Path(input_path).parent)
            default_name = Path(input_path).stem + "_upscaled_4x" + default_ext
        else:
            default_dir = ""
            default_name = ""

        file_path, _ = QFileDialog.getSaveFileName(
            self, "选择输出文件",
            os.path.join(default_dir, default_name),
            file_filter
        )
        if file_path:
            self.output_path_edit.setText(file_path)

    def _update_output_path(self, input_path: str):
        """根据输入路径自动更新输出路径"""
        if not input_path:
            return
        if self.output_path_edit.text():  # 用户已手动设置
            return

        if self.image_mode_radio.isChecked():
            default_ext = ".png"
        else:
            default_ext = ".mp4"

        p = Path(input_path)
        output = p.parent / (p.stem + "_upscaled_4x" + default_ext)
        self.output_path_edit.setText(str(output))

    def _get_unique_output_path(self, output_path: str) -> str:
        """生成不重名的输出路径: file.png -> file-副本1.png -> file-副本2.png"""
        p = Path(output_path)
        stem = p.stem
        suffix = p.suffix
        parent = p.parent

        # 如果已有 -副本N 标记, 先去掉
        import re
        match = re.match(r'^(.+)-副本(\d+)$', stem)
        if match:
            base_stem = match.group(1)
            start = int(match.group(2)) + 1
        else:
            base_stem = stem
            start = 1

        for i in range(start, start + 100):
            new_name = f"{base_stem}-副本{i}{suffix}"
            new_path = parent / new_name
            if not new_path.exists():
                return str(new_path)

        # 极端情况: 用时间戳
        import time
        ts = time.strftime("%H%M%S")
        return str(parent / f"{base_stem}-副本{ts}{suffix}")

    # ----------------------------------------------------------
    #  模式切换
    # ----------------------------------------------------------
    def _on_mode_changed(self):
        """模式切换"""
        # 如果用户选择了视频模式，弹窗提示并切回图片模式
        if self.video_mode_radio.isChecked():
            QMessageBox.information(
                self, "功能提示",
                "视频放大功能尚在开发中，已自动切换回图片模式。\n\n"
                "当前版本仅支持图片放大功能，敬请期待后续版本。"
            )
            # 切回图片模式 (断开信号避免递归)
            self.mode_button_group.blockSignals(True)
            self.image_mode_radio.setChecked(True)
            self.mode_button_group.blockSignals(False)

        self.input_path_edit.clear()
        self.output_path_edit.clear()
        self.progress_bar.setValue(0)

        if self.image_mode_radio.isChecked():
            self.progress_label.setText("🖼️ 图片放大模式 - 请选择输入图片")
            self.statusBar().showMessage("已切换到图片放大模式")

    # ----------------------------------------------------------
    #  开始/停止处理
    # ----------------------------------------------------------
    def _start_processing(self):
        """开始处理"""
        input_path = self.input_path_edit.text().strip()
        output_path = self.output_path_edit.text().strip()

        # 验证
        if not input_path:
            QMessageBox.warning(self, "⚠️ 提示", "请选择输入文件")
            return
        if not output_path:
            QMessageBox.warning(self, "⚠️ 提示", "请选择输出文件")
            return
        if not os.path.exists(input_path):
            QMessageBox.warning(self, "⚠️ 提示", f"输入文件不存在:\n{input_path}")
            return
        if not self.model_path.exists():
            # 搜索所有格式
            base = self.model_path.parent
            found = False
            for ext in ['.xml', '.onnx', '.pth']:
                if (base / f'{self.model_path.stem}{ext}').exists():
                    self.model_path = base / f'{self.model_path.stem}{ext}'
                    found = True
                    break
            if not found:
                QMessageBox.warning(
                    self, '提示',
                    f'模型文件不存在:\n{base / "RealESRGAN_x4plus.*"}\n\n'
                    '请确保模型文件 (xml/bin, onnx, 或 pth) 在程序同目录下'
                )
                return

        # 检查输出文件是否已存在
        if os.path.exists(output_path):
            reply = QMessageBox.question(
                self, "文件已存在",
                f"输出文件已存在:\n{output_path}\n\n是否覆盖？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                # 自动加 -副本N 标记
                output_path = self._get_unique_output_path(output_path)
                self.output_path_edit.setText(output_path)
                self._append_log(f"输出路径调整为: {output_path}")

        # 获取设置
        device_type = self.hardware_combo.currentData()
        scale_text = self.scale_combo.currentText()
        scale_factor = int(scale_text.replace("x", ""))
        mode = "image"  # 当前版本仅支持图片放大

        self._append_log(f"开始处理: 模式={mode}, 设备={device_type}, 放大={scale_text}")
        self._append_log(f"输入: {input_path}")
        self._append_log(f"输出: {output_path}")

        # 创建并启动线程
        self.worker_thread = UpscaleWorker(
            mode=mode,
            input_path=input_path,
            output_path=output_path,
            device_type=device_type,
            scale_factor=scale_factor,
            model_path=str(self.model_path),
            t_tile=self._t_tile
        )

        self.worker_thread.progress_updated.connect(self._on_progress)
        self.worker_thread.processing_finished.connect(self._on_finished)
        self.worker_thread.status_updated.connect(self._on_status)
        self.worker_thread.log_message.connect(self._append_log)
        self.worker_thread.eta_updated.connect(self._on_eta)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.hardware_combo.setEnabled(False)
        self.scale_combo.setEnabled(False)
        self.progress_bar.setValue(0)

        self.worker_thread.start()
        self.statusBar().showMessage("⏳ 正在处理...")

    def _stop_processing(self):
        """停止处理"""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.stop()
            self.statusBar().showMessage("⏹ 正在停止...")
            self._append_log("用户请求停止处理")

    # ----------------------------------------------------------
    #  槽函数
    # ----------------------------------------------------------
    @Slot(float, int, int)
    def _on_progress(self, progress: float, current_frame: int, total_frames: int):
        self.progress_bar.setValue(int(progress))
        self.progress_label.setText(
            f"进度: {progress:.1f}% | Tile: {current_frame}/{total_frames}"
        )

    @Slot(str)
    def _on_eta(self, eta_text: str):
        """更新预计时间显示"""
        if eta_text:
            self.eta_label.setVisible(True)
            self.eta_label.setText(eta_text)
        else:
            self.eta_label.setVisible(False)

    @Slot(bool, str)
    def _on_finished(self, success: bool, message: str):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.hardware_combo.setEnabled(True)
        self.scale_combo.setEnabled(True)

        if success:
            self.progress_bar.setValue(100)
            self.progress_label.setText("🎉 处理完成!")
            self.statusBar().showMessage("✅ 处理完成")
            self._append_log(message)
            QMessageBox.information(self, "完成", message)
        else:
            self.progress_bar.setValue(0)
            self.progress_label.setText("❌ 处理失败")
            self.statusBar().showMessage("❌ 处理失败")
            self._append_log(message)
            QMessageBox.critical(self, "错误", message)

        self.eta_label.setVisible(False)

        if self.worker_thread:
            self.worker_thread.deleteLater()
            self.worker_thread = None

    @Slot(str)
    def _on_status(self, status: str):
        self.progress_label.setText(status)

    # ----------------------------------------------------------
    #  日志
    # ----------------------------------------------------------
    def _append_log(self, message: str):
        """追加日志"""
        self.log_text.append(message)
        # 自动滚动到底部
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_text.setTextCursor(cursor)

    # ----------------------------------------------------------
    #  关闭事件
    # ----------------------------------------------------------
    def closeEvent(self, event):
        if self.worker_thread and self.worker_thread.isRunning():
            reply = QMessageBox.question(
                self, '确认退出',
                '处理正在运行中，确定要退出吗？',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.worker_thread.stop()
                self.worker_thread.wait(3000)
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()