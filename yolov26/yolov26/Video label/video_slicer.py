import sys
import os
import json
import time

os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)

try:
    import cv2
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("警告: OpenCV未安装，视频处理功能将不可用")

pyqt5_path = os.path.dirname(__import__('PyQt5').__file__)
qt_plugins_path = os.path.join(pyqt5_path, 'Qt5', 'plugins')
if os.path.exists(qt_plugins_path):
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_plugins_path
else:
    qt_plugins_path = os.path.join(pyqt5_path, 'Qt', 'plugins')
    if os.path.exists(qt_plugins_path):
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_plugins_path

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QFileDialog, QProgressBar,
                             QStatusBar, QSplitter, QGroupBox, QComboBox, QSlider,
                             QMessageBox, QTextEdit, QCheckBox)
from PyQt5.QtCore import Qt, QTimer, QSize, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont


class VideoThread(QThread):
    frame_signal = pyqtSignal(QImage)
    time_signal = pyqtSignal(float)
    finished_signal = pyqtSignal()

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.running = False
        self.paused = False
        self.cap = None
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 0
        self.speed = 1.0

    def run(self):
        self.running = True
        self.paused = False
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            self.finished_signal.emit()
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        frame_interval = 1.0
        last_time = time.time()

        while self.running:
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                current_time = self.current_frame / self.fps
                
                if self.speed >= 1.0:
                    frame_interval = int(self.speed)
                    for _ in range(frame_interval - 1):
                        if self.running and not self.paused:
                            self.cap.read()
                else:
                    frame_interval = int(1.0 / self.speed)
                    for _ in range(frame_interval - 1):
                        if self.running and not self.paused:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            h, w, ch = frame_rgb.shape
                            bytes_per_line = ch * w
                            q_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                            self.frame_signal.emit(q_image)
                            self.time_signal.emit(current_time)
                            elapsed = time.time() - last_time
                            sleep_time = max(0, (1 / self.fps) - elapsed)
                            time.sleep(sleep_time)
                            last_time = time.time()
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                bytes_per_line = ch * w
                q_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                
                self.frame_signal.emit(q_image)
                self.time_signal.emit(current_time)
                
                elapsed = time.time() - last_time
                sleep_time = max(0, (1 / (self.fps * self.speed)) - elapsed)
                time.sleep(sleep_time)
                last_time = time.time()
            else:
                time.sleep(0.01)

        if self.cap:
            self.cap.release()
        self.finished_signal.emit()

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def stop(self):
        self.running = False
        self.paused = False

    def set_speed(self, speed):
        self.speed = speed

    def get_current_time(self):
        if self.cap:
            return self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        return 0

    def set_position(self, frame_number):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.current_frame = frame_number

    def seek_by_time(self, seconds):
        if self.cap:
            current_time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            new_time = max(0, min(current_time + seconds, self.get_total_time()))
            self.cap.set(cv2.CAP_PROP_POS_MSEC, new_time * 1000)
            self.current_frame = int(new_time * self.fps)

    def get_total_time(self):
        if self.fps > 0 and self.total_frames > 0:
            return self.total_frames / self.fps
        return 0


class ConfigManager:
    def __init__(self, config_file="config.json"):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self):
        default_config = {
            "last_source_video": "",
            "last_label_video": "",
            "last_output_path": "",
            "output_format": "png"
        }
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return {**default_config, **json.load(f)}
            except:
                return default_config
        return default_config

    def save_config(self):
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存配置失败: {e}")

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value
        self.save_config()


class VideoSlicer(QMainWindow):
    VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.mpg', '.mpeg')
    
    def __init__(self):
        super().__init__()
        self.config = ConfigManager()
        self.video_thread = None
        self.source_video_path = ""
        self.label_video_path = ""
        self.output_path = ""
        
        self.source_files = []
        self.label_files = []
        self.original_source_files = []
        self.original_label_files = []
        self.current_index = 0
        self.source_is_folder = False
        self.label_is_folder = False
        self.sort_match_enabled = False
        
        self.init_ui()
        self.load_saved_paths()

    def init_ui(self):
        self.setWindowTitle("视频切片工具")
        self.setMinimumSize(QSize(1000, 600))
        
        self.splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(self.splitter)

        font = QFont()
        font.setPointSize(11)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_widget.setMinimumWidth(250)
        left_widget.setMaximumWidth(500)

        file_group = QGroupBox("文件路径设置")
        file_group.setFont(font)
        file_layout = QVBoxLayout(file_group)
        file_layout.setSpacing(8)

        source_layout = QHBoxLayout()
        source_label = QLabel("原视频:")
        source_label.setFixedWidth(65)
        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("选择文件或文件夹")
        source_file_btn = QPushButton("文件")
        source_file_btn.setFixedWidth(55)
        source_file_btn.clicked.connect(lambda: self.select_source_video(False))
        source_folder_btn = QPushButton("文件夹")
        source_folder_btn.setFixedWidth(60)
        source_folder_btn.clicked.connect(lambda: self.select_source_video(True))
        source_layout.addWidget(source_label)
        source_layout.addWidget(self.source_edit)
        source_layout.addWidget(source_file_btn)
        source_layout.addWidget(source_folder_btn)
        file_layout.addLayout(source_layout)

        label_layout = QHBoxLayout()
        label_label = QLabel("标注视频:")
        label_label.setFixedWidth(65)
        self.label_edit = QLineEdit()
        self.label_edit.setPlaceholderText("选择文件或文件夹")
        label_file_btn = QPushButton("文件")
        label_file_btn.setFixedWidth(55)
        label_file_btn.clicked.connect(lambda: self.select_label_video(False))
        label_folder_btn = QPushButton("文件夹")
        label_folder_btn.setFixedWidth(60)
        label_folder_btn.clicked.connect(lambda: self.select_label_video(True))
        label_layout.addWidget(label_label)
        label_layout.addWidget(self.label_edit)
        label_layout.addWidget(label_file_btn)
        label_layout.addWidget(label_folder_btn)
        file_layout.addLayout(label_layout)

        match_layout = QHBoxLayout()
        self.sort_match_check = QCheckBox("默认排序匹配")
        self.sort_match_check.stateChanged.connect(self.toggle_sort_match)
        match_layout.addWidget(self.sort_match_check)
        match_layout.addStretch()
        file_layout.addLayout(match_layout)

        output_layout = QHBoxLayout()
        output_label = QLabel("输出路径:")
        output_label.setFixedWidth(65)
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("选择保存路径")
        output_btn = QPushButton("浏览")
        output_btn.setFixedWidth(55)
        output_btn.clicked.connect(self.select_output_path)
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_edit)
        output_layout.addWidget(output_btn)
        file_layout.addLayout(output_layout)

        format_layout = QHBoxLayout()
        format_label = QLabel("输出格式:")
        format_label.setFixedWidth(65)
        self.format_combo = QComboBox()
        self.format_combo.addItems(["PNG", "JPG", "BMP", "TIFF"])
        self.format_combo.setFixedWidth(80)
        format_layout.addWidget(format_label)
        format_layout.addWidget(self.format_combo)
        format_layout.addStretch()
        file_layout.addLayout(format_layout)

        left_layout.addWidget(file_group)
        left_layout.addStretch()

        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(5)

        video_group = QGroupBox("视频预览与控制")
        video_group.setFont(font)
        video_layout = QVBoxLayout(video_group)
        video_layout.setContentsMargins(5, 5, 5, 5)
        video_layout.setSpacing(5)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #000000;")
        video_layout.addWidget(self.video_label, stretch=1)

        control_layout = QHBoxLayout()
        
        self.play_btn = QPushButton("播放")
        self.play_btn.clicked.connect(self.start_video)
        self.play_btn.setEnabled(False)
        
        self.stop_btn = QPushButton("停止 (空格)")
        self.stop_btn.clicked.connect(self.toggle_stop)
        self.stop_btn.setEnabled(False)
        
        self.slice_btn = QPushButton("切片")
        self.slice_btn.clicked.connect(self.slice_video)
        self.slice_btn.setEnabled(False)
        self.slice_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")

        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("上一个 (A)")
        self.prev_btn.clicked.connect(self.prev_video)
        self.prev_btn.setEnabled(False)
        self.file_counter = QLabel("")
        self.file_counter.setFixedWidth(100)
        self.file_counter.setAlignment(Qt.AlignCenter)
        self.next_btn = QPushButton("下一个 (D)")
        self.next_btn.clicked.connect(self.next_video)
        self.next_btn.setEnabled(False)
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.file_counter)
        nav_layout.addWidget(self.next_btn)

        control_layout.addWidget(self.play_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.slice_btn)
        control_layout.addLayout(nav_layout)
        control_layout.addStretch()

        speed_layout = QHBoxLayout()
        speed_label = QLabel("播放速度:")
        speed_label.setFixedWidth(70)
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.1x", "0.25x", "0.5x", "0.75x", "1x", "1.5x", "2x", "3x"])
        self.speed_combo.setCurrentText("1x")
        self.speed_combo.currentTextChanged.connect(self.change_speed)
        self.speed_combo.setFixedWidth(80)
        speed_layout.addWidget(speed_label)
        speed_layout.addWidget(self.speed_combo)
        speed_layout.addStretch()

        video_layout.addLayout(control_layout)
        video_layout.addLayout(speed_layout)

        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setFixedWidth(110)
        progress_layout.addWidget(self.time_label)

        video_layout.addLayout(progress_layout)

        slider_layout = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setValue(0)
        self.slider.sliderMoved.connect(self.seek_video)
        slider_layout.addWidget(self.slider)
        video_layout.addLayout(slider_layout)

        center_layout.addWidget(video_group, stretch=1)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_widget.setMinimumWidth(200)
        right_widget.setMaximumWidth(500)

        info_group = QGroupBox("操作日志")
        info_group.setFont(font)
        info_layout = QVBoxLayout(info_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        info_layout.addWidget(self.log_text)
        right_layout.addWidget(info_group)
        right_layout.addStretch()

        self.splitter.addWidget(left_widget)
        self.splitter.addWidget(center_widget)
        self.splitter.addWidget(right_widget)
        
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")

        self.add_log("视频切片工具已启动")
        self.add_log("支持格式: MP4, AVI, MOV, MKV, FLV等")

    def load_saved_paths(self):
        last_source = self.config.get("last_source_video", "")
        last_label = self.config.get("last_label_video", "")
        last_output = self.config.get("last_output_path", "")
        last_format = self.config.get("output_format", "png")

        if last_source:
            self.source_edit.setText(last_source)
            self.source_video_path = last_source

        if last_label:
            self.label_edit.setText(last_label)
            self.label_video_path = last_label

        if last_output:
            self.output_edit.setText(last_output)
            self.output_path = last_output

        index = self.format_combo.findText(last_format.upper())
        if index >= 0:
            self.format_combo.setCurrentIndex(index)

    def select_source_video(self, is_folder=False):
        if is_folder:
            dir_path = QFileDialog.getExistingDirectory(
                self, "选择原视频文件夹", 
                self.source_video_path if self.source_video_path else os.getcwd()
            )
            if dir_path:
                self.source_video_path = dir_path
                self.source_is_folder = True
                self.original_source_files = self.get_video_files(dir_path)
                self.source_files = self.original_source_files.copy()
                self.source_edit.setText(f"{dir_path} ({len(self.source_files)}个文件)")
                self.config.set("last_source_video", dir_path)
                self.add_log(f"原视频文件夹已选择: {dir_path}, 共 {len(self.source_files)} 个视频")
                self.check_ready()
        else:
            file_paths, _ = QFileDialog.getOpenFileNames(
                self, "选择原视频文件（支持多选）", 
                os.path.dirname(self.source_video_path) if self.source_video_path else "",
                "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv *.wmv);;所有文件 (*.*)"
            )
            if file_paths:
                if len(file_paths) == 1:
                    self.source_video_path = file_paths[0]
                    self.source_is_folder = False
                    self.source_files = file_paths
                    self.source_edit.setText(file_paths[0])
                    self.config.set("last_source_video", file_paths[0])
                    self.add_log(f"原视频文件已选择: {os.path.basename(file_paths[0])}")
                else:
                    self.source_video_path = os.path.dirname(file_paths[0])
                    self.source_is_folder = False
                    self.source_files = file_paths
                    self.source_edit.setText(f"{len(file_paths)} 个文件已选择")
                    self.add_log(f"原视频文件已选择: {len(file_paths)} 个文件")
                self.check_ready()

    def select_label_video(self, is_folder=False):
        if is_folder:
            dir_path = QFileDialog.getExistingDirectory(
                self, "选择标注视频文件夹", 
                self.label_video_path if self.label_video_path else os.getcwd()
            )
            if dir_path:
                self.label_video_path = dir_path
                self.label_is_folder = True
                self.original_label_files = self.get_video_files(dir_path)
                self.label_files = self.original_label_files.copy()
                self.label_edit.setText(f"{dir_path} ({len(self.label_files)}个文件)")
                self.config.set("last_label_video", dir_path)
                self.add_log(f"标注视频文件夹已选择: {dir_path}, 共 {len(self.label_files)} 个视频")
                self.check_ready()
        else:
            file_paths, _ = QFileDialog.getOpenFileNames(
                self, "选择标注视频文件（支持多选）", 
                os.path.dirname(self.label_video_path) if self.label_video_path else "",
                "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv *.wmv);;所有文件 (*.*)"
            )
            if file_paths:
                if len(file_paths) == 1:
                    self.label_video_path = file_paths[0]
                    self.label_is_folder = False
                    self.label_files = file_paths
                    self.label_edit.setText(file_paths[0])
                    self.config.set("last_label_video", file_paths[0])
                    self.add_log(f"标注视频文件已选择: {os.path.basename(file_paths[0])}")
                else:
                    self.label_video_path = os.path.dirname(file_paths[0])
                    self.label_is_folder = False
                    self.label_files = file_paths
                    self.label_edit.setText(f"{len(file_paths)} 个文件已选择")
                    self.add_log(f"标注视频文件已选择: {len(file_paths)} 个文件")
                self.check_ready()

    def get_video_files(self, folder_path):
        video_files = []
        all_files = os.listdir(folder_path)
        self.add_log(f"扫描文件夹: {folder_path}, 共 {len(all_files)} 个文件")
        
        for filename in all_files:
            lower_name = filename.lower()
            ext = os.path.splitext(lower_name)[1]
            if ext in self.VIDEO_EXTENSIONS:
                video_files.append(os.path.join(folder_path, filename))
        
        self.add_log(f"找到视频文件: {len(video_files)} 个")
        if video_files:
            self.add_log(f"示例文件名: {os.path.basename(video_files[0])}")
        return sorted(video_files)

    def toggle_sort_match(self, state):
        self.sort_match_enabled = state == Qt.Checked
        if self.source_is_folder and self.label_is_folder:
            self.match_videos()
        self.add_log(f"默认排序匹配: {'启用' if self.sort_match_enabled else '禁用'}")

    def match_videos(self):
        if not self.source_is_folder or not self.label_is_folder:
            return
        
        source_list = self.original_source_files.copy()
        label_list = self.original_label_files.copy()
        
        self.add_log(f"开始匹配: 源视频 {len(source_list)} 个, 标注视频 {len(label_list)} 个")
        
        if self.sort_match_enabled:
            min_count = min(len(source_list), len(label_list))
            self.source_files = sorted(source_list)[:min_count]
            self.label_files = sorted(label_list)[:min_count]
            self.add_log(f"按默认排序匹配: {len(self.source_files)} 对视频")
        else:
            label_name_map = {}
            for label_file in label_list:
                name_without_ext = os.path.splitext(os.path.basename(label_file))[0].lower()
                label_name_map[name_without_ext] = label_file
            
            matched_source = []
            matched_label = []
            
            for source_file in source_list:
                source_name = os.path.splitext(os.path.basename(source_file))[0].lower()
                if source_name in label_name_map:
                    matched_source.append(source_file)
                    matched_label.append(label_name_map[source_name])
            
            if len(matched_source) > 0:
                self.source_files = matched_source
                self.label_files = matched_label
                self.add_log(f"按文件名匹配: {len(self.source_files)} 对视频成功匹配")
            else:
                min_count = min(len(source_list), len(label_list))
                self.source_files = sorted(source_list)[:min_count]
                self.label_files = sorted(label_list)[:min_count]
                self.add_log(f"文件名匹配失败，自动切换为默认排序匹配: {len(self.source_files)} 对视频")
        
        self.current_index = 0
        self.update_file_counter()

    def find_similar_files(self, target_name, name_map):
        similar = []
        target_lower = target_name.lower()
        for name, path in name_map.items():
            name_lower = name.lower()
            if target_lower in name_lower or name_lower in target_lower:
                similar.append((name, path))
        return similar

    def show_similar_files_dialog(self, source_file, similar_files):
        source_basename = os.path.basename(source_file)
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Question)
        msg.setWindowTitle("文件匹配")
        msg.setText(f"未找到与 '{source_basename}' 同名的标注文件")
        
        if similar_files:
            similar_text = "\n".join([f"  - {name}" for name, _ in similar_files])
            msg.setInformativeText(f"找到以下相似文件，请选择：\n{similar_text}")
            
            buttons = []
            for i, (name, path) in enumerate(similar_files):
                btn = msg.addButton(name, QMessageBox.ActionRole)
                btn.setProperty("path", path)
                buttons.append(btn)
            
            cancel_btn = msg.addButton("跳过", QMessageBox.RejectRole)
            msg.exec_()
            
            clicked_btn = msg.clickedButton()
            if clicked_btn != cancel_btn:
                return clicked_btn.property("path")
        else:
            msg.setInformativeText("未找到相似文件，请检查文件命名")
            msg.exec_()
        
        return None

    def select_output_path(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择输出目录", 
            self.output_path if self.output_path else os.getcwd()
        )
        if dir_path:
            self.output_path = dir_path
            self.output_edit.setText(dir_path)
            self.config.set("last_output_path", dir_path)
            self.add_log(f"输出路径已设置: {dir_path}")
            self.check_ready()

    def check_ready(self):
        has_source = bool(self.source_video_path)
        has_label = bool(self.label_video_path)
        
        if has_source and has_label:
            self.play_btn.setEnabled(True)
            
            if self.source_is_folder and self.label_is_folder:
                self.match_videos()
            elif len(self.source_files) > 0 and len(self.label_files) > 0:
                min_count = min(len(self.source_files), len(self.label_files))
                if len(self.source_files) != len(self.label_files):
                    self.add_log(f"警告: 原视频 {len(self.source_files)} 个，标注视频 {len(self.label_files)} 个，将按 {min_count} 对匹配")
                self.current_index = 0
                self.update_file_counter()
            else:
                self.source_files = [self.source_video_path] if self.source_video_path else []
                self.label_files = [self.label_video_path] if self.label_video_path else []
                self.current_index = 0
                self.update_file_counter()
            
            self.add_log("可以开始播放视频")

    def update_file_counter(self):
        if len(self.source_files) > 1:
            self.file_counter.setText(f"{self.current_index + 1} / {len(self.source_files)}")
            self.prev_btn.setEnabled(self.current_index > 0)
            self.next_btn.setEnabled(self.current_index < len(self.source_files) - 1)
        else:
            self.file_counter.setText("")
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)

    def toggle_play(self):
        if not self.video_thread or not self.video_thread.isRunning():
            self.start_video()
        elif self.video_thread.paused:
            self.resume_video()
        else:
            self.pause_video()

    def start_video(self):
        if not self.label_files:
            QMessageBox.warning(self, "警告", "请先选择标注视频文件")
            return

        current_label = self.label_files[self.current_index]
        if not os.path.exists(current_label):
            QMessageBox.warning(self, "错误", f"标注视频文件不存在: {os.path.basename(current_label)}")
            return

        if self.video_thread:
            self.video_thread.stop()
            self.video_thread.wait()

        self.video_thread = VideoThread(current_label)
        self.video_thread.frame_signal.connect(self.update_frame)
        self.video_thread.time_signal.connect(self.update_time)
        self.video_thread.finished_signal.connect(self.video_finished)
        self.video_thread.start()

        self.play_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.slice_btn.setEnabled(False)
        self.status_bar.showMessage(f"正在播放: {os.path.basename(current_label)}")
        self.add_log(f"开始播放标注视频: {os.path.basename(current_label)}")

    def pause_video(self):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.pause()
            self.play_btn.setEnabled(True)
            self.slice_btn.setEnabled(True)
            self.status_bar.showMessage("已暂停")
            self.add_log("视频已暂停")

    def resume_video(self):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.resume()
            self.play_btn.setEnabled(False)
            self.slice_btn.setEnabled(False)
            self.status_bar.showMessage("正在播放...")
            self.add_log("继续播放视频")

    def toggle_stop(self):
        if self.video_thread and self.video_thread.isRunning():
            if self.video_thread.paused:
                self.resume_video()
            else:
                self.pause_video()

    def stop_video(self):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread.wait()
        
        self.play_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.slice_btn.setEnabled(False)
        self.video_label.clear()
        self.progress_bar.setValue(0)
        self.slider.setValue(0)
        self.time_label.setText("00:00 / 00:00")
        self.status_bar.showMessage("已停止")
        self.add_log("视频播放已停止")

    def prev_video(self):
        if len(self.source_files) > 0 and len(self.label_files) > 0 and self.current_index > 0:
            self.stop_video()
            self.current_index -= 1
            self.update_file_counter()
            if self.current_index < len(self.label_files):
                self.add_log(f"切换到上一个视频: {os.path.basename(self.label_files[self.current_index])}")
            else:
                self.add_log(f"切换到上一个视频: {os.path.basename(self.source_files[self.current_index])}")
            self.start_video()

    def next_video(self):
        if len(self.source_files) > 0 and len(self.label_files) > 0 and self.current_index < len(self.source_files) - 1 and self.current_index < len(self.label_files) - 1:
            self.stop_video()
            self.current_index += 1
            self.update_file_counter()
            if self.current_index < len(self.label_files):
                self.add_log(f"切换到下一个视频: {os.path.basename(self.label_files[self.current_index])}")
            else:
                self.add_log(f"切换到下一个视频: {os.path.basename(self.source_files[self.current_index])}")
            self.start_video()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_A:
            self.prev_video()
        elif event.key() == Qt.Key_D:
            self.next_video()
        elif event.key() == Qt.Key_Space:
            if self.video_thread and self.video_thread.isRunning():
                if self.video_thread.paused:
                    self.resume_video()
                else:
                    self.pause_video()
        elif event.key() == Qt.Key_Left:
            if self.video_thread and self.video_thread.isRunning():
                self.video_thread.seek_by_time(-0.5)
                self.add_log("回退 0.5 秒")
        elif event.key() == Qt.Key_Right:
            if self.video_thread and self.video_thread.isRunning():
                self.video_thread.seek_by_time(0.5)
                self.add_log("快进 0.5 秒")
        else:
            super().keyPressEvent(event)

    def seek_video(self, value):
        if self.video_thread and self.video_thread.cap:
            total_frames = int(self.video_thread.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_number = int((value / 1000) * total_frames)
            self.video_thread.set_position(frame_number)

    def change_speed(self, speed_text):
        speed = float(speed_text.replace('x', ''))
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.set_speed(speed)
        self.status_bar.showMessage(f"播放速度: {speed_text}")
        self.add_log(f"播放速度已调整为: {speed_text}")

    def update_frame(self, q_image):
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)

    def update_time(self, current_time):
        total_time = self.video_thread.get_total_time()
        progress = int((current_time / total_time) * 1000) if total_time > 0 else 0
        
        self.progress_bar.setValue(progress)
        self.slider.setValue(progress)
        
        current_str = self.format_time(current_time)
        total_str = self.format_time(total_time)
        self.time_label.setText(f"{current_str} / {total_str}")

    def video_finished(self):
        self.play_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.slice_btn.setEnabled(False)
        self.status_bar.showMessage("播放结束")
        self.add_log("视频播放结束")

    def slice_video(self):
        if not self.source_files:
            QMessageBox.warning(self, "警告", "请先选择原视频文件")
            return

        current_source = self.source_files[self.current_index]
        if not os.path.exists(current_source):
            QMessageBox.warning(self, "错误", f"原视频文件不存在: {os.path.basename(current_source)}")
            return

        if not self.output_path:
            default_path = os.path.join(os.getcwd(), "outcome")
            os.makedirs(default_path, exist_ok=True)
            self.output_path = default_path
            self.output_edit.setText(default_path)
            self.config.set("last_output_path", default_path)

        try:
            current_time = self.video_thread.get_current_time()
            
            cap = cv2.VideoCapture(current_source)
            if not cap.isOpened():
                QMessageBox.critical(self, "错误", "无法打开原视频文件")
                return

            frame_number = int(current_time * cap.get(cv2.CAP_PROP_FPS))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            ret, frame = cap.read()
            cap.release()

            if not ret:
                QMessageBox.warning(self, "警告", "无法从原视频中提取帧")
                return

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            format_lower = self.format_combo.currentText().lower()
            source_basename = os.path.splitext(os.path.basename(current_source))[0]
            filename = f"{source_basename}_{timestamp}.{format_lower}"
            save_path = os.path.join(self.output_path, filename)

            if format_lower == "jpg":
                cv2.imwrite(save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            elif format_lower == "png":
                cv2.imwrite(save_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            else:
                cv2.imwrite(save_path, frame)

            self.config.set("output_format", format_lower)
            self.status_bar.showMessage(f"图片已保存: {filename}")
            self.add_log(f"切片成功！时间点: {self.format_time(current_time)}, 保存路径: {save_path}")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"切片过程中发生错误: {str(e)}")
            self.add_log(f"切片失败: {str(e)}")

    def format_time(self, seconds):
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 100)
        return f"{minutes:02d}:{secs:02d}.{millis:02d}"

    def add_log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        scroll_bar = self.log_text.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())

    def closeEvent(self, event):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread.wait()
        event.accept()


def main():
    if not OPENCV_AVAILABLE:
        print("错误: 需要安装OpenCV库，请运行 pip install opencv-python")
        return

    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = VideoSlicer()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()