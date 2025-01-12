import sys
import cv2
import numpy as np
import pyqtgraph as pg
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from PyQt5 import QtWidgets, QtCore, QtGui
import csv

class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, current_params=None, calibration_data=None, grayscale_weights=None, baseline_image_exists=False, alpha=0.05):
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.setGeometry(300, 300, 400, 400)

        self.current_params = current_params or {'height': 500, 'prominence': 100, 'distance': 50, 'width': 20}
        self.calibration_data = calibration_data or []  # 保留已存在的校准数据
        self.grayscale_weights = grayscale_weights or {'R': 0.299, 'G': 0.587, 'B': 0.114}  # 灰度转换默认权值
        self.baseline_image_exists = baseline_image_exists  # 检查是否有基线图像
        self.alpha = alpha  # 添加平滑因子初始值

        # 创建选项卡
        self.tabs = QtWidgets.QTabWidget()
        self.calibration_tab = QtWidgets.QWidget()
        self.threshold_tab = QtWidgets.QWidget()
        self.grayscale_tab = QtWidgets.QWidget()
        self.baseline_tab = QtWidgets.QWidget()

        # 校准设置页面
        self.create_calibration_tab()

        # 阈值调整页面
        self.create_threshold_tab()

        # 灰度权重设置页面
        self.create_grayscale_tab()

        # 基线图像设置页面
        self.create_baseline_tab()

        # 添加选项卡
        self.tabs.addTab(self.calibration_tab, "校准设置")
        self.tabs.addTab(self.threshold_tab, "参数设置")
        self.tabs.addTab(self.grayscale_tab, "灰度权值设置")
        self.tabs.addTab(self.baseline_tab, "基线设置")

        # 确定按钮
        self.confirm_button = QtWidgets.QPushButton("确定")
        self.confirm_button.clicked.connect(self.accept)

        # 布局设置
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.tabs)
        layout.addWidget(self.confirm_button)
        self.setLayout(layout)

    def create_calibration_tab(self):
        """创建校准设置页面"""
        layout = QtWidgets.QVBoxLayout()

        # 波长和像素输入框
        self.wavelength_label = QtWidgets.QLabel("已知波长 (nm):")
        self.wavelength_input = QtWidgets.QLineEdit()
        layout.addWidget(self.wavelength_label)
        layout.addWidget(self.wavelength_input)

        self.pixel_label = QtWidgets.QLabel("像素位置:")
        self.pixel_input = QtWidgets.QLineEdit()
        layout.addWidget(self.pixel_label)
        layout.addWidget(self.pixel_input)

        # 添加数据按钮
        self.add_data_button = QtWidgets.QPushButton("添加数据")
        self.add_data_button.clicked.connect(self.add_calibration_data)
        layout.addWidget(self.add_data_button)

        # 显示添加的数据（带有删除按钮）
        self.data_list_widget = QtWidgets.QListWidget()
        layout.addWidget(self.data_list_widget)

        # 将之前的校准数据展示出来
        self.load_existing_calibration_data()

        # 将布局添加到选项卡
        self.calibration_tab.setLayout(layout)

    def create_threshold_tab(self):
        """创建阈值调整页面"""
        layout = QtWidgets.QVBoxLayout()

        # height设置
        self.height_label = QtWidgets.QLabel("光强过滤阈值")
        self.height_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.height_slider.setRange(0, 1000)
        self.height_slider.setValue(self.current_params['height'])
        self.height_value = QtWidgets.QLineEdit(str(self.current_params['height']))
        self.height_value.setFixedWidth(80)
        layout.addWidget(self.height_label)
        layout.addWidget(self.height_slider)
        layout.addWidget(self.height_value)

        # prominence设置
        self.prominence_label = QtWidgets.QLabel("波峰明显程度阈值")
        self.prominence_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.prominence_slider.setRange(0, 500)
        self.prominence_slider.setValue(self.current_params['prominence'])
        self.prominence_value = QtWidgets.QLineEdit(str(self.current_params['prominence']))
        self.prominence_value.setFixedWidth(80)
        layout.addWidget(self.prominence_label)
        layout.addWidget(self.prominence_slider)
        layout.addWidget(self.prominence_value)

        # distance设置
        self.distance_label = QtWidgets.QLabel("波峰距离阈值")
        self.distance_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.distance_slider.setRange(10, 500)
        self.distance_slider.setValue(self.current_params['distance'])
        self.distance_value = QtWidgets.QLineEdit(str(self.current_params['distance']))
        self.distance_value.setFixedWidth(80)
        layout.addWidget(self.distance_label)
        layout.addWidget(self.distance_slider)
        layout.addWidget(self.distance_value)

        # width设置
        self.width_label = QtWidgets.QLabel("波峰宽度阈值")
        self.width_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.width_slider.setRange(0, 100)
        self.width_slider.setValue(self.current_params['width'])
        self.width_value = QtWidgets.QLineEdit(str(self.current_params['width']))
        self.width_value.setFixedWidth(80)
        layout.addWidget(self.width_label)
        layout.addWidget(self.width_slider)
        layout.addWidget(self.width_value)

        # 平滑因子设置
        self.alpha_label = QtWidgets.QLabel("平滑因子")
        self.alpha_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.alpha_slider.setRange(1, 100)  # 将平滑因子范围设为0.01到1.00，滑块值为1到100
        self.alpha_slider.setValue(int(self.alpha * 100))  # 初始值为0.05
        self.alpha_value = QtWidgets.QLineEdit(str(self.alpha))
        self.alpha_value.setFixedWidth(80)
        layout.addWidget(self.alpha_label)
        layout.addWidget(self.alpha_slider)
        layout.addWidget(self.alpha_value)

        # 滑块与输入框联动
        self.height_slider.valueChanged.connect(self.update_height_value)
        self.prominence_slider.valueChanged.connect(self.update_prominence_value)
        self.distance_slider.valueChanged.connect(self.update_distance_value)
        self.width_slider.valueChanged.connect(self.update_width_value)
        self.alpha_slider.valueChanged.connect(self.update_alpha_value)

        # 将布局添加到选项卡
        self.threshold_tab.setLayout(layout)

    def create_grayscale_tab(self):
        """创建灰度权重设置页面"""
        layout = QtWidgets.QVBoxLayout()

        # R通道权重设置
        self.r_label = QtWidgets.QLabel("红色通道权重 (R):")
        self.r_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.r_slider.setRange(0, 100)
        self.r_slider.setValue(int(self.grayscale_weights['R'] * 100))
        self.r_value = QtWidgets.QLineEdit(str(self.grayscale_weights['R']))
        self.r_value.setFixedWidth(80)
        layout.addWidget(self.r_label)
        layout.addWidget(self.r_slider)
        layout.addWidget(self.r_value)

        # G通道权重设置
        self.g_label = QtWidgets.QLabel("绿色通道权重 (G):")
        self.g_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.g_slider.setRange(0, 100)
        self.g_slider.setValue(int(self.grayscale_weights['G'] * 100))
        self.g_value = QtWidgets.QLineEdit(str(self.grayscale_weights['G']))
        self.g_value.setFixedWidth(80)
        layout.addWidget(self.g_label)
        layout.addWidget(self.g_slider)
        layout.addWidget(self.g_value)

        # B通道权重设置
        self.b_label = QtWidgets.QLabel("蓝色通道权重 (B):")
        self.b_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.b_slider.setRange(0, 100)
        self.b_slider.setValue(int(self.grayscale_weights['B'] * 100))
        self.b_value = QtWidgets.QLineEdit(str(self.grayscale_weights['B']))
        self.b_value.setFixedWidth(80)
        layout.addWidget(self.b_label)
        layout.addWidget(self.b_slider)
        layout.addWidget(self.b_value)

        # 将滑块与文本框联动
        self.r_slider.valueChanged.connect(self.update_r_value)
        self.g_slider.valueChanged.connect(self.update_g_value)
        self.b_slider.valueChanged.connect(self.update_b_value)

        self.grayscale_tab.setLayout(layout)

    def create_baseline_tab(self):
        """创建基线图像设置页面"""
        layout = QtWidgets.QVBoxLayout()

        # 删除基线图像按钮
        self.delete_baseline_button = QtWidgets.QPushButton("删除基线图像")
        self.delete_baseline_button.clicked.connect(self.delete_baseline_image)
        layout.addWidget(self.delete_baseline_button)

        # 如果没有基线图像，则禁用删除按钮
        if not self.baseline_image_exists:
            self.delete_baseline_button.setEnabled(False)

        # 将布局添加到选项卡
        self.baseline_tab.setLayout(layout)



    def update_height_value(self):
        self.height_value.setText(str(self.height_slider.value()))

    def update_prominence_value(self):
        self.prominence_value.setText(str(self.prominence_slider.value()))

    def update_distance_value(self):
        self.distance_value.setText(str(self.distance_slider.value()))

    def update_width_value(self):
        self.width_value.setText(str(self.width_slider.value()))

    def update_r_value(self):
        self.r_value.setText(str(self.r_slider.value() / 100))

    def update_g_value(self):
        self.g_value.setText(str(self.g_slider.value() / 100))

    def update_b_value(self):
        self.b_value.setText(str(self.b_slider.value() / 100))

    def update_alpha_value(self):
        self.alpha_value.setText(str(self.alpha_slider.value() / 100))

    def get_settings(self):
        """返回用户设置的参数、校准数据和灰度权值"""
        return {
            'height': int(self.height_value.text()),
            'prominence': int(self.prominence_value.text()),
            'distance': int(self.distance_value.text()),
            'width': int(self.width_value.text()),
            'calibration_data': self.calibration_data,
            'grayscale_weights': {
                'R': float(self.r_value.text()),
                'G': float(self.g_value.text()),
                'B': float(self.b_value.text())
            },
            'alpha': float(self.alpha_value.text())  # 返回平滑因子
        }

    def add_calibration_data(self):
        """添加校准数据"""
        try:
            wavelength = float(self.wavelength_input.text())
            pixel = float(self.pixel_input.text())
            self.calibration_data.append((wavelength, pixel))

            # 更新显示
            self.add_calibration_item(wavelength, pixel)

            self.wavelength_input.clear()
            self.pixel_input.clear()
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "输入错误", "请输入有效的数值！")

    def add_calibration_item(self, wavelength, pixel):
        """在校准列表中显示添加的项，并添加删除按钮"""
        item_widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout()

        # 显示校准数据
        data_label = QtWidgets.QLabel(f"波长: {wavelength:.1f} nm, 像素位置: {pixel:.1f}")
        layout.addWidget(data_label)

        # 删除按钮
        delete_button = QtWidgets.QPushButton("删除")
        delete_button.clicked.connect(lambda: self.remove_calibration_data(item_widget, wavelength, pixel))
        layout.addWidget(delete_button)

        item_widget.setLayout(layout)

        # 添加到列表
        list_item = QtWidgets.QListWidgetItem(self.data_list_widget)
        list_item.setSizeHint(item_widget.sizeHint())
        self.data_list_widget.addItem(list_item)
        self.data_list_widget.setItemWidget(list_item, item_widget)

    def load_existing_calibration_data(self):
        """加载已有的校准数据并显示"""
        for wavelength, pixel in self.calibration_data:
            self.add_calibration_item(wavelength, pixel)

    def remove_calibration_data(self, item_widget, wavelength, pixel):
        """从校准数据中移除一项"""
        index = self.calibration_data.index((wavelength, pixel))
        if index != -1:
            self.calibration_data.pop(index)

        # 从列表中移除显示项
        row = self.data_list_widget.indexAt(item_widget.pos()).row()
        self.data_list_widget.takeItem(row)

    def delete_baseline_image(self):
        """删除基线图像"""
        if self.parent():
            self.parent().delete_baseline_image()
            self.delete_baseline_button.setEnabled(False)  # 删除后禁用按钮

class Spectrometer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # 设置全局样式，背景白色，曲线颜色等
        pg.setConfigOption('background', 'w')  # 白色背景
        pg.setConfigOption('foreground', 'k')  # 黑色线条、文字等前景

        # 创建窗口和图形视图
        self.setWindowTitle('数字光谱仪By邮宛大理包')
        self.setGeometry(100, 100, 1920, 1080)
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QtWidgets.QVBoxLayout(self.central_widget)
        self.graphics_view = pg.GraphicsView()
        self.layout.addWidget(self.graphics_view)

        self.plot_widget = pg.PlotItem(title="光谱图")
        self.plot_widget.setLabel('left', '相对光强 (Intensity)', units='AU')  # 添加纵轴标签
        self.plot_widget.setLabel('bottom', '波长 (Wavelength)', units='nm')  # 添加横轴标签
        self.graphics_view.setCentralItem(self.plot_widget)

        self.image_view = pg.ImageView()
        self.image_view.ui.histogram.hide()  # 隐藏直方图和亮度滑动条
        self.image_view.ui.roiBtn.hide()  # 隐藏 ROI 按钮
        self.image_view.ui.menuBtn.hide()  # 隐藏 Menu 按钮
        self.layout.addWidget(self.image_view)

        self.cap = cv2.VideoCapture(0)
        # 获取摄像头实际的分辨率
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"摄像头分辨率: {self.width}x{self.height}")

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(100)  # 每100毫秒更新一次

        # 设置曲线为黑色
        self.plot_data_item = self.plot_widget.plot(pen=pg.mkPen('k', width=2))  # 设置为黑色曲线

        self.wavelengths = np.linspace(380, 780, self.width)  # 假设摄像头宽度对应380nm到780nm可见光范围
        self.intensity = np.zeros(self.width)

        self.paused = False
        self.setstart = False
        self.baseline_image = None

        # 初始化颜色梯度映射
        self.cmap = self.create_color_map()
        self.smooth_intensity = np.zeros(self.width)  # 初始化平滑后的强度值

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Camera FPS: {self.fps}")
        self.accumulated_intensity = np.zeros(self.width)  # 用于累积处理后的光强
        self.accumulated_images = np.zeros((self.height, self.width))  # 存储累积的100帧图像的灰度值
        self.frame_count = 0  # 帧计数器，用于记录已处理的帧数

        self.peak_lines = []  # 用于保存绘制的红线对象
        self.peak_labels = []  # 用于保存绘制的坐标标签对象
        self.fill_between_items = []  # 用于保存填充的区域对象
        # 默认的find_peaks参数
        self.peak_params = {'height': 0, 'prominence': 0, 'distance': 50, 'width': 20}
        # 初始化默认灰度权值
        self.grayscale_weights = {'R': 0.299, 'G': 0.587, 'B': 0.114}

        # 校准数据
        self.calibration_data = []  # 用于存储波长和像素点的配对数据
        self.calibrated_wavelengths = np.linspace(380, 780, self.width)  # 校准后的波长
        self.is_calibrated = False  # 标志是否已校准
        
        self.export_requested = False  # 用于判断是否请求导出

    def create_color_map(self):
        """
        创建颜色映射：将可见光范围（380nm到780nm）映射到RGB颜色。
        """
        cmap = pg.ColorMap(
            pos=np.linspace(0, 1, 6),
            color=[
                (148, 0, 211),  # 紫色
                (75, 0, 130),   # 靛色
                (0, 0, 255),    # 蓝色
                (0, 255, 0),    # 绿色
                (255, 255, 0),  # 黄色
                (255, 0, 0),    # 红色
            ]
        )
        return cmap

    def map_wavelength_to_color(self):
        """
        根据波长映射到RGB颜色。
        """
        norm_wavelengths = (self.wavelengths - 380) / (780 - 380)  # 将波长归一化到[0, 1]
        colors = self.cmap.map(norm_wavelengths, mode='qcolor')
        return colors


    def fill_color_under_curve(self, wavelengths, intensity):
        """
        根据波长填充光谱曲线下方区域并添加颜色。
        """
        colors = self.map_wavelength_to_color()  # 获取对应波长的颜色

        for i in range(len(wavelengths) - 1):
            x_vals = wavelengths[i:i + 2]
            y_vals = intensity[i:i + 2]

            # 创建渐变填充区域
            fill_brush = pg.mkBrush(colors[i])  # 对应波长的颜色
            fill_between = pg.FillBetweenItem(
                self.plot_widget.plot(x_vals, y_vals, pen=None),  # 上限曲线
                self.plot_widget.plot(x_vals, [0, 0], pen=None),  # 下限曲线为0
                brush=fill_brush
            )
            self.plot_widget.addItem(fill_between)

    def clear_previous_fills(self):
        """清除之前绘制的填充区域，避免内存泄漏和卡顿"""
        for item in self.fill_between_items:
            self.plot_widget.removeItem(item)
        self.fill_between_items.clear()

    def set_baseline_image(self):
        ret, frame = self.cap.read()
        if ret:
            flipped_frame = cv2.flip(frame, 1)
            self.baseline_image = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2GRAY)
            print("Baseline image set.")
            QtWidgets.QMessageBox.information(self, "提示", "基线图像已添加！")

    def delete_baseline_image(self):
        """删除基线图像"""
        if self.baseline_image is not None:
            self.baseline_image = None
            print("Baseline image deleted.")
            QtWidgets.QMessageBox.information(self, "提示", "基线图像已删除！")
        else:
            QtWidgets.QMessageBox.warning(self, "警告", "尚未添加基线图像！")

    # 按键设置
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.show_settings()
        elif event.key() == QtCore.Qt.Key_E:
            self.take_screenshot()
        elif event.key() == QtCore.Qt.Key_P:
            self.paused = not self.paused
        elif event.key() == QtCore.Qt.Key_B:
            self.set_baseline_image()
        elif event.key() == QtCore.Qt.Key_S:
            if not self.is_calibrated:
                print("请先进行校准！")
                QtWidgets.QMessageBox.warning(self, "错误", "请先进行校准！")
            elif self.frame_count < 100:
                print("等待累计100帧后保存数据...")
                self.export_requested = True
            else:
                self.save_data_to_csv()

    def show_settings(self):
        dialog = SettingsDialog(self, current_params=self.peak_params, calibration_data=self.calibration_data,
                                grayscale_weights=self.grayscale_weights,
                                baseline_image_exists=self.baseline_image is not None,
                                alpha=self.peak_params.get('alpha', 0.05))
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            settings = dialog.get_settings()
            self.peak_params.update({
                'height': settings['height'],
                'prominence': settings['prominence'],
                'distance': settings['distance'],
                'width': settings['width'],
                'alpha': settings['alpha']  # 更新平滑因子
            })
            self.calibration_data = settings['calibration_data']
            self.grayscale_weights = settings['grayscale_weights']
            self.perform_calibration()

    def convert_to_grayscale(self, frame):
        """使用自定义权值计算灰度图像"""
        r_weight = self.grayscale_weights['R']
        g_weight = self.grayscale_weights['G']
        b_weight = self.grayscale_weights['B']

        # Split the channels
        b_channel, g_channel, r_channel = cv2.split(frame)

        # Apply the custom grayscale conversion formula
        gray_frame = cv2.addWeighted(r_channel, r_weight, g_channel, g_weight, 0)
        gray_frame = cv2.addWeighted(gray_frame, 1.0, b_channel, b_weight, 0)

        return gray_frame

    def perform_calibration(self):
        if len(self.calibration_data) >= 2:
            wavelengths, pixels = zip(*self.calibration_data)
            coeffs = np.polyfit(pixels, wavelengths, 1)
            self.calibrated_wavelengths = np.polyval(coeffs, np.arange(self.width))
            self.is_calibrated = True
            print(f"Calibration Coefficients: {coeffs}")
        else:
            self.is_calibrated = False

    def update(self):
        if self.paused:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        # 将图像进行水平翻转
        flipped_frame = cv2.flip(frame, 1)

        # 将图像转换为RGB图进行显示
        rgb_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)

        # 使用自定义权重计算灰度图像
        gray_frame = self.convert_to_grayscale(flipped_frame)
        # 如果基线图像已设置，进行减法操作
        if self.baseline_image is not None:
            rgb_frame = cv2.absdiff(rgb_frame, cv2.cvtColor(self.baseline_image, cv2.COLOR_BGR2RGB))
            gray_frame = cv2.absdiff(gray_frame, self.baseline_image)

        # 显示彩色图像
        self.image_view.setImage(np.rot90(rgb_frame, 3))

        # 累积当前的灰度图像数据
        self.accumulated_images += gray_frame
        self.frame_count += 1

        # 每积累 100 帧时计算平均值并更新
        if self.frame_count >= 2 * self.fps:
            # 计算100帧的平均图像
            average_image = self.accumulated_images / (self.fps * 2)

            # 计算每列像素的灰度均值
            current_intensity = np.mean(average_image, axis=0)

            # 对当前强度数据进行高斯平滑处理以减少噪点
            current_intensity = gaussian_filter1d(current_intensity, sigma=2)

            alpha = self.peak_params.get('alpha', 0.05)  # 从设置中获取平滑因子
            self.smooth_intensity = alpha * current_intensity + (1 - alpha) * self.smooth_intensity

            # 判断是否已校准，选择使用的波长
            if self.is_calibrated:
                wavelengths_to_use = self.calibrated_wavelengths
            else:
                wavelengths_to_use = self.wavelengths  # 使用默认的波长范围

            # 更新绘图数据时使用平滑后的强度值和适当的波长
            self.plot_data_item.setData(wavelengths_to_use, self.smooth_intensity)
            
            # 如果用户按下S键并且满足累计100帧的条件，保存数据
            if self.export_requested:
                self.save_data_to_csv()
                self.export_requested = False  # 重置请求标志
            # # 清除之前的填充区域
            # self.clear_previous_fills()
            #
            # # 填充曲线下方区域并根据波长映射颜色
            # self.fill_color_under_curve(wavelengths_to_use, self.smooth_intensity)

            # 重置累积器和帧计数器
            self.accumulated_images = np.zeros((self.height, self.width))
            self.frame_count = 0

        # 查找平滑数据中的极大值位置，使用自定义参数
        peaks, _ = find_peaks(self.smooth_intensity, height=self.peak_params['height'],
                              prominence=self.peak_params['prominence'], distance=self.peak_params['distance'], width=self.peak_params['width'])

        # 删除之前绘制的红线和标签
        for line in self.peak_lines:
            self.plot_widget.removeItem(line)
        for label in self.peak_labels:
            self.plot_widget.removeItem(label)

        # 绘制红色竖线标记极大值位置并标出坐标
        self.peak_lines = []
        self.peak_labels = []
        for peak in peaks:
            # 使用校准波长来绘制红线
            if self.is_calibrated:
                wavelength_at_peak = self.calibrated_wavelengths[peak]
            else:
                wavelength_at_peak = self.wavelengths[peak]

            # 绘制红线
            line = pg.PlotCurveItem(x=[wavelength_at_peak, wavelength_at_peak],
                                    y=[0, self.smooth_intensity[peak]],
                                    pen=pg.mkPen(color='r', width=2))
            self.plot_widget.addItem(line)
            self.peak_lines.append(line)

            # 标出峰值点的坐标
            label = pg.TextItem(f"({wavelength_at_peak:.1f}, {self.smooth_intensity[peak]:.1f})",
                                anchor=(0.5, 1.5), color='r')
            label.setPos(wavelength_at_peak, self.smooth_intensity[peak])
            self.plot_widget.addItem(label)
            self.peak_labels.append(label)

    def delete_calibration_points(self):
        self.calibration_data = []
        self.calibrated_wavelengths = np.linspace(380, 780, self.width)  # 重置为默认的波长范围
        self.is_calibrated = False  # 重置校准标志
        print("Calibration points deleted.")

    def set_start(self):
        self.setstart = True
        self.accumulated_intensity = None
        print("Start accumulating.")

    # 截图操作
    def take_screenshot(self):
        screenshot = self.grab()
        screenshot.save('screenshot.png')
        print('Screenshot saved as screenshot.png')

    def closeEvent(self, event):
        self.cap.release()
        event.accept()
    
    def save_data_to_csv(self):
        """将校准后的波长和相对光强保存到CSV文件"""
        if not self.is_calibrated:
            QtWidgets.QMessageBox.warning(self, "错误", "未进行校准，无法保存数据！")
            return

        # 打开文件保存对话框，让用户选择保存文件路径和文件名
        options = QtWidgets.QFileDialog.Options()
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "保存光谱数据", "", "CSV Files (*.csv)", options=options)
    
        if file_path:  # 如果用户没有取消选择文件路径
            with open(file_path, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(['Wavelength (nm)', 'Relative Intensity'])
                for wavelength, intensity in zip(self.calibrated_wavelengths, self.smooth_intensity):
                    csvwriter.writerow([wavelength, intensity])
            print(f"数据已保存到 {file_path}")

# 添加启动界面
def show_splash_screen(app):
    # 创建启动界面
    splash_pix = QtGui.QPixmap(600, 300)  # 启动界面大小
    splash_pix.fill(QtCore.Qt.white)  # 背景颜色设置为白色

    # 在启动界面上绘制文本
    painter = QtGui.QPainter(splash_pix)
    painter.setFont(QtGui.QFont("Arial", 16))

    # 绘制软件介绍
    painter.drawText(splash_pix.rect(), QtCore.Qt.AlignCenter, "数字光谱仪\n\n"
                        "版本: 2.0\n"
                        "作者: 邮宛大理包\n"
                        "按键说明:\n"
                        "Esc: 打开设置\n"
                        "B: 添加基线图像\n"
                        "P: 暂停\n"
                        "E: 截图\n"
                        "程序启动中，请耐心等待...\n")
    painter.end()

    # 显示启动界面
    splash = QtWidgets.QSplashScreen(splash_pix, QtCore.Qt.WindowStaysOnTopHint)
    splash.setMask(splash_pix.mask())

    # 显示启动界面
    splash.show()

    # 模拟加载过程
    QtWidgets.qApp.processEvents()
    QtCore.QThread.sleep(5)  # 暂停5秒以显示启动界面

    return splash

def main():
    app = QtWidgets.QApplication(sys.argv)
    spectrometer = Spectrometer()
    show_splash_screen(app)
    spectrometer.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()