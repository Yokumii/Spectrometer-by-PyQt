**Digital Spectrometer by PyQt**

本项目是一个基于 Python 的数字光谱仪软件，能够通过摄像头实时采集光谱图像，并提供图像处理、光谱校准和参数调节功能。用户可以直观地查看光谱数据，并根据需要进行校准和数据分析。



**功能特点**

* 实时显示光谱数据，包括波长与光强分布。
* 支持光谱校准功能，用户可根据已知波长和像素点校准波长。
* 可调整光谱数据的平滑、波峰检测参数。
* 提供基线图像功能，支持设置与移除。
* 具有吸收光谱样式的渐变填充显示。
* 可将项目打包为独立的 .exe 文件，便于分发和运行。



**环境要求**

* **Python** 3.8 或以上版本
* **操作系统**：Windows、Linux 或 macOS（推荐 Windows）



**安装依赖**

使用以下命令安装项目依赖：

```bash
pip install -r requirements.txt
```



**使用说明**

1. **启动项目**

运行以下命令启动程序：

```bash
python spectrometer.py
```



2. **快捷键说明**

* **Esc**：打开设置面板。
* **B**：设置基线图像。
* **P**：暂停或恢复实时绘图。
* **E**：截图当前界面并保存为 screenshot.png。



3. **打包为 .exe 文件**

项目中包含 installer.py 脚本，可用于将 Python 项目打包为独立的 .exe 文件。



**打包步骤**

1. 确保已安装 pyinstaller：

```bash
pip install pyinstaller
```

2. 运行 installer.py 脚本：

```bash
python installer.py
```

3. 打包完成后，生成的 .exe 文件将位于 dist/ 目录下。



如有问题或建议，请提交Issue!



**授权协议**

本项目遵循 [MIT License](LICENSE) 协议，欢迎自由使用、修改和分发。