# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 14:17:34 2024

@author: yokumi
"""

import PyInstaller.__main__

PyInstaller.__main__.run([
    'spectrometer.py',  # 你的主脚本文件名
    '--onefile',  # 生成一个独立的可执行文件
    '--windowed',  # 不创建控制台窗口
])
