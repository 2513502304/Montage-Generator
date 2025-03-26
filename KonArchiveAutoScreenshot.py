"""
Author: 未来可欺 2513502304@qq.com
Date: 2025-03-22 01:10:54
LastEditors: 未来可欺 2513502304@qq.com
LastEditTime: 2025-03-26 19:57:33
Description: KonArchive 自动截屏脚本
"""

import pyautogui
import subprocess
import os

KonArchive = "https://archive.org/details/k-on-k-on-movie-illustration-archives-2009-2012"

# KonArchive 可执行文件路径
exec_file = r"D:/KonArchive/KonArchive.exe"
# 保存截屏的文件夹路径
screenshot_dir = r"./Data/K-ON!"
# 程序运行时调试文件夹路径
debug_dir = r"./Data/debug"


def auto_screenshot(exec_file: str, screenshot_dir: str, debug_dir: str) -> None:
    """
    KonArchive 自动截屏

    Args:
        exec_file (str): KonArchive 可执行文件路径
        screenshot_dir (str): 保存截屏的文件夹路径
        debug_dir (str): 程序运行时调试文件夹路径
    """
    # 若保存截屏的文件夹不存在，则自动创建
    if not os.path.exists(screenshot_dir):
        os.makedirs(screenshot_dir)
    # 若调试文件夹不存在，则自动创建
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    # 禁用故障保护
    pyautogui.FAILSAFE = False

    # 获取主监视器的大小
    screen_width, screen_height = pyautogui.size()
    # 主显示器中心点
    center = (screen_width // 2, screen_height // 2)

    # 启动运行程序（非阻塞）
    popen = subprocess.Popen(exec_file, shell=True)
    # 等待程序启动，并定位程序边界
    exe_box = pyautogui.locateOnScreen('./Data/src/KonArchive.png', minSearchTime=3, confidence=0.9)
    # 程序边界
    exe_region = tuple(map(int, (exe_box.left - 5, exe_box.top - 5, exe_box.width + 10, exe_box.height + 10)))
    pyautogui.screenshot('./Data/debug/程序边界.png', exe_region)  #! 减少图像匹配误差

    # 定位状态栏边界
    status_bar_box = pyautogui.locateOnScreen('./Data/src/status_bar.png', minSearchTime=3, region=exe_region, confidence=0.9)
    status_bar_region = tuple(map(int, (status_bar_box.left - 220, status_bar_box.top - 50, status_bar_box.width + 440, status_bar_box.height + 100)))
    # 移动到状态栏中心
    status_bar_pos = pyautogui.center(status_bar_box)
    pyautogui.moveTo(*status_bar_pos)
    # 打开状态栏
    pyautogui.drag(yOffset=100, button='left', duration=0.2)
    pyautogui.screenshot('./Data/debug/状态栏边界.png', status_bar_region)  #! 减少图像匹配误差

    # 图像库位置
    exhibit_pos = pyautogui.locateCenterOnScreen('./Data/src/exhibit.png', minSearchTime=3, region=status_bar_region, confidence=0.9)
    # 点击图像库
    pyautogui.click(*exhibit_pos)

    # 图像库中起始图像位置
    start_image_pos = pyautogui.locateCenterOnScreen('./Data/src/start_image.png', minSearchTime=3, region=exe_region, confidence=0.8)
    # 点击图像库中起始图像位置
    pyautogui.click(*start_image_pos)

    # 定位下一个选项位置
    right_pos = pyautogui.locateCenterOnScreen('./Data/src/right.png', minSearchTime=3, region=status_bar_region, confidence=0.9)

    # 遍历所有图像
    for i in range(1, 423 + 1, 1):
        # 第一次运行时，初始化各个选项位置
        if i == 1:
            # 定位高清选项位置
            HD_pos = pyautogui.locateCenterOnScreen('./Data/src/HD.png', minSearchTime=3, region=exe_region, confidence=0.9)
            # 点击高清选项位置
            pyautogui.click(*HD_pos)

            # 定位配置选项位置
            setting_pos = pyautogui.locateCenterOnScreen('./Data/src/setting.png', minSearchTime=3, region=exe_region, confidence=0.9)
            # 点击配置选项位置
            pyautogui.click(*setting_pos)

            # 定位适合屏幕选项位置
            fit_pos = pyautogui.locateCenterOnScreen('./Data/src/fit.png', minSearchTime=3, region=exe_region, confidence=0.9)
            # 点击适合屏幕选项位置
            pyautogui.click(*fit_pos)

            # 定位 ok 选项位置
            ok_pos = pyautogui.locateCenterOnScreen('./Data/src/ok.png', minSearchTime=3, region=exe_region, confidence=0.9)
            # 点击 ok 选项位置
            pyautogui.click(*ok_pos)
        else:
            # 点击下一个选项位置
            pyautogui.click(*right_pos)

            # 点击高清选项位置
            pyautogui.click(*HD_pos)

            # 点击配置选项位置
            pyautogui.click(*setting_pos)

            # 点击适合屏幕选项位置
            pyautogui.click(*fit_pos)

            # 点击 ok 选项位置
            pyautogui.click(*ok_pos)

        # 移动鼠标到屏幕右下角（屏幕外）
        pyautogui.moveTo(screen_width - 1, screen_height - 1)
        # 截屏当前图像
        path = os.path.join(screenshot_dir, f"{i:0>3d}.png")
        pyautogui.screenshot(path)
        # 点击任意屏幕内像素点以退出大屏
        pyautogui.click(*center)


if __name__ == '__main__':
    auto_screenshot(exec_file=exec_file, screenshot_dir=screenshot_dir, debug_dir=debug_dir)
