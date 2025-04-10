from skimage.color import deltaE_cie76, deltaE_ciede2000, deltaE_ciede94, deltaE_cmc
from typing import Sequence, MutableSequence
import numpy as np
import cv2 as cv
import os


def deltaE_manhattan(image1: Sequence[Sequence[int]], image2: Sequence[Sequence[int]], channel_axis: int = -1, weight: Sequence[int] = (1, 1, 1)) -> np.ndarray:
    """
    附带权重的两点之间的曼哈顿距离

    Args:
        image1 (Sequence[Sequence[int]]): image1
        image2 (Sequence[Sequence[int]]): image2
        channel_axis (int, optional): 参与运算的轴. Defaults to -1.
        weight (Sequence[int], optional): 图像通道权重. Defaults to (1, 1, 1).

    Returns:
        np.ndarray: 附带权重的两点之间的曼哈顿距离
        
    References:
        - https://wikimedia.org/api/rest_v1/media/math/render/svg/766971fac976a11f71166fb485df533072c886fb
    """
    c11, c12, c13 = np.moveaxis(image1.astype(np.float32), source=channel_axis, destination=0)[:3]
    c21, c22, c23 = np.moveaxis(image2.astype(np.float32), source=channel_axis, destination=0)[:3]
    c1_w, c2_w, c3_w = weight
    return c1_w * np.abs(c21 - c11) + c2_w * np.abs(c22 - c12) + c3_w * np.abs(c23 - c13)  # 最终结果与通道排列顺序无关


def deltaE_euclidean(image1: Sequence[Sequence[int]], image2: Sequence[Sequence[int]], channel_axis: int = -1, weight: Sequence[int] = (1, 1, 1)) -> np.ndarray:
    """
    附带权重的两点之间的欧几里得距离

    Args:
        image1 (Sequence[Sequence[int]]): image1
        image2 (Sequence[Sequence[int]]): image2
        channel_axis (int, optional): 参与运算的轴. Defaults to -1.
        weight (Sequence[int], optional): 图像通道权重. Defaults to (1, 1, 1).

    Returns:
        np.ndarray: 附带权重的两点之间的欧几里得距离
        
    References:
        - https://wikimedia.org/api/rest_v1/media/math/render/svg/766971fac976a11f71166fb485df533072c886fb
    """
    c11, c12, c13 = np.moveaxis(image1.astype(np.float32), source=channel_axis, destination=0)[:3]
    c21, c22, c23 = np.moveaxis(image2.astype(np.float32), source=channel_axis, destination=0)[:3]
    c1_w, c2_w, c3_w = weight
    return np.sqrt(c1_w * (c21 - c11)**2 + c2_w * (c22 - c12)**2 + c3_w * (c23 - c13)**2)  # 最终结果与通道排列顺序无关


def deltaE_approximation_bgr(image1: Sequence[Sequence[int]], image2: Sequence[Sequence[int]], channel_axis: int = -1) -> np.ndarray:
    """
    一种低成本近似方法
    这个公式的结果非常接近 L*u*v*（具有修改后的亮度曲线），更重要的是，它是一种更稳定的算法：它不存在一个颜色范围，在这个范围内会突然给出远离最优结果的结果

    Args:
        image1 (Sequence[Sequence[int]]): image1
        image2 (Sequence[Sequence[int]]): image2
        channel_axis (int, optional): 参与运算的轴. Defaults to -1.

    Returns:
        np.ndarray: RGB 色彩空间中两点之间的低成本近似距离
    
    Reference:
        - https://wikimedia.org/api/rest_v1/media/math/render/svg/95ee06baaa28944c5b1e06876439d1b579cf03c9
        - https://web.archive.org/web/20210327221240/https://www.compuphase.com/cmetric.htm
    """
    b1, g1, r1 = np.moveaxis(image1.astype(np.float32), source=channel_axis, destination=0)[:3]
    b2, g2, r2 = np.moveaxis(image2.astype(np.float32), source=channel_axis, destination=0)[:3]
    r_mean = (r1 + r2) / 2
    delat_r = r2 - r1
    delat_g = g2 - g1
    delta_b = b2 - b1
    return np.sqrt((2 + r_mean / 256) * delat_r**2 + 4 * delat_g**2 + (2 + (255 - r_mean) / 256) * delta_b**2)  # 最终结果与通道排列顺序有关


def deltaE_approximation_rgb(rgb1: Sequence[Sequence[int]], rgb2: Sequence[Sequence[int]], channel_axis: int = -1) -> np.ndarray:
    """
    一种低成本近似方法
    这个公式的结果非常接近 L*u*v*（具有修改后的亮度曲线），更重要的是，它是一种更稳定的算法：它不存在一个颜色范围，在这个范围内会突然给出远离最优结果的结果

    Args:
        rgb1 (Sequence[Sequence[int]]): rgb1
        rgb2 (Sequence[Sequence[int]]): rgb2
        channel_axis (int, optional): 参与运算的轴. Defaults to -1.

    Returns:
        np.ndarray: RGB 色彩空间中两点之间的低成本近似距离
    
    Reference:
        - https://wikimedia.org/api/rest_v1/media/math/render/svg/95ee06baaa28944c5b1e06876439d1b579cf03c9
        - https://web.archive.org/web/20210327221240/https://www.compuphase.com/cmetric.htm
    """
    r1, g1, b1 = np.moveaxis(rgb1.astype(np.float32), source=channel_axis, destination=0)[:3]
    r2, g2, b2 = np.moveaxis(rgb2.astype(np.float32), source=channel_axis, destination=0)[:3]
    r_mean = (r1 + r2) / 2
    delat_r = r2 - r1
    delat_g = g2 - g1
    delta_b = b2 - b1
    return np.sqrt((2 + r_mean / 256) * delat_r**2 + 4 * delat_g**2 + (2 + (255 - r_mean) / 256) * delta_b**2)  # 最终结果与通道排列顺序有关


def safe_imread(file: str) -> np.ndarray:
    """
    支持非 ASCII 编码路径的图像读取

    Args:
        file (str): 要读取的图像路径

    Returns:
        np.ndarray: 读取的 BGR 图像
    """
    # 以 uint8 数据类型，读取并解析二进制图像文件中的数据
    buffer = np.fromfile(file, dtype=np.uint8)
    # 解码数据为图像
    image = cv.imdecode(buffer, cv.IMREAD_COLOR)
    assert image is not None, f"Failed to load image: {file}"
    return image


def safe_imwrite(file: str, image: np.ndarray) -> True:
    """
    支持非 ASCII 编码路径的图像写入

    Args:
        file (str): 要写入的图像路径
        image (np.ndarray): 要写入的图像

    Returns:
        True: 是否写入成功
    """
    # 获取图像扩展名
    root, ext = os.path.splitext(file)
    # 将图像编码为指定文件格式的二进制数据
    ret, buffer = cv.imencode(ext, image)
    # 编码成功
    if ret:
        # 写入数据
        buffer.tofile(file)
        return True
    # 编码失败
    else:
        return False
