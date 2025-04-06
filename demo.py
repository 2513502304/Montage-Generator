from montage import Montage, deltaE_cie76, deltaE_ciede94, deltaE_ciede2000, deltaE_cmc, deltaE_manhattan, deltaE_euclidean, deltaE_approximation_bgr, deltaE_approximation_rgb, safe_imread, safe_imwrite
from typing import Callable, Literal
import numpy as np
import settings
import os


def color_metric_difference(
    image: np.ndarray,
    save_dir: str,
    mode: Literal['bgr', 'lab'] = 'bgr',
    func: Callable[..., np.ndarray] = deltaE_euclidean,
    func_kwargs: dict = {},
) -> None:
    """
    采用不同颜色差异计算方法对输入图像进行蒙太奇操作，并将其转换后的蒙太奇图像写入到文件名为函数名 + 函数参数的图像文件中
    
    Args:
        image (np.ndarray): 输入图像
        save_dir (str): 输出图像文件夹路径
        mode (Literal['bgr', 'lab'], optional): 衡量图像颜色差异的颜色空间，必须与 func 计算所使用的颜色空间对应，可选. Defaults to 'bgr'
        func (Callable, optional): 衡量颜色差异的可调用对象，该函数接受的第一个参数为参考颜色，第二个参数为比较颜色，函数签名请阅览 skimage.color 中提供的 deltaE_cie76, deltaE_ciede94, deltaE_ciede2000, deltaE_cmc 函数。可选. Defaults to deltaE_euclidean
        func_kwargs (dict, optional): 要传递给 func 的关键字参数，可选. Defaults to {}
    """
    montage: Montage = Montage(
        image_dir=settings.image_dir,
        block_height=settings.block_height,
        block_width=settings.block_width,
        lazy_load=settings.lazy_load,
        mode=mode,
        func=func,
        func_kwargs=func_kwargs,
    )
    montage_image = montage.generate_image(image=image, scale=settings.scale, fit_size=settings.fit_size)
    filename = " ".join([mode, func.__name__, " ".join([f'{key}={value}' for key, value in func_kwargs.items()])]) + ".jpg"
    safe_imwrite(os.path.join(save_dir, filename), montage_image)


def dominant_color_difference(
    reference: np.ndarray,
    comparison: np.ndarray,
) -> np.ndarray:
    """
    评估不同颜色差异计算方法的显示效果，对两张图像主色调采用欧几里得距离

    Args:
        reference (np.ndarray): 参考图像
        load_dir (str): 比较图像
    
    Returns:
        np.ndarray: 两张图像主色调的欧几里得距离
    """
    return deltaE_euclidean(Montage.calculate_dominant_color(reference), Montage.calculate_dominant_color(comparison))


if __name__ == '__main__':
    # 输入图像
    image = safe_imread(settings.image_input_path)

    # 创建输出图像文件夹路径
    metric_path = './Data/metric'
    if not os.path.exists(metric_path):
        os.makedirs(metric_path)

    # manhattan
    color_metric_difference(image, save_dir=metric_path, mode='bgr', func=deltaE_manhattan, func_kwargs={})

    # euclidean
    color_metric_difference(image, save_dir=metric_path, mode='bgr', func=deltaE_euclidean, func_kwargs={})

    # deltaE_approximation_rgb
    color_metric_difference(image, save_dir=metric_path, mode='bgr', func=deltaE_approximation_bgr, func_kwargs={})

    # deltaE_cie76
    color_metric_difference(image, save_dir=metric_path, mode='lab', func=deltaE_cie76, func_kwargs={})

    # deltaE_ciede94 with kL=1
    color_metric_difference(image, save_dir=metric_path, mode='lab', func=deltaE_ciede94, func_kwargs={'kL': 1})
    # deltaE_ciede94 with kL=2
    color_metric_difference(image, save_dir=metric_path, mode='lab', func=deltaE_ciede94, func_kwargs={'kL': 2})

    # deltaE_cie2000 with kL=1
    color_metric_difference(image, save_dir=metric_path, mode='lab', func=deltaE_ciede2000, func_kwargs={'kL': 1})
    # deltaE_cie2000 with kL=2
    color_metric_difference(image, save_dir=metric_path, mode='lab', func=deltaE_ciede2000, func_kwargs={'kL': 2})

    # deltaE_cmc with kL=1
    color_metric_difference(image, save_dir=metric_path, mode='lab', func=deltaE_cmc, func_kwargs={'kL': 1})
    # deltaE_cmc with kL=2
    color_metric_difference(image, save_dir=metric_path, mode='lab', func=deltaE_cmc, func_kwargs={'kL': 2})

    # 评估不同颜色差异计算方法的显示效果
    image_paths = [os.path.join(metric_path, name) for name in os.listdir(metric_path)]
    for image_file in image_paths:
        reference = image
        comparison = safe_imread(image_file)
        print(f'{image_file}: {dominant_color_difference(reference=reference, comparison=comparison)}')
