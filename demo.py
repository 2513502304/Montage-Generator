from montage import Montage, deltaE_cie76, deltaE_ciede94, deltaE_ciede2000, deltaE_cmc, deltaE_manhattan, deltaE_euclidean, deltaE_approximation_bgr, deltaE_approximation_rgb, safe_imread, safe_imwrite
from typing import Callable, Literal
import numpy as np
import settings
import os


def show_color_metric_difference(
    image: np.ndarray,
    save_dir: str,
    mode: Literal['bgr', 'lab'] = 'bgr',
    func: Callable[..., np.ndarray] = deltaE_euclidean,
    func_kwargs: dict = {},
) -> None:
    """
    评估不同颜色差异计算方法的显示效果，并将其写入到文件名为函数名 + 函数参数的图像文件中
    
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


if __name__ == '__main__':
    # 输入图像
    image = safe_imread(settings.image_input_path)

    # 创建输出图像文件夹路径
    metric_path = './Data/metric'
    if not os.path.exists(metric_path):
        os.makedirs(metric_path)

    # manhattan
    show_color_metric_difference(image, save_dir=metric_path, mode='bgr', func=deltaE_manhattan, func_kwargs={})

    # euclidean
    show_color_metric_difference(image, save_dir=metric_path, mode='bgr', func=deltaE_euclidean, func_kwargs={})

    # deltaE_approximation_rgb
    show_color_metric_difference(image, save_dir=metric_path, mode='bgr', func=deltaE_approximation_bgr, func_kwargs={})

    # deltaE_cie76
    show_color_metric_difference(image, save_dir=metric_path, mode='lab', func=deltaE_cie76, func_kwargs={})

    # deltaE_ciede94 with kL=1
    show_color_metric_difference(image, save_dir=metric_path, mode='lab', func=deltaE_ciede94, func_kwargs={'kL': 1})
    # deltaE_ciede94 with kL=2
    show_color_metric_difference(image, save_dir=metric_path, mode='lab', func=deltaE_ciede94, func_kwargs={'kL': 2})

    # deltaE_cie2000 with kL=1
    show_color_metric_difference(image, save_dir=metric_path, mode='lab', func=deltaE_ciede2000, func_kwargs={'kL': 1})
    # deltaE_cie2000 with kL=2
    show_color_metric_difference(image, save_dir=metric_path, mode='lab', func=deltaE_ciede2000, func_kwargs={'kL': 2})

    # deltaE_cmc with kL=1
    show_color_metric_difference(image, save_dir=metric_path, mode='lab', func=deltaE_cmc, func_kwargs={'kL': 1})
    # deltaE_cmc with kL=2
    show_color_metric_difference(image, save_dir=metric_path, mode='lab', func=deltaE_cmc, func_kwargs={'kL': 2})
