from utils import deltaE_cie76, deltaE_ciede94, deltaE_ciede2000, deltaE_cmc, deltaE_manhattan, deltaE_euclidean, deltaE_approximation_bgr, deltaE_approximation_rgb, safe_imread, safe_imwrite
from rich.progress import Progress, track
from typing import Callable, Literal
import numpy as np
import cv2 as cv
import os


class Montage:

    def __init__(
        self,
        image_dir: str,
        block_height: int,
        block_width: int,
        lazy_load: bool = True,
        mode: Literal['bgr', 'lab'] = 'bgr',
        func: Callable[..., np.ndarray] = deltaE_euclidean,
        func_kwargs: dict = {},
    ) -> None:
        """
        图像蒙太奇
        
        Args:
            image_dir (str): 存储作为 block 区域图像的文件夹路径
            block_height (int): block 区域图像的高
            block_width (int): block 区域图像的宽
            lazy_load (bool, optional): 是否懒加载 block 区域图像，若为 False，则一次性缓存所有的 block 区域图像至内存中，该设置可显著加快程序运行速度，但不适用于 block 区域图像过多的场景。强烈建议图像数量不超过 1w 张以上时均设置为 False. Defaults to True.
            mode (Literal['bgr', 'lab'], optional): 衡量图像颜色差异的颜色空间，必须与 func 计算所使用的颜色空间对应，可选. Defaults to 'bgr'
            func (Callable, optional): 衡量颜色差异的可调用对象，该函数接受的第一个参数为参考颜色，第二个参数为比较颜色，函数签名请阅览 skimage.color 中提供的 deltaE_cie76, deltaE_ciede94, deltaE_ciede2000, deltaE_cmc 函数。可选. Defaults to deltaE_euclidean
            func_kwargs (dict, optional): 要传递给 func 的关键字参数，可选. Defaults to {}
        """
        # 作为 block 区域图像的文件夹路径
        self.image_dir = image_dir
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Directory {self.image_dir} does not exist.")
        # 作为 block 区域图像的文件路径列表
        names = os.listdir(self.image_dir)
        if not names:
            raise ValueError(f"Directory {self.image_dir} is empty.")
        self.image_paths = [os.path.join(self.image_dir, name) for name in names]
        # block 区域图像的高宽
        self.block_height, self.block_width = block_height, block_width
        self.block_size = (self.block_width, self.block_height)
        # 作为 block 区域图像的色调表，shape=(-1, 3)
        self.colormaps = np.empty((len(self.image_paths), 3), dtype=np.uint8)
        # 是否懒加载 block 区域图像
        self.lazy_load = lazy_load
        # 非懒加载 block 区域图像集
        if not self.lazy_load:
            self.images = np.empty((len(self.image_paths), self.block_height, self.block_width, 3), dtype=np.uint8)
        # 进度条
        with Progress() as progress:
            # 设置进度条任务
            block_task = progress.add_task("Processing block images...", total=len(self.image_paths))
            # 获取图像列表的色调表
            for i, image_file in enumerate(self.image_paths):
                image = safe_imread(image_file)
                image = cv.resize(image, dsize=self.block_size, interpolation=cv.INTER_CUBIC)
                self.colormaps[i] = self.calculate_dominant_color(image)
                if not self.lazy_load:
                    self.images[i] = image
                # 更新进度条
                progress.update(block_task, advance=1)
        # 衡量图像颜色差异的颜色空间
        self.mode = mode.lower()
        # 衡量颜色差异的可调用对象
        self.func = func
        # 要传递给 func 的关键字参数
        self.func_kwargs = func_kwargs
        # 匹配衡量图像颜色差异的颜色空间
        match self.mode:
            case 'bgr':
                pass
            case 'lab':
                self.colormaps = cv.cvtColor(self.colormaps.reshape(-1, 1, 3), cv.COLOR_BGR2LAB).reshape(-1, 3)
            case _:
                raise NotImplementedError()

    def calculate_dominant_color(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        逐步计算图像各通道像素的平均值，以获取图像颜色主色调
        
        Args:
            image (np.ndarray): 输入图像，shape=(h, w, 3)

        Returns:
            np.ndarray: 图像颜色主色调，shape=(3, )
        """
        c1 = np.mean(image[:, :, 0])
        c2 = np.mean(image[:, :, 1])
        c3 = np.mean(image[:, :, 2])
        return np.asarray((c1, c2, c3), dtype=np.uint8)

    def generate_image(
        self,
        image: np.ndarray,
        scale: float = 1.0,
        fit_size: bool = True,
        unique: bool = False,
    ) -> np.ndarray:
        """
        生成蒙太奇图像
        
        Args:
            image (np.ndarray): 要生成的蒙太奇图像对象
            scale (float, optional): 蒙太奇图像的缩放比例. Defaults to 1.0.
            fit_size (bool, optional): 是否在蒙太奇图像缩放过后，进一步将图像大小调整，以自适应匹配 block 区域图像的比例，避免在蒙太奇图像最右方与最下方的 block 区域图像出现截断所导致的显示不全现象. Defaults to True.
            unique (bool, optional): 是否限制每个 block 区域图像仅出现一次. Defaults to False.
            
        Returns:
            np.ndarray: 蒙太奇图像
        """
        #!使用 NumPy 切片性质，避免了图像高宽边界的条件判定，并运用矢量化操作进行性能优化
        # 原图像高宽
        height, width = image.shape[:2]
        # 缩放后图像高宽
        new_height, new_width = int(height * scale), int(width * scale)
        # 自适应比例
        if fit_size:
            remain_height = new_height % self.block_height
            remain_width = new_width % self.block_width
            new_height = new_height - remain_height if remain_height < self.block_height / 2 else new_height + self.block_height - remain_height
            new_width = new_width - remain_width if remain_width < self.block_width / 2 else new_width + self.block_width - remain_width
        new_size = (new_width, new_height)
        # 对原图像进行缩放
        image = cv.resize(image, dsize=new_size, interpolation=cv.INTER_CUBIC)
        # 每个 block 区域图像仅出现一次
        if unique:
            # 所需要的 block 区域图像个数
            block_count = np.ceil(new_height / self.block_height) * np.ceil(new_width / self.block_width)
            # 实际的 block 区域图像个数
            real_count = len(self.image_paths)
            assert block_count <= real_count, f"需要 {block_count} 个 block 区域图像，但 {self.image_dir} 只提供了 {real_count} 个 block 区域图像"
            # 掩码对象
            mask = np.zeros(real_count, dtype=bool)
        # 进度条
        with Progress() as progress:
            # 设置进度条任务
            rows_task = progress.add_task("[red]Processing image rows...", total=new_height // self.block_height)
            cols_task = progress.add_task("[green]Processing image columns...", total=new_width // self.block_width)
            # 遍历 block 区域图像在原图像中对应的像素
            for h in range(0, new_height, self.block_height):
                # 更新行进度条
                progress.update(rows_task, advance=1)
                # 重置列进度条
                progress.reset(cols_task)
                for w in range(0, new_width, self.block_width):
                    # 截取原图像中 block 区域
                    block_image = image[h:h + self.block_height, w:w + self.block_width]
                    # 实际切片大小（若切片范围超出了数组的实际大小，则切片操作将自动截取到数组的边界）
                    real_block_height, real_block_width = block_image.shape[:2]
                    real_block_size = (real_block_width, real_block_height)
                    # 计算原图像中 block 区域主色调与色调表的差异并找到最相近的图像
                    dominant_color = self.calculate_dominant_color(block_image)
                    # 匹配衡量图像颜色差异的颜色空间
                    match self.mode:
                        case 'bgr':
                            pass
                        case 'lab':
                            dominant_color = cv.cvtColor(dominant_color.reshape(-1, 1, 3), cv.COLOR_BGR2LAB).reshape(-1, 3)
                        case _:
                            raise NotImplementedError()
                    color_diffs = self.func(self.colormaps, dominant_color, **self.func_kwargs).reshape(-1)
                    # 将当前掩码位置下的值设置为 np.inf，这样在计算最大值索引时会被忽略
                    if unique:
                        color_diffs[mask] = np.inf
                    # 最小色调差异所在颜色表中索引
                    index = np.argmin(color_diffs)
                    # 更新掩码对象
                    if unique:
                        mask[index] = True
                    # 非懒加载 block 区域图像集
                    if not self.lazy_load:
                        # 加载最小色调差异 block 图像并修改其大小
                        replacement_image = self.images[index] if self.block_size == real_block_size else cv.resize(
                            self.images[index], dsize=real_block_size, interpolation=cv.INTER_CUBIC)
                    else:
                        # 读取最小色调差异 block 图像并修改其大小
                        replacement_image = safe_imread(self.image_paths[index])
                        replacement_image = cv.resize(replacement_image, dsize=real_block_size, interpolation=cv.INTER_CUBIC)
                    # 将其原图像中 block 区域替换
                    block_image[:, :] = replacement_image  # block_image = replacement_image 只是简单的更改了 block_image 的引用，而不是修改原图像的 block 区域，要修改原图像的 block 区域，必须通过切片操作（block_image 是 image 的视图（view））进行赋值
                    # 更新列进度条
                    progress.update(cols_task, advance=1)
        return image

    def generate_video(
        self,
        video_input_path: str,
        video_output_path: str,
        scale: float = 1.0,
        fit_size: bool = True,
        unique: bool = False,
    ) -> None:
        """
        读取视频帧，根据每一帧生成蒙太奇图像，并将其写入到给定视频文件中
        
        Args:
            video_input_path (str): 输入视频路径
            video_output_path (str): 输出视频路径
            scale (float, optional):  蒙太奇图像的缩放比例. Defaults to 1.0.
            fit_size (bool, optional): 是否在蒙太奇图像缩放过后，进一步将图像大小调整，以自适应匹配 block 区域图像的比例，避免在蒙太奇图像最右方与最下方的 block 区域图像出现截断所导致的显示不全现象. Defaults to True.
            unique (bool, optional): 是否限制每个 block 区域图像仅出现一次. Defaults to False.

        Raises:
            RuntimeError: 输入视频打开失败
            RuntimeError: 输出视频打开失败
        """
        # 打开视频
        capture = cv.VideoCapture(video_input_path)
        if not capture.isOpened():
            raise RuntimeError('VideoCapture is not initialized')
        # 获取视频的总帧数
        frame_count = capture.get(cv.CAP_PROP_FRAME_COUNT)
        # 获取视频帧数
        fps = capture.get(cv.CAP_PROP_FPS)
        # 指定编解码器的 fourcc
        fourcc = cv.VideoWriter_fourcc(*'DIVX')
        # 获取视频每帧图像大小
        size = (int(capture.get(cv.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv.CAP_PROP_FRAME_HEIGHT)))
        width, height = size
        # 缩放后图像高宽
        new_height, new_width = int(height * scale), int(width * scale)
        # 自适应比例
        if fit_size:
            remain_height = new_height % self.block_height
            remain_width = new_width % self.block_width
            new_height = new_height - remain_height if remain_height < self.block_height / 2 else new_height + self.block_height - remain_height
            new_width = new_width - remain_width if remain_width < self.block_width / 2 else new_width + self.block_width - remain_width
        new_size = (new_width, new_height)
        # 写入视频
        writer = cv.VideoWriter(video_output_path, fourcc=fourcc, fps=fps, size=new_size, isColor=True)
        if not writer.isOpened():
            raise RuntimeError('VideoWriter is not initialized')
        # 进度条
        with Progress() as progress:
            # 设置进度条任务
            frame_task = progress.add_task("[blue]Processing video frame...", total=frame_count)
            # 不适用于使用相机的情况 VideoCapture(0)
            for _ in range(frame_count):
                # 获取视频的每一帧图像
                ret, image = capture.read()
                # 逐帧生成蒙太奇图像
                montage_image = self.generate_image(image, scale=scale, fit_size=fit_size, unique=unique)
                # 视频帧写入
                writer.write(montage_image)
                # 更新进度条
                progress.update(frame_task, advance=1)
        # 释放读入
        capture.release()
        # 释放写入
        writer.release()
