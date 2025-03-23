from typing import Literal

# 存储作为 block 区域图像的文件夹路径
image_dir: str = './Data/K-ON!'

# block 区域图像的高
block_height: int = 9 * 6

# block 区域图像的宽
block_width: int = 16 * 6

# 是否懒加载 block 区域图像，若为 False，则一次性缓存所有的 block 区域图像至内存中，该设置可显著加快程序运行速度，但不适用于 block 区域图像过多的场景。强烈建议图像数量不超过 1w 张以上时均设置为 False
lazy_load: bool = False

# 蒙太奇图像的缩放比例
scale: int | float = 3.0

# 是否在蒙太奇图像缩放过后，进一步将图像大小调整，以自适应匹配 block 区域图像的比例，避免在蒙太奇图像最右方与最下方的 block 区域图像出现截断所导致的显示不全现象
fit_size: bool = True

# 蒙太奇图像输入路径
image_input_path: str = './Data/input/Yui-Azusa.jpg'

# 蒙太奇图像输出路径
image_output_path: str = './Data/output/Yui-Azusa.jpg'

# 保存训练好的蒙太奇对象路径
# 若提供该路径，则优先加载该路径下的蒙太奇对象，并判断是否该对象参数是否与当前配置参数是否一致，若一致则直接使用该对象
# 若不一致，或者未提供该路径，则会训练对应的蒙太奇对象，并将其保存至该路径
pkl_path: str = 'K-ON!.pkl'

# 输入视频路径
video_input_path: str = ''

# 输出视频路径
video_output_path: str = ''

# 将蒙太奇效果应用于图像（image）还是视频（video）
mode: Literal['image', 'video'] = 'image'
