"""
Author: 未来可欺 2513502304@qq.com
Date: 2025-03-22 12:31:06
LastEditors: 未来可欺 2513502304@qq.com
LastEditTime: 2025-03-23 19:55:54
Description: 应用于视频或图像的蒙太奇生成工具
"""

from montage import Montage, safe_imread, safe_imwrite
import settings
import cv2 as cv
import joblib
import os

def main():
    if os.path.exists(settings.pkl_path):
        montage: Montage = joblib.load(settings.pkl_path)
        different = montage.image_dir != settings.image_dir or montage.block_height != settings.block_height or montage.block_width != settings.block_width or montage.lazy_load != settings.lazy_load
        if different:
            montage = Montage(image_dir=settings.image_dir, block_height=settings.block_height, block_width=settings.block_width, lazy_load=settings.lazy_load)
            joblib.dump(montage, settings.pkl_path)
    else:
        montage: Montage = Montage(image_dir=settings.image_dir, block_height=settings.block_height, block_width=settings.block_width, lazy_load=settings.lazy_load)
        joblib.dump(montage, settings.pkl_path)
    match settings.mode.lower():
        case 'image':
            image = safe_imread(settings.image_input_path)
            montage_image = montage.generate_image(image=image, scale=settings.scale, fit_size=settings.fit_size)
            safe_imwrite(settings.image_output_path, montage_image)
        case 'video':
            montage.generate_video(video_input_path=settings.video_input_path, video_output_path=settings.video_output_path, scale=settings.scale, fit_size=settings.fit_size)
        case _:
            raise NotImplementedError()
        
if __name__ == '__main__':
    main()