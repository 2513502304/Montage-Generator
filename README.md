# ***Montage-Generator***

一个应用于视频或图像的蒙太奇生成工具。通过将原始视频帧或给定图像分割为多个小块，并用色调相近的图像块代替原始图块，生成具有艺术效果的蒙太奇图像。

---

## 功能简介

- **图像蒙太奇生成**：支持自定义小块大小、生成的蒙太奇图像大小，并根据提供的小块大小，自适应调整生成蒙太奇图像的比例，避免在蒙太奇图像中内嵌的小块出现显示不全现象。
- **可选小块去重**：支持小块去重选项，通过限制每个小块仅出现一次，确保在大规模数据集下生成的蒙太奇图像中，每个小块都独一无二，充分发挥海量数据集的优势，以还原完美的蒙太奇图像。
- **高效处理**：支持懒加载、缓存对图像块训练过后的蒙太奇对象，适用于大规模图像数据集。对于 18048 x 12258 的蒙太奇图像，使用 96 x 54 的小块图像进行填充，在 Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz 下只需大约 3s 即可生成！
- **视频蒙太奇生成**：支持对视频逐帧处理，生成动态的蒙太奇视频。

---

## 示例展示

### 输入与输出示例 1

- **输入图像**：`Yui-Azusa.jpg`  
![image](https://github.com/user-attachments/assets/1d644b2d-8966-4907-a7e4-89d198d6a5a6)

- **输出蒙太奇图像**：
![image](https://github.com/user-attachments/assets/ab7c9938-1125-4d5e-88b9-abb45ea32a0c)

- **输出蒙太奇图像细节**：
![image](https://github.com/user-attachments/assets/5e844eee-957a-40cc-9dff-95e81894c018)

---
### 输入与输出示例 2

- **输入图像**：`Mio-Ritsu-Tsumugi.jpg`  
![image](https://github.com/user-attachments/assets/5b34b8e0-b419-428a-91fa-36b2706e74bc)

- **输出蒙太奇图像**：
![image](https://github.com/user-attachments/assets/ef4a7aa5-6cf7-4b65-8753-c2719aa5fee4)

- **输出蒙太奇图像细节**：
![image](https://github.com/user-attachments/assets/432d6e8e-15e1-4338-96b7-f161d7531c49)

---

## 使用方法

1. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```
   
2. **配置参数**：
   在 settings.py 文件中修改以下参数：
   - `image_dir`：存储作为 block 区域图像的文件夹路径。
   - `block_height` 和 `block_width`：block 区域图像的高宽。
   - `lazy_load`：是否懒加载 block 区域图像，若为 False，则一次性缓存所有的 block 区域图像至内存中，该设置可显著加快程序运行速度，但不适用于 block 区域图像过多的场景。强烈建议图像数量不超过 1w 张以上时均设置为 False。
   - `scale`：蒙太奇图像的缩放比例。
   - `fit_size`：是否在蒙太奇图像缩放过后，进一步将图像大小调整，以自适应匹配 block 区域图像的比例，避免在蒙太奇图像最右方与最下方的 block 区域图像出现截断所导致的显示不全现象。
   - `unique`：是否限制每个 block 区域图像仅出现一次。
   - `image_input_path`：蒙太奇图像输入路径。
   - `image_output_path`：蒙太奇图像输出路径。
   - `pkl_path`：保存训练好的蒙太奇对象路径。若提供该路径，则优先加载该路径下的蒙太奇对象，并判断是否该对象参数是否与当前配置参数是否一致，若一致则直接使用该对象。若不一致，或者未提供该路径，则会训练对应的蒙太奇对象，并将其保存至该路径
   - `video_input_path`：输入视频路径
   - `video_output_path`：输出视频路径
   - `mode`：将蒙太奇效果应用于图像（image）还是视频（video）

3. **运行程序**：
   ```bash
   python main.py
   ```
   
---

## 附加程序

使用 PyAutoGUI 实现的 KonArchive 自动截屏脚本，用以获取 KonArchive 软件本体中 423 张 K-ON! 1920 x 1080 大图

1. **安装依赖**：
安装 [KonArchive](https://archive.org/details/k-on-k-on-movie-illustration-archives-2009-2012) 软件本体

![image](https://github.com/user-attachments/assets/7a5ee476-3638-43a0-9401-05e8501ae543)

2. **配置参数**：
   在 KonArchiveAutoScreenshot.py 文件中修改以下参数：
   - `exec_file`：存储作为 block 区域图像的文件夹路径。
   - `screenshot_dir`：保存截屏的文件夹路径。
   - `debug_dir`：程序运行时调试文件夹路径。

3. **运行程序**：
   ```bash
   python KonArchiveAutoScreenshot.py
   ```
   
---

## 项目结构

```
Montage-Generator/
├── Data/                             # 数据文件夹
│   ├── input/                        # 输入图像或视频
│   ├── output/                       # 输出图像或视频
│   ├── metric/                       # 运行 demo 脚本后，自动生成的图像文件
│   ├── src/                          # KonArchive 自动截屏脚本所需的资源文件
│   ├── K-ON!/                        # KonArchive 自动截屏脚本所截取的图像文件
│   └── debug/                        # 运行 KonArchive 自动截屏脚本后，自动生成的调试文件
├── montage.py                        # 核心蒙太奇生成逻辑
├── settings.py                       # 配置文件
├── main.py                           # 主程序入口
├── utils.py                          # 实用函数文件
├── KonArchiveAutoScreenshot.py       # KonArchive 自动截屏脚本
├── demo.py                           # 评估不同颜色差异计算方法的显示效果，并将其写入到文件名为函数名 + 函数参数的图像文件中
├── README.md                         # 项目说明文件
├── requirements.txt                  # 依赖库列表
└── LICENSE                           # 许可证
```

---

## 依赖环境

- Python 3.8 或更高版本

---

## 贡献指南

欢迎对本项目提出建议或贡献代码！请通过以下步骤参与贡献：

1. Fork 本仓库。

2. 创建新分支：
   ```bash
   git checkout -b feature/your-feature-name
   ```
   
3. 提交更改并推送到你的分支：
   ```bash
   git commit -m "Add your commit message"
   git push origin feature/your-feature-name
   ```

4. 提交 Pull Request。

---
## 许可证

本项目基于 MIT License 开源。

---
## 联系方式

如有任何问题或建议，请联系作者：
- **Email**: 2513502304@qq.com
