SDAE: Stacked Denoising Autoencoder
此项目是一个基于 Stacked Denoising Autoencoder (SDAE) 的图像去噪模型，能够在 MNIST 手写数字数据集上实现图像的降噪。通过引入噪声数据并使用 SDAE 进行训练和测试，最终实现图像的降噪效果。

项目概览
SDAE 是一种堆叠式的自编码器架构，用于通过学习数据的低维表示来进行降噪处理。该项目通过 SDAE 架构对 MNIST 数据集中的手写数字进行去噪实验，效果图包含噪声图、原图、去噪图，便于对比。

文件目录
sdae文件: 包含实现 SDAE 模型的代码文件。
MNIST图像数据: 示例数据集，包含手写数字图像及其噪声、原图、去噪后的对比图。
命令行运行截图: 运行 SDAE 模型的命令行输出截图。
快速开始
环境要求
Python 3.8.19
依赖库：TensorFlow 或 PyTorch，numpy，matplotlib 

许可证
本项目采用 MIT License。
