# [Conv_NN_TutorialLink](https://cs231n.github.io/convolutional-networks/#conv)
## 1. ConvNN OverView  

Convolutional Neural Networks are very similar to ordinary Neural Networks from the previous chapter: they are made up of neurons that have learnable weights and biases. Each neuron receives some inputs, performs a dot product and optionally follows it with a non-linearity. The whole network still expresses a single differentiable score function: from the raw image pixels on one end to class scores at the other. And they still have a loss function (e.g. SVM/Softmax) on the last (fully-connected) layer and all the tips/tricks we developed for learning regular Neural Networks still apply.

So what changes? ConvNet architectures make the explicit assumption that the inputs are images, which allows us to encode certain properties into the architecture. These then make the forward function more efficient to implement and vastly reduce the amount of parameters in the network.
## 2.Architecture Overview
常规神经网络。正如我们在前一章中看到的，神经网络接收一个输入（单个向量），并通过一系列隐藏层对其进行转换。每个隐藏层由一组神经元组成，其中每个神经元与前一层中的所有神经元完全连接，并且单层中的神经元完全独立运行，不共享任何连接。最后一个全连接层称为“输出层”，在分类设置中它代表类分数。

常规神经网络不能很好地扩展到完整图像。在 CIFAR-10 中，图像的大小仅为 32x32x3（32 宽，32 高，3 个颜色通道），因此常规神经网络的第一个隐藏层中的单个全连接神经元将具有 32*32*3 = 3072 个权重. 这个数量似乎仍然可以管理，但显然这种完全连接的结构无法扩展到更大的图像。例如，更大尺寸的图像（例如 200x200x3）会导致神经元的权重为 200*200*3 = 120,000。此外，我们几乎肯定想要有几个这样的神经元，所以参数会很快加起来！显然，这种完全连接是浪费的，大量的参数会很快导致过度拟合。

神经元的 3D 体积。卷积神经网络利用了输入由图像组成的事实，它们以更明智的方式约束架构。特别是，与常规神经网络不同，ConvNet 的层具有按 3 个维度排列的神经元：宽度、高度、深度。（注意这个词的深度这里指的是一个激活体积的第三维，而不是一个完整的神经网络的深度，可以指一个网络中的总层数。）例如，CIFAR-10中的输入图像是一个输入体积激活，并且体积的尺寸为 32x32x3（分别为宽度、高度、深度）。正如我们很快将看到的，一层中的神经元只会连接到前一层的一小部分区域，而不是以全连接方式连接到所有神经元。此外，CIFAR-10 的最终输出层将具有 1x1x10 的尺寸，因为在 ConvNet 架构结束时，我们会将整个图像缩减为单个类别分数向量，沿深度维度排列。这是一个可视化：
![1](https://cs231n.github.io/assets/cnn/cnn.jpeg)
左：常规的 3 层神经网络。右图：ConvNet 在三个维度（宽度、高度、深度）中排列其神经元，如其中一层所示。ConvNet 的每一层都将 3D 输入量转换为神经元激活的 3D 输出量。在此示例中，红色输入层保存图像，因此其宽度和高度将是图像的尺寸，深度将为 3（红、绿、蓝通道）。
***
```
ConvNet 由层组成。每个层都有一个简单的 API：它将输入 3D 体积转换为输出 3D 体积，  
并带有一些可能有参数也可能没有参数的可微函数。
```
## 3.Layers used to build ConvNets
 We use three main types of layers to build ConvNet architectures:   
 **Convolutional Layer, Pooling Layer, and Fully-Connected Layer (exactly as seen in regular Neural Networks)**  
 We will stack these layers to form a full ConvNet architecture.
 **note:** 用于 CIFAR-10 分类的简单 ConvNet 可能具有 [INPUT - CONV - RELU - POOL - FC] 架构:
* INPUT [32x32x3] 将保存图像的原始像素值，在这种情况下，图像的宽度为 32，高度为 32，并具有三个颜色通道 R、G、B。
* CONV 层将计算连接到输入中局部区域的神经元的输出，每个神经元计算它们的权重和它们在输入体积中连接到的小区域之间的点积。如果我们决定使用 12 个过滤器，这可能会导致诸如 [32x32x12] 的体积。
* RELU 层将应用元素激活函数，例如将 \(max(0,x)\) 阈值设置为零。这使卷的大小保持不变（[32x32x12]）。
* POOL 层将沿空间维度（宽度、高度）执行下采样操作，从而产生诸如 [16x16x12] 的体积。  
* FC（即全连接）层将计算类别分数，产生大小为 [1x1x10] 的体积，其中 10 个数字中的每一个都对应一个类别分数，例如 CIFAR-10 的 10 个类别中。与普通神经网络一样，顾名思义，这一层中的每个神经元都将连接到前一卷中的所有数字。

通过这种方式，ConvNets 将原始图像从原始像素值逐层转换为最终的类分数。  
请注意，某些层包含参数，而其他层则不包含。特别是，CONV/FC 层执行的转换不仅是输入体积中的激活函数，而且是参数（神经元的权重和偏差）的函数。另一方面，RELU/POOL 层将实现一个固定的功能。CONV/FC 层中的参数将使用梯度下降进行训练，以便 ConvNet 计算的类分数与训练集中每个图像的标签一致。
 ***
### 1. conv kernel how to acculate:
![11](https://pic1.zhimg.com/v2-6428cf505ac1e9e1cf462e1ec8fe9a68_b.webp)  
卷积后的特征图大小可由下面公式进行计算，这里卷积后尺寸大小为 Hn , Hn-1表示输入特征图大小：
![22](https://www.zhihu.com/equation?tex=H_n%3D%5Cfrac%7BH_%7Bn-1%7D-k%2B2p%7D%7Bs%7D%2B1)  
![88](https://cs231n.github.io/assets/cnn/stride.jpeg)  
为了方便理解，这里举个栗子。对于输入维度为(255，255，3)的RGB图像，采用卷积核大小3 * 3 * 3进行卷积，步长为2，边缘填充为1，卷积核的个数为16。那么得到的相应的输出特征图大小为(255-3+4)/2+1 = 129，即129 * 129大小，一共16个。所以最终输出的张量大小为129*129*16。
![33](https://pic4.zhimg.com/80/v2-dcc083f2efd13537016a9f659f5b0c4b_720w.jpg)
卷积按照特征图的卷积维度不同可以分成一维卷积、二维卷积和三维卷积。

* 一维卷积（1d conv）：卷积核在一维空间进行滑动计算，通常用于对文本、股票价格等序列的处理，如图5所示。
* 二维卷积（2d conv）：卷积核在二维空间进行滑动计算，用于对图像的特征提取；
* 三维卷积（3d conv）：卷积核在三维空间进行滑动计算，用于视频数据的处理；
![44](https://pic3.zhimg.com/80/v2-dd8ef8d8ce84e3f6715b21421e17d736_720w.jpg)  
![55](https://pic1.zhimg.com/80/v2-210f200f49d9cf8f6801c011ea74dc50_720w.jpg)  
![66](https://pic2.zhimg.com/80/v2-b0ce446e6169021a4eca9939234babb5_720w.jpg)
### 2.卷积参数量的计算
对于一层卷积而言，其中的每个卷积核的参数量即为同卷积核的大小，那么对于有 [公式] 个卷积核的卷积层而言，并加上了偏置项。其参数量P为：
![77](https://www.zhihu.com/equation?tex=P%3D%28k%2Ak%2Ac_%7Bin%7D%2B1%29%2Ac_%7Bout%7D)  
[others](https://zhuanlan.zhihu.com/p/63174774)
***



 
