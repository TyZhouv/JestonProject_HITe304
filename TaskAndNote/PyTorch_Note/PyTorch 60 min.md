# PyTorch_Note
# PyTorch Deep Learning  : 60 min [Weblink](https://pytorch.apachecn.org/#/docs/1.7/02)
## 什么是PyTorch？
* 无缝替换NumPy，并且通过利用GPU的算力来实现神经网络的加速。  
* 通过自动微分机制，来让神经网络的实现变得更加容易。  

##本次教程的目标：
* 深入了解PyTorch的张量单元以及如何使用Pytorch来搭建神经网络。
* 自己动手训练一个小型神经网络来实现图像的分类。  

##张量
张量如同数组和矩阵一样, 是一种特殊的数据结构。在PyTorch中, 神经网络的输入、输出以及网络的参数等数据, 都是使用张量来进行描述。

张量的使用和Numpy中的ndarrays很类似, 区别在于**张量可以在GPU或其它专用硬件上运行, 这样可以得到更快的加速效果。** 如果你对ndarrays很熟悉的话, 张量的使用对你来说就很容易了。  
直接上Demo:
```python
import torch
import numpy as np

# 1.Create a tensor directly
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
# 2.Create a tendor With Numpy
# np_array = np.array(data)
# x_np = torch.from_numpy(np_array)
```
[Tensor与Numpy的转化](https://pytorch.apachecn.org/#/docs/1.7/03?id=jump)  
通过已有的张量来生成新的张量、通过指定数据维度来生成张量 soso
**张量属性**  
从张量属性我们可以得到张量的维数、数据类型以及它们所存储的设备(CPU或GPU)。
```python
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
```
out:
```
Shape of tensor: torch.Size([3, 4])   # 维数
Datatype of tensor: torch.float32     # 数据类型
Device tensor is stored on: cpu       # 存储设备

```
**张量运算**
有超过100种张量相关的运算操作, 例如转置、索引、切片、数学运算、线性代数、随机采样等。更多的运算可以在这里[查看](https://pytorch.org/docs/stable/torch.html)。

所有这些运算都可以在GPU上运行(相对于CPU来说可以达到更高的运算速度)。如果你使用的是Google的Colab环境, 可以通过 Edit > Notebook Settings 来分配一个GPU使用。  
1. 张量的索引和切片
```python
tensor = torch.ones(4, 4)
tensor[:,1] = 0            # 将第1列(从0开始)的数据全部赋值为0
print(tensor)
```
display：
```
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
```
2. 张量的拼接
```python
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)
```
dispaly:  

![image](https://user-images.githubusercontent.com/68489038/153022116-b81a849f-0c88-4352-8081-bd93dc47c085.png)  

3. 张量的乘积和矩阵乘法
4. [自动赋值运算](https://pytorch.apachecn.org/#/docs/1.7/03)
##torch.autograd的简要介绍  
torch.autograd是 PyTorch 的自动差分引擎，可为神经网络训练提供支持。 在本节中，您将获得有关 Autograd 如何帮助神经网络训练的概念性理解。  
神经网络（NN）是在某些输入数据上执行的嵌套函数的集合。 这些函数由参数（由权重和偏差组成）定义，这些参数在 PyTorch 中存储在张量中。

训练 NN 分为两个步骤：

正向传播：在正向传播中，NN 对正确的输出进行最佳猜测。 它通过其每个函数运行输入数据以进行猜测。

反向传播：在反向传播中，NN 根据其猜测中的误差调整其参数。 它通过从输出向后遍历，收集有关函数参数（梯度）的误差导数并使用梯度下降来优化参数来实现。 有关反向传播的更详细的演练，请查看 3Blue1Brown 的视频。  


