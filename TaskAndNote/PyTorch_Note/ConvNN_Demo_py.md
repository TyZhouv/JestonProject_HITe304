# 1.Neural-NetWork
## Overview  
![ConvNN](https://cs231n.github.io/assets/nn1/neural_net2.jpeg)  
![1](https://pytorch.apachecn.org/docs/1.7/img/3250cbba812d68265cf7815d987bcd1b.png)    
[CNNS in detail](https://cs231n.github.io/convolutional-networks/#conv)  

可以使用torch.nn包构建Conv神经网络。
nn依赖于autograd来定义模型并对其进行微分。 nn.Module包含层，以及返回output的方法forward(input)。

[PyTorchLink](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py)  

***
**神经网络的典型训练过程如下：**
1. 定义具有一些可学习参数（或权重）的神经网络
2. 遍历输入数据集
3. 通过网络处理输入
4. 计算损失（输出正确的距离有多远）
5. 将梯度传播回网络参数
6. 通常使用简单的更新规则来更新网络的权重：weight = weight - learning_rate * gradient  

## 2.torch.nn.Conv2d() Func Description
|param|paramType|Description|
|----|----|----|
|in_channels|int|Number of channels in the input image|
|out_channels|int|Number of channels produced by the convolution|
|kernels_size|int or tuple|Size of the convolving kernel|
## 3.Define the Network
```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
				
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  #first param is input_size,second param is hidden_size
        self.fc2 = nn.Linear(120, 84)#2th hidden_size output_size , 3th hidden_size input
        self.fc3 = nn.Linear(84, 10)#3th hidden_size input , output size (num_classes)
				#Normally we call input and output layer as fully connected layer.
				
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
print(net)

```
out:
```
Net(
  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
```

