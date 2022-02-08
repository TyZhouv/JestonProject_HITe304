# Demos in PyTorch NN  
## Training Demo
```python
#让我们来看一个训练步骤。 对于此示例，我们从torchvision加载了经过预训练的 resnet18 模型。  
import torch, torchvision
model = torchvision.models.resnet18(pretrained=True)
#我们创建一个随机数据张量来表示具有 3 个通道的单个图像，高度&宽度为 64，其对应的label初始化为一些随机值。
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)#这是label 正确值

#接下来，我们通过模型的每一层运行输入数据以进行预测。 这是正向传播。  
prediction = model(data) # forward pass 这是网络初始值计算出的预测值

#我们使用模型的预测和相应的标签来计算误差（loss）。 下一步是通过网络反向传播此误差。 当我们在误差张量上调用.backward()时，开始反向传播。  
#然后，Autograd 会为每个模型参数计算梯度并将其存储在参数的.grad属性中。
loss = (prediction - labels).sum() #初始网络模型计算出的预测值 - 给定的label
loss.backward() # backward pass

#接下来，我们加载一个优化器，在本例中为 SGD，学习率为 0.01，动量为 0.9。 我们在优化器中注册模型的所有参数。  
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

#最后，我们调用.step()启动梯度下降。 优化器通过.grad中存储的梯度来调整每个参数。
optim.step() #gradient descent
```
[以下各节详细介绍了 Autograd 的工作原理以及计算图](https://pytorch.apachecn.org/#/docs/1.7/04)，后续进行学习，**至此已经有了用label和网络模型进行训练求解网络模型参数的步骤**。
