# 利用CNN分类CIFAR-10数据集

在该文档中，主要通过训练卷积神经网络来完成对CIFAR-10数据集进行图像分类。

首先我的老电脑是没有GPU的，可以来看一下


```python
import torch
import numpy as np

# 检查是否可以利用GPU
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.')
else:
    print('CUDA is available!')

```

    CUDA is not available.
    

因为贫穷的原因确实是not available，所以文档到此结束。
咳咳，还试得硬着头皮搞的，时间太长跑不完就用co-lab嫖一下好了。

### 加载数据
加载训练数据集，并且将数据分为训练集和测试集，再为每个数据集创建DataLoader


```python
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

# number of subprocesses to use for data loading
num_workers = 0
# 每批加载16张图片
batch_size = 16
# percentage of training set to use as validation
valid_size = 0.2

# 将数据转换为torch.FloatTensor，并标准化。
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# 选择训练集与测试集的数据
train_data = datasets.CIFAR10('data', train=True,
                              download=True, transform=transform)
test_data = datasets.CIFAR10('data', train=False,
                             download=True, transform=transform)

# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
    sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
    num_workers=num_workers)

# 图像分类中10类别
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

```

    Files already downloaded and verified
    Files already downloaded and verified
    

### 定义CNN结构


```python
import torch.nn as nn
import torch.nn.functional as F

# 定义卷积神经网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 卷积层 (32x32x3的图像)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # 卷积层(16x16x16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # 卷积层(8x8x32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # 最大池化层
        self.pool = nn.MaxPool2d(2, 2)
        # linear layer (64 * 4 * 4 -> 500)
        self.fc1 = nn.Linear(64 * 4 * 4, 500)
        # linear layer (500 -> 10)
        self.fc2 = nn.Linear(500, 10)
        # dropout层 (p=0.3)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flatten image input
        x = x.view(-1, 64 * 4 * 4)
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = self.fc2(x)
        return x

# create a complete CNN
model = Net()
print(model)

# 使用GPU
if train_on_gpu:
    model.cuda()

```

    Net(
      (conv1): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv2): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (fc1): Linear(in_features=1024, out_features=500, bias=True)
      (fc2): Linear(in_features=500, out_features=10, bias=True)
      (dropout): Dropout(p=0.3, inplace=False)
    )
    

### 选择损失函数与优化函数


```python
import torch.optim as optim
# 使用交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# 使用随机梯度下降，学习率lr=0.01
optimizer = optim.SGD(model.parameters(), lr=0.01)

```

### 训练卷积神经网络模型


```python
# 训练模型的次数
n_epochs = 30

valid_loss_min = np.Inf # track change in validation loss

for epoch in range(1, n_epochs+1):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    
    ###################
    # 训练集的模型 #
    ###################
    model.train()
    for data, target in train_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()*data.size(0)
        
    ######################    
    # 验证集的模型#
    ######################
    model.eval()
    for data, target in valid_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update average validation loss 
        valid_loss += loss.item()*data.size(0)
    
    # 计算平均损失
    train_loss = train_loss/len(train_loader.sampler)
    valid_loss = valid_loss/len(valid_loader.sampler)
        
    # 显示训练集与验证集的损失函数 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    
    # 如果验证集损失函数减少，就保存模型。
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
        torch.save(model.state_dict(), 'model_cifar.pt')
        valid_loss_min = valid_loss

```

    h:\python\lib\site-packages\torch\nn\functional.py:718: UserWarning: Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable. (Triggered internally at  ..\c10/core/TensorImpl.h:1156.)
      return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
    

    Epoch: 1 	Training Loss: 2.078950 	Validation Loss: 1.758009
    Validation loss decreased (inf --> 1.758009).  Saving model ...
    Epoch: 2 	Training Loss: 1.620644 	Validation Loss: 1.498452
    Validation loss decreased (1.758009 --> 1.498452).  Saving model ...
    Epoch: 3 	Training Loss: 1.451228 	Validation Loss: 1.313831
    Validation loss decreased (1.498452 --> 1.313831).  Saving model ...
    Epoch: 4 	Training Loss: 1.332905 	Validation Loss: 1.222225
    Validation loss decreased (1.313831 --> 1.222225).  Saving model ...
    Epoch: 5 	Training Loss: 1.230245 	Validation Loss: 1.148164
    Validation loss decreased (1.222225 --> 1.148164).  Saving model ...
    Epoch: 6 	Training Loss: 1.142225 	Validation Loss: 1.068547
    Validation loss decreased (1.148164 --> 1.068547).  Saving model ...
    Epoch: 7 	Training Loss: 1.067228 	Validation Loss: 1.008666
    Validation loss decreased (1.068547 --> 1.008666).  Saving model ...
    Epoch: 8 	Training Loss: 1.012903 	Validation Loss: 0.939235
    Validation loss decreased (1.008666 --> 0.939235).  Saving model ...
    Epoch: 9 	Training Loss: 0.954310 	Validation Loss: 0.920974
    Validation loss decreased (0.939235 --> 0.920974).  Saving model ...
    Epoch: 10 	Training Loss: 0.903835 	Validation Loss: 0.923358
    Epoch: 11 	Training Loss: 0.867460 	Validation Loss: 0.848017
    Validation loss decreased (0.920974 --> 0.848017).  Saving model ...
    Epoch: 12 	Training Loss: 0.827535 	Validation Loss: 0.818963
    Validation loss decreased (0.848017 --> 0.818963).  Saving model ...
    Epoch: 13 	Training Loss: 0.791537 	Validation Loss: 0.826229
    Epoch: 14 	Training Loss: 0.765741 	Validation Loss: 0.812143
    Validation loss decreased (0.818963 --> 0.812143).  Saving model ...
    Epoch: 15 	Training Loss: 0.733144 	Validation Loss: 0.785563
    Validation loss decreased (0.812143 --> 0.785563).  Saving model ...
    Epoch: 16 	Training Loss: 0.702631 	Validation Loss: 0.760952
    Validation loss decreased (0.785563 --> 0.760952).  Saving model ...
    Epoch: 17 	Training Loss: 0.675755 	Validation Loss: 0.766690
    Epoch: 18 	Training Loss: 0.650206 	Validation Loss: 0.763200
    Epoch: 19 	Training Loss: 0.628957 	Validation Loss: 0.753950
    Validation loss decreased (0.760952 --> 0.753950).  Saving model ...
    Epoch: 20 	Training Loss: 0.603371 	Validation Loss: 0.746298
    Validation loss decreased (0.753950 --> 0.746298).  Saving model ...
    Epoch: 21 	Training Loss: 0.577675 	Validation Loss: 0.760672
    Epoch: 22 	Training Loss: 0.560931 	Validation Loss: 0.729024
    Validation loss decreased (0.746298 --> 0.729024).  Saving model ...
    Epoch: 23 	Training Loss: 0.534910 	Validation Loss: 0.725528
    Validation loss decreased (0.729024 --> 0.725528).  Saving model ...
    Epoch: 24 	Training Loss: 0.518099 	Validation Loss: 0.746252
    Epoch: 25 	Training Loss: 0.501059 	Validation Loss: 0.731015
    Epoch: 26 	Training Loss: 0.484397 	Validation Loss: 0.726336
    Epoch: 27 	Training Loss: 0.468918 	Validation Loss: 0.753496
    Epoch: 28 	Training Loss: 0.449131 	Validation Loss: 0.745466
    Epoch: 29 	Training Loss: 0.437546 	Validation Loss: 0.726344
    Epoch: 30 	Training Loss: 0.423924 	Validation Loss: 0.740021
    

### 测试训练好的数据


```python
# track test loss
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval()
# iterate over test data
for data, target in test_loader:
    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the batch loss
    loss = criterion(output, target)
    # update test loss 
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)    
    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    # calculate test accuracy for each object class
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# average test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))

```

    Test Loss: 0.735211
    
    Test Accuracy of airplane: 80% (803/1000)
    Test Accuracy of automobile: 87% (878/1000)
    Test Accuracy of  bird: 68% (684/1000)
    Test Accuracy of   cat: 60% (600/1000)
    Test Accuracy of  deer: 69% (698/1000)
    Test Accuracy of   dog: 66% (660/1000)
    Test Accuracy of  frog: 83% (834/1000)
    Test Accuracy of horse: 76% (764/1000)
    Test Accuracy of  ship: 88% (880/1000)
    Test Accuracy of truck: 79% (790/1000)
    
    Test Accuracy (Overall): 75% (7591/10000)
    

结果是75%的准确率，这样一个大概的简单的神经网络的训练模型就完成了，非常简洁，但是可以看到效果的的确确是有的。


```python

```
