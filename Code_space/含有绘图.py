
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F

# 数据处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为tensor格式
    # 对图像进行标准化，即让数据集的图像的均值变为0，标准差变为1，把图片3个通道中的数据整理到[-1,1]的区间中
    # 输入的参数第一个括号内是3个通道的均值，第二个是3个通道的标准差，这些数据需要自己算好再放进这个函数里，不然每次运行normalize函数都要遍历一遍数据集
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 定义训练集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# 如果没有import torch.utils.data，这里会出现Warning:Cannot find reference ‘data‘ in ‘__init__.py‘
# torch.utils.data.DataLoader用于将已有的数据读取接口的输入按照batch size封装成Tensor
# shuffle参数表示是否在每个epoch后打乱数据；num_workers表示用多少个子进程加载数据，0表示数据将在主进程中加载，默认为0，这里不知道为啥设置多线程会报错
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False)
# testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# 显示图片函数
def imshow(img):
    img = img / 2 + 0.5  # 逆归一化，公式似乎是img/(均值*batchsize)+方差
    # numpy不能读取CUDA tensor 需要将它转化为 CPU tensor
    npimg = img.cpu().numpy()
    # plt.imshow()中应有的参数为(imagesize1,imagesize2,channels)，在RGB图像中，channels=3，imagesize1为行数，imagesize2为列数，即分别为图片的高和宽
    # npimg中的参数顺序为(channels,imagesize1,imagesize2)
    # np.transpose(0,2,1)表示将数据的第二维和第三维交换
    # 则np.transpose(npimg, (1, 2, 0))就能将npimg变成(imagesize1,imagesize2,channels)的参数顺序，然后输入plt.imshow()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    plt.pause(0.5)
    plt.close()


# get some random training images得到随机的batchsize张图片（随机是因为之前定义trainloader的时候设置开启了随机）
# dataloader本质上是一个可迭代对象，可以使用iter()进行访问，采用iter(dataloader)返回的是一个迭代器，然后可以使用next()或者enumerate访问
dataiter = iter(trainloader)
# 访问iter(dataloader)时，imgs在前，labels在后，分别表示：图像转换0~1之间的值，labels为标签值（在这里labels就是图像所属的分类的标号）。并且imgs和labels是按批次进行输入的。
# 因为之前设置了batch_size=4，所以这里的images中会有4张图片
images, labels = dataiter.next()

# show images
# torchvision.utils.make_grid()将多张图片组合成一张图片，padding为多张图片之间的间隙
imshow(torchvision.utils.make_grid(images, padding=2))
# 按顺序输出四张图片的标签（所属分类的名字）
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


# 定义网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 第一个卷积，输入1通道，用6个5×5的卷积核
        self.conv1 = nn.Conv2d(3, 6, (5, 5))
        # 2×2maxpool池化
        self.pool = nn.MaxPool2d(2, 2)
        # 第二个卷积，输入6个通道（因为上一层卷积中用了6个卷积核），用16个5×5的卷积核
        self.conv2 = nn.Conv2d(6, 16, (5, 5))
        # 第一个全连接层，输入16个5×5的图像（5×5是因为最开始输入的是32×32的图像，然后经过2个卷积2个池化变成了5×5），用全连接将其变为120个节点（一维化）
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 将120个节点变为84个
        self.fc2 = nn.Linear(120, 84)
        # 将84个节点变为10个
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # conv1的结果先用ReLU激活，再以2×2的池化核做max pool池化
        x = self.pool(F.relu(self.conv1(x)))
        # conv2的结果先用ReLU激活，再以2×2的池化核做max pool池化
        x = self.pool(F.relu(self.conv2(x)))
        # 将高维向量转为一维
        x = x.view(-1, 16 * 5 * 5)
        # 用ReLU激活fc1的结果
        x = F.relu(self.fc1(x))
        # 用ReLU激活fc2的结果
        x = F.relu(self.fc2(x))
        # 计算出fc3的结果
        x = self.fc3(x)
        return x


net = Net()

# 如果GPU（CUDA)可用，则用GPU，否则用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 打印使用的是GPU/CPU
print(device)
# 将网络放置到GPU
net.to(device)


plt.ion()
loss_data = [0]
times = [0]
for epoch in range(3):  # 多次循环访问整个数据集（这里用了两个epoch，即循环访问2遍整个数据集）

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs   得到输入的batchsize张图片，放在inputs中，labels存储该图片所属分类的名字
        inputs, labels = data
        # 将数据放置到GPU上
        inputs, labels = inputs.to(device), labels.to(device)
        # 计算交叉熵https://zhuanlan.zhihu.com/p/98785902
        criterion = nn.CrossEntropyLoss()
        # optim.SGD表示使用随机梯度下降算法
        # lr是学习率；momentum是动量（在梯度下降算法中添加动量法）https://blog.csdn.net/weixin_40793406/article/details/84666803
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        # 归零参数梯度
        optimizer.zero_grad()

        # forward + backward + optimize
        # 向神经网络输入数据，然后得到输出outputs
        outputs = net(inputs)
        # 输入神经网络输出的预测和实际的数据的labels，计算交叉熵（偏差）
        loss = criterion(outputs, labels)
        # 将误差反向传播
        loss.backward()
        # 更新所有参数（权重）
        optimizer.step()

        # 累加经过这一个batchsize张图片学习后的误差
        # 《pytorch学习：loss为什么要加item()》https://blog.csdn.net/github_38148039/article/details/107144632
        running_loss += loss.item()
        if i % 2000 == 1999:  # 每2000个mini-batches（即2000个batchsize次）打印一次，然后归零running_loss
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

        # 可视化训练过程
        with torch.no_grad():

            if i % 100 == 0:
                # loss_data.append(loss.cpu())  # 纵轴的list（loss）# 可以
                # times.append(i + 1 + epoch * 12000)  # 横轴的list   # 可以
                loss_data.append(loss.cpu().tolist())
                times.append(times[len(times) - 1] + 1)     # 行了！
                # print(times)
                # print(loss_data)
                plt.cla()
                # plt.plot(loss_data, 'r-', lw=1)  # 直接输入y轴坐标，不输入x轴坐标是可以的
                plt.plot(times, loss_data, 'r-', lw=1)
                plt.ylabel('Loss')
                plt.title('loss=%.4f step=%d' % (loss.cpu(), i))
                plt.pause(0.1)



print('Finished Training')

# 创建测试集的迭代器
dataiter = iter(testloader)
# 读取测试集中的前四张图片
images, labels = dataiter.next()
# 将数据放置到GPU上
images, labels = images.to(device), labels.to(device)

# 显示前面读取出来的四张图片和其所属的分类的名字（label）
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

# 传入神经网络得到预测结果
outputs = net(images)

# 输入的第一个参数是softmax输出的一个tensor，这里就是outputs所存储的内容；第二个参数是max函数索引的维度，0是取每列的最大值，1是每行的最大值
# 返回的第一个参数是预测出的实际概率，由于我们不需要得知实际的概率，所以在返回的第一个参数填入_不读取，第二个返回是概率的最大值的索引，存在predicted中
# 《torch.max()使用详解》https://www.jianshu.com/p/3ed11362b54f
_, predicted = torch.max(outputs, 1)

# 打印四张图片预测的分类
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

# 计数器初始化置零
correct = 0
total = 0
# 计算总准确率
# 被该语句包裹起来的代码将不会被跟踪梯度，如果测试集进行的运算被跟踪进度可能会导致显存爆炸
with torch.no_grad():
    for data in testloader:
        # 读取数据（batchsize个数据，这里为4张图片）
        images, labels = data
        # 将数据放置到GPU上
        images, labels = images.to(device), labels.to(device)
        # 得到神经网络的输出
        outputs = net(images)
        # 返回每行概率最大值的索引
        _, predicted = torch.max(outputs.data, 1)
        # labels.size(0)指batchsize的值，这里batchsize=4
        total += labels.size(0)
        # predicted == labels对predicted和labels中的每一项判断是否相等
        # (predicted == labels).sum()返回一个tensor，tensor中是判断为真的数量，比如有一项是相同的，则返回tensor(1)
        # 如果有一项是相同的，(predicted == labels).sum().item()返回1
        # correct在这里即为4张图片中预测正确的数量（这里计算的是总概率）
        correct += (predicted == labels).sum().item()

# 输出神经网络在测试集上的准确率
print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
# 计算每个分类的准确率
# 被该语句包裹起来的代码将不会被跟踪梯度
with torch.no_grad():
    for data in testloader:
        # 读取数据（batchsize个数据，这里为4张图片）
        images, labels = data
        # 将数据放置到GPU上
        images, labels = images.to(device), labels.to(device)
        # 得到神经网络的输出
        outputs = net(images)
        # 返回每行概率最大值的索引
        _, predicted = torch.max(outputs, 1)
        # squeeze()用于去除维数为1的维度，比如1行3列矩阵就会去掉行这个维度，变成第一维含有3个元素
        c = (predicted == labels).squeeze()
        for i in range(4):
            # label存储当前图像所属分类的索引号
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
# 输出神经网络在测试集上的每个分类准确率
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
