import glob
import torch
from torch.utils import data
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms


def get_species(path):
    all_species_path = glob.glob(path)
    species = []
    for name in all_species_path:
        s = name.split('/')[-1]
        species.append(s)
    return species


def my_transform(img_size):
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()  # 第二步转换，作用：第一转换成Tensor，第二将图片取值范围转换成0-1之间，第三会将channel置前
        ]
    )
    return transform


def All_Img_Label(all_imgs_path, species):
    """
    制作所有图片的标签
    :param all_imgs_path: 所有图片的地址
    :param species: 图片的种类名称（就是每个类表所在文件夹的名称）
    :return: 返回所有图片的标签
    """
    all_Img_label = []
    for img in all_imgs_path:
        for i, c in enumerate(species):
            if c in img:
                all_Img_label.append(i)
    return all_Img_label


def Partition_Dataset(all_imgs_path, all_imgs_label, scale):
    """

    :param all_imgs_path: 所有图片的路径 list
    :param all_imgs_label: list
    :param scale: 训练集和测试集的比例
    :return: 返回一个元组
    """
    index = np.random.permutation(len(all_imgs_path))
    all_img_path = np.array(all_imgs_path)[index]
    all_img_label = np.array(all_imgs_label)[index]
    s = int(len(all_img_path) * scale)
    st = len(all_img_path) - s

    train_img = all_img_path[:s]
    train_label = all_img_label[:s]

    test_img = all_img_path[s:]
    test_label = all_img_label[s:]

    return train_img, train_label, test_img, test_label, st


def Dataset_transform():
    pass


class myDataset_class(data.Dataset):
    """
    data.Dataset 是抽象类，子类继承后里面有三个函数需要我们具体实现

    为什么要写这样一个类？
    答：后面我们要将我们的数据交给data.DataLoader()，
    他能帮我们把数据按照一个batchsize的大小分好组，
    通过训练时的迭代器扔进训练模型中。
    而这个函数需要的是一个data.Dataset类型的数据
    """

    def __init__(self, img_path, label, transform):
        """
        :param img_path: 所有图片的路径list
        :param label: 所有图片的标签list
        :param transform: 对图片转换的操作
        """
        self.data = img_path
        self.label = label
        self.transform = transform

    def __getitem__(self, index):  # 根据给出的索引进行切片，并对其进行数据处理转换成Tensor，返回成Tensor
        """
        :param index: 默认
        :return: 返回制作好的数据，标签
        """
        img = self.data[index]
        label = self.label[index]
        pill_img = Image.open(img)
        data = self.transform(pill_img)
        return data, label

    def __len__(self):
        """
        :return: 返回数据的长度
        """
        return len(self.data)


def Model_To_Cuda(mo):
    """

    :param Model: 模型
    :return:
    """
    model = mo()
    if torch.cuda.is_available():
        model = model.cuda()
    else:
        print("No device use")
    return model


def Out_Put_loss(epoch, step, Step_number, loss):
    """
    :param epoch:当前轮次
    :param step:当前步
    :param Step_number: 一轮多少个patch输出一次
    :param loss:损失数据
    :return:无

    """
    if step % Step_number == 0:
        print("Epoch: ", epoch, '|step:', step,
              '|train loss: %.4f' % loss.item())


def Data_To_Cuda(data):
    """

    :param data: 要传入cuda中的数据
    :return: 返回传入后的数据
    """
    my_data = data
    if torch.cuda.is_available():
        my_data = my_data.cuda()
    else:
        print('No device use')
    return my_data


def View_loss(X_axis, Y_axis, X_label, Y_label, title):
    """

    :param X_axis: x轴
    :param Y_axis: y轴
    :param X_label: x 标签
    :param Y_label: y 标签
    :param title: 图标题
    :return: 无
    """
    """
    
    plt.cla()  # 清除轴， 即当前图形中当前活动的轴，使其它轴保持不变
    plt.plot(X_axis, Y_axis, 'r-', lw=2)  # 画表函数 'r-' r是红色，‘-’是线条的形状 lw是线条的宽度
    plt.ylabel('loss')  # x轴表示的含义
    plt.xlabel('rate of progress ')  # y轴的含义
    plt.title('epoch=%d step=%d loss=%.4f ' % (epoch, step, loss.cpu()))  # 图表的含义（标题）
    plt.pause(0.1)  # 用于暂停0.1秒
    
    """
    plt.cla()
    plt.plot(X_axis, Y_axis, 'r-', lw=1)
    plt.xlabel(X_label)
    plt.ylabel(Y_label)
    plt.title(title)
    plt.savefig('/home/helei/PycharmProjects/My_DeepLearning/Image/' + title + '.png')
    plt.pause(0.2)


def Train_Function(Epoch, model, Train_Data, lossFunction, optimizer, Step_number, model_path, model_save_name):
    """
    训练函数
    :param model_path: 模型保存地址
    :param Epoch: 训练轮数
    :param model: 模型
    :param Train_Data:训练数据
    :param lossFunction:损失函数
    :param optimizer:优化器
    :param Step_number:一轮中多少步输出一次
    :param model_save_name: 模型保存时的名称
    :return: 无返回
    """
    X_axis = [0]
    Y_axis = [0]
    X_label = 'rate of progress'
    Y_label = 'loss'
    for epoch in range(Epoch):
        for step, (train_data, train_labels) in enumerate(Train_Data):
            # 数据扔到cuda中
            train_data = Data_To_Cuda(train_data)
            train_labels = Data_To_Cuda(train_labels)

            y = model(train_data)
            loss = lossFunction(y, train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                if step % 10 == 9:
                    # 每100个batch记录一个batch的损失
                    Y_axis.append(loss.cpu().tolist())  # 将loss拿到cpu中（cpu无法直接访问GPU内的数据），转换成list。将list追加到Y轴数组中
                    X_axis.append(X_axis[len(X_axis) - 1] + 1)  # 计算X坐标，将其追加到X数组中

                if step % 30 == 29:  # step 每500个batch 画一次图
                    title = ('epoch=%d step=%d loss=%.4f ' % (epoch, step, loss.cpu()))
                    View_loss(X_axis, Y_axis, X_label, Y_label, title)
            # Out_Put_loss(epoch, step, Step_number, loss)
    torch.save(model.state_dict(), model_path + '/' + model_save_name)


def test_accuracy(epoch, test_data, number_data, model, lossFunction):
    model = model
    sum_loss = 0.0
    sum_correct = 0.0
    for step, (data, label) in enumerate(test_data):
        data = Data_To_Cuda(data)
        label = Data_To_Cuda(label)
        y = model(data)
        loss = lossFunction(y, label)
        _, pred = torch.max(y.data, dim=1)
        correct = pred.eq(label.data).sum()
        sum_loss += loss.item()
        sum_correct += correct.item()
    test_loss = sum_loss * 1.0 / number_data
    test_correct = sum_correct * 1.0 / number_data
    print("epoch is:", epoch, "loss is:%.4f" % test_loss, "correct is:%.2f" % test_correct)


def Train_Tets_Function(Epoch, model, Train_Data, lossFunction, optimizer, model_path, model_save_name, Test_data,
                        number_test_data):
    """
    训练函数
    :param model_path: 模型保存地址
    :param Epoch: 训练轮数
    :param model: 模型
    :param Train_Data:训练数据
    :param lossFunction:损失函数
    :param optimizer:优化器
    :param Step_number:一轮中多少步输出一次
    :param model_save_name: 模型保存时的名称
    :return: 无返回
    """
    X_axis = [0]
    Y_axis = [0]
    X_label = 'rate of progress'
    Y_label = 'loss'
    for epoch in range(Epoch):
        for step, (train_data, train_labels) in enumerate(Train_Data):
            # 数据扔到cuda中
            train_data = Data_To_Cuda(train_data)
            train_labels = Data_To_Cuda(train_labels)

            y = model(train_data)
            loss = lossFunction(y, train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # with torch.no_grad():
            #     if step % 10 == 9:
            #         # 每100个batch记录一个batch的损失
            #         Y_axis.append(loss.cpu().tolist())  # 将loss拿到cpu中（cpu无法直接访问GPU内的数据），转换成list。将list追加到Y轴数组中
            #         X_axis.append(X_axis[len(X_axis) - 1] + 1)  # 计算X坐标，将其追加到X数组中
            #
            #     if step % 20 == 19:  # step 每500个batch 画一次图
            #         title = ('epoch=%d step=%d loss=%.4f ' % (epoch, step, loss.cpu()))
            #         View_loss(X_axis, Y_axis, X_label, Y_label, title)
        test_accuracy(epoch, Test_data, number_test_data, model, lossFunction)
        # Out_Put_loss(epoch, step, Step_number, loss)
    torch.save(model.state_dict(), model_path + '/' + model_save_name)
