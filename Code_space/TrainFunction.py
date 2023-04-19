import torch
from myFunction import Data_To_Cuda, View_loss, test_accuracy_img


def Train_Function(Epoch, model, Train_Data, lossFunction, optimizer, step_number, model_save_path, model_save_name,
                   loss_image_save_path):
    """
    训练函数
    :param Epoch: 训练轮数
    :param model: 模型
    :param Train_Data:训练数据
    :param lossFunction:损失函数
    :param optimizer:优化器
    :param step_number: 多少步记录一次数据，画图
    :param model_save_path: 模型保存地址
    :param model_save_name: 模型保存时的名称
    :param loss_image_save_path: 损失图像保存地址
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
                if step % step_number == (step_number - 1):
                    # 记录一个batch的损失
                    Y_axis.append(loss.cpu().tolist())  # 将loss拿到cpu中（cpu无法直接访问GPU内的数据），转换成list。将list追加到Y轴数组中
                    X_axis.append(X_axis[len(X_axis) - 1] + 1)  # 计算X坐标，将其追加到X数组中
                if step % (2 * step_number) == (2 * step_number - 1):  # 画一次图
                    title = ('epoch=%d step=%d loss=%.4f ' % (epoch, step, loss.cpu()))
                    View_loss(X_axis, Y_axis, X_label, Y_label, title, loss_image_save_path)
    torch.save(model.state_dict(), model_save_path + '/' + model_save_name)


def Train_Tets_Function(
        Epoch, model, Train_Data,
        lossFunction, optimizer, model_save_path,
        model_save_name, Test_data, number_test_data,
        loss_image_save_path, correct_save_path
):
    """
    :param Epoch: 训练轮次
    :param model: 训练模型
    :param Train_Data: 训练数据
    :param lossFunction: 损失函数
    :param optimizer: 优化器
    :param model_save_path: 模型保存地址
    :param model_save_name: 模型保存名
    :param Test_data: 测试数据
    :param number_test_data: 测试数据的个数
    :param loss_image_save_path: 损失图像保存地址
    :param correct_save_path: 正确率图像保存地址
    :return:  None
    """
    X_axis = [0]
    Y_axis = [0]
    X_label = 'rate of progress'
    Y_label = 'loss'
    epoch_list = []
    correct_list = []
    for epoch in range(Epoch):
        epoch_list.append(epoch)
        for step, (train_data, train_labels) in enumerate(Train_Data):
            # 数据扔到cuda中
            train_data = Data_To_Cuda(train_data)
            train_labels = Data_To_Cuda(train_labels)

            y = model(train_data)
            loss = lossFunction(y, train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        Y_axis.append(loss.cpu().tolist())
        X_axis.append(X_axis[len(X_axis) - 1] + 1)
        title = ('epoch=%d  loss=%.4f ' % (epoch, loss.cpu()))
        View_loss(X_axis, Y_axis, X_label, Y_label, title, loss_image_save_path)
        test_accuracy_img(epoch_list, Test_data, number_test_data, model, lossFunction, correct_list, correct_save_path)
    torch.save(model.state_dict(), model_save_path + '/' + model_save_name)
