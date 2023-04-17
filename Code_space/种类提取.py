import glob

path = '/home/helei/PycharmProjects/My_DeepLearning/Data_space/CelebDataProcessed/*'
all_species_path = glob.glob(path)
species = []
for name in all_species_path:
    s = name.split('/')[-1]
    species.append(s)

if __name__ == '__main__':  # 解决windows下的报错，因为使用了num_work,具体为啥我也不太清楚
    for epoch in range(epoch_num):
        net.train()  # 表明当前网络为训练的过程 train BN dropout
        # 如果在网络层定义了Batchnorm层则需要用net.train
        # 如果在网络层定义了dropout层则需要用net.eval()
        for i, data in enumerate(train_loader):
            inputs, labels = data
            # inputs, labels = inputs.to(device), labels.to(device) #转到GPU训练

            outputs = net(inputs)
            loss = loss_func(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_size = inputs.size(0)
            _, pred = torch.max(outputs.data, dim=1)
            correct = pred.eq(labels.data).sum()
            print("train step", i, "loss is:", loss.item(), "mini-batch correct is:", 1.0 * correct / batch_size)

        # 用来保存训练后的模型
        if not os.path.exists("models"):
            os.mkdir("models")
        torch.save(net.state_dict(), "models/{}.pth".format(epoch + 1))
        scheduler.step()  # 更新学习率

        sum_loss = 0
        sum_correct = 0

        for i, data in enumerate(test_loader):
            net.eval()
            inputs, labels = data
            # inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            loss = loss_func(outputs, labels)  # 测试集不再进行反向传播
            _, pred = torch.max(outputs.data, dim=1)
            correct = pred.eq(labels.data).sum()

            sum_loss += loss.item()
            sum_correct += correct.item()

            im = torchvision.utils.make_grid(inputs)

        test_loss = sum_loss * 1.0 / len(test_loader)
        test_correct = sum_correct * 1.0 / len(test_loader) / batch_size

        print("epoch", epoch + 1, "loss is:", test_loss, "mini-batch correct is:", test_correct)


