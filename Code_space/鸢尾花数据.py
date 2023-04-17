import torch.nn as nn
import torch
import torch.utils.data as Data


def getdata():
    from sklearn.datasets import load_iris
    import pandas as pd
    import numpy as np
    train_data = load_iris()
    data = train_data['data']
    labels = train_data['target'].reshape(-1, 1)
    total_data = np.hstack((data, labels))
    np.random.shuffle(total_data)
    train = total_data[0:80, :-1]
    test = total_data[80:, :-1]
    train_label = total_data[0:80, -1].reshape(-1, 1)
    test_label = total_data[80:, -1].reshape(-1, 1)
    return data, labels, train, test, train_label, test_label


# 网络类
class mynet(nn.Module):
    def __init__(self):
        super(mynet, self).__init__()
        self.fc = nn.Sequential(  # 添加神经元以及激活函数
            nn.Linear(4, 20),
            nn.ReLU(),
            nn.Linear(20, 30),
            nn.ReLU(),
            nn.Linear(30, 3)
        )
        self.mse = nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(params=self.parameters(), lr=0.1)

    def forward(self, inputs):
        outputs = self.fc(inputs)
        return outputs

    def train(self, x, label):
        out = self.forward(x)  # 正向传播
        loss = self.mse(out, label)  # 根据正向传播计算损失
        self.optim.zero_grad()  # 梯度清零
        loss.backward()  # 计算梯度
        self.optim.step()  # 应用梯度更新参数

    def test(self, test_):
        return self.fc(test_)


if __name__ == '__main__':
    data, labels, train, test, train_label, test_label = getdata()
    mynet = mynet()
    train_dataset = Data.TensorDataset(torch.from_numpy(train).float(), torch.from_numpy(train_label).long())
    BATCH_SIZE = 10
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    for epoch in range(100):
        for step, (x, y) in enumerate(train_loader):
            y = torch.reshape(y, [BATCH_SIZE])
            mynet.train(x, y)
            if epoch % 20 == 0:
                print('Epoch: ', epoch, '| Step: ', step, '| batch y: ', y.numpy())
    out = mynet.test(torch.from_numpy(data).float())
    prediction = torch.max(out, 1)[1]  # 1返回index  0返回原值
    pred_y = prediction.data.numpy()
    test_y = labels.reshape(1, -1)
    target_y = torch.from_numpy(test_y).long().data.numpy()
    accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
    print("莺尾花预测准确率", accuracy)


