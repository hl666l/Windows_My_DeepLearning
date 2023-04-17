import pandas as pd
import torch.nn as nn
import torch


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 3),
            nn.Sigmoid(),
            nn.Linear(3, 3),
            nn.Sigmoid(),
            nn.Linear(3, 1),
        )
        self.mls = nn.MSELoss()
        self.opt = torch.optim.Adam(params=self.parameters(), lr=0.001)

    def get_data(self):
        inputs = []
        labels = []
        with open('/home/helei/PycharmProjects/My_DeepLearning/Data_space/iris.csv') as file:
            df = pd.read_csv(file, header=None)
            x = df.iloc[1:, 0:4].values
            y = df.iloc[1:, 4].values
            for i in range(len(x)):
                inputs.append(x[i])
            for j in range(len(y)):
                a = []
                a.append(y[j])
                labels.append(a)

        return inputs, labels

    def forward(self, inputs):
        out = self.fc(inputs)
        return out

    def train(self, x, label):
        out = self.forward(x)
        loss = self.mls(out, label)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def test(self, x):
        return self.fc(x)


if __name__ == '__main__':
    net = MyNet()
    inputs, labels = net.get_data()
    for i in range(1000):
        for index, input in enumerate(inputs):
            # 这里不加.float()会报错，可能是数据格式的问题吧
            input = torch.from_numpy(input).float()
            label = torch.Tensor(labels[index])
            net.train(input, label)
    # 简单测试一下
    c = torch.Tensor([[5.6, 2.7, 4.2, 1.3]])
    print(net.test(c))
