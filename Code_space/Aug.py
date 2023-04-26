# 导包
import glob
import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms

# 使用glob方法来获取数据图片的所有路径
all_imgs_path = glob.glob(r'Data_space/AugmentedAlzheimerDataset/*/*.jpg')

species = ['MildTemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

all_labels = []
# 对所有图片路径进行迭代
for img in all_imgs_path:
    # 区分出每个img，应该属于什么类别
    for i, c in enumerate(species):
        if c in img:
            all_labels.append(i)
# print(all_labels)  # 得到所有标签

# 对数据进行转换处理
transform = transforms.Compose([
    transforms.Resize((208, 208)),  # 做的第一步转换
    transforms.ToTensor()  # 第二步转换，作用：第一转换成Tensor，第二将图片取值范围转换成0-1之间，第三会将channel置前
])


class Mydatasetpro(data.Dataset):
    # 类初始化
    def __init__(self, img_paths, labels, transform):
        self.imgs = img_paths
        self.labels = labels
        self.transforms = transform

    # 进行切片
    def __getitem__(self, index):  # 根据给出的索引进行切片，并对其进行数据处理转换成Tensor，返回成Tensor
        img = self.imgs[index]
        label = self.labels[index]
        pil_img = Image.open(img)  # pip install pillow
        data = self.transforms(pil_img)
        return data, label

    # 返回长度
    def __len__(self):
        return len(self.imgs)


BATCH_SIZE = 20

# 划分测试集和训练集
index = np.random.permutation(len(all_imgs_path))

all_imgs_path = np.array(all_imgs_path)[index]
all_labels = np.array(all_labels)[index]
s = int(len(all_imgs_path) * 0.9)

train_imgs = all_imgs_path[:s]
train_labels = all_labels[:s]
test_imgs = all_imgs_path[s:]
test_labels = all_labels[s:]

train_ds = Mydatasetpro(train_imgs, train_labels, transform)
test_ds = Mydatasetpro(test_imgs, test_labels, transform)
train_dl = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_dl = data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)

test_data = iter(test_dl)
(test_inputs, test_labels) = next(test_data)
test_labels = test_labels.type(torch.LongTensor)


# 网络框架
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,
                            out_channels=6,
                            kernel_size=5,
                            padding=2,
                            stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(in_channels=6,
                            out_channels=16,
                            kernel_size=5,
                            stride=1,
                            padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(in_channels=16,
                            out_channels=8,
                            kernel_size=5,
                            stride=1,
                            padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)  # 8x26x26
        )
        self.linear1 = torch.nn.Linear(5408, 1352)
        self.linear2 = torch.nn.Linear(1352, 338)
        self.linear3 = torch.nn.Linear(338, 120)
        self.linear4 = torch.nn.Linear(120, 4)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        return x


model = Model()
if torch.cuda.is_available():
    model = model.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
loss = torch.nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss = loss.cuda()
if __name__ == '__main__':
    for epoch in range(30):
        for step, (data, labels) in enumerate(train_dl):
            if torch.cuda.is_available():
                data = data.cuda()
            y = model(data)
            labels = labels.type(torch.LongTensor)
            if torch.cuda.is_available():
                labels = labels.cuda()

            loss_y = loss(y, labels)
            optimizer.zero_grad()
            loss_y.backward()
            optimizer.step()
            if step % 200 == 0:
                if torch.cuda.is_available():
                    test_inputs = test_inputs.cuda()
                test_output = model(test_inputs)
                test_output = test_output.cpu()
                pred_y = torch.max(test_output, 1)[1].data.numpy()
                test_labels = test_labels.cpu()
                loss_y = loss_y.cpu()
                accuracy = float((pred_y == test_labels.data.numpy()).astype(int).sum()) / float(test_labels.size(0))
                print("Epoch:", epoch, '| train loss: %.4f' % loss_y.data.numpy(), '|test accuracy: %.2f' % accuracy)
torch.save(model.state_dict(), 'model.pk2')
