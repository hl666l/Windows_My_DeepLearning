import glob
import torch
from torch.utils import data
import myFunction
from myFunction import myDataset_class as MC
from Model import Model as MD

"""

scale: 训练集和测试集比例
epoch： 训练多少轮
step_number： 训练步数
model_path: 模型保存的地址
model_name: 模型保存时的名称
img_size: 训练时需要的图片尺寸
img_path:匹配路径
BATCH_SIZE：每个batch的大小

"""
scale = 0.9
epoch = 50
step_number = 200
mode_path = '/home/helei/PycharmProjects/My_DeepLearning/Model_space'
model_name = 'model.pk'
img_size = 208
img_path = '/home/helei/PycharmProjects/My_DeepLearning/Data_space/AugmentedAlzheimerDataset/*/*.jpg'
BATCH_SIZE = 400
# 使用glob方法来获取数据图片的所有路径
all_imgs_path = glob.glob(img_path)

species = ['MildTemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

all_labels = myFunction.All_Img_Label(all_imgs_path, species)
# 对数据进行转换处理
transform = myFunction.my_transform(img_size)
# 划分测试集和训练集
train_imgs, train_labels, test_imgs, test_labels, s = myFunction.Partition_Dataset(all_imgs_path, all_labels, scale)

train_ds = MC(train_imgs, train_labels, transform)
test_ds = MC(test_imgs, test_labels, transform)

train_dl = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True)
test_dl = data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)

test_data = iter(test_dl)
(test_inputs, test_labels) = next(test_data)
test_labels = test_labels.type(torch.LongTensor)

model = myFunction.Model_To_Cuda(MD)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss = torch.nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss = loss.cuda()

if __name__ == '__main__':
    myFunction.Train_Function(epoch, model, train_dl, loss, optimizer, step_number, model_path=mode_path,
                              model_save_name=model_name)
