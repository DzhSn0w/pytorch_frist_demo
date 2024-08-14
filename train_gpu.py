import torch
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import Conv2d,MaxPool2d,Flatten,Linear
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import time

#准备数据集
train_data = torchvision.datasets.CIFAR10("./dataset1",True,transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10("./dataset1",False,transform=torchvision.transforms.ToTensor(),
                                         download=True)
#获取数据集长度
train_len = len(train_data)
test_len = len(test_data)
print("训练集长度为{}".format(train_len))
print("测试集长度为{}".format(test_len))

#加载数据集
train_dataloader = DataLoader(train_data,batch_size=64,shuffle=True)
test_dataloader = DataLoader(test_data,batch_size=64,shuffle=True)

if torch.cuda.is_available():
    print("检测到CUDA，使用GPU进行加速训练")
else:
    print("未检测到CUDA，使用CPU进行训练")

#构建网络架构

class Zhen(nn.Module):
    def __init__(self):
        super(Zhen,self).__init__()
        self.conv1 = Conv2d(3,32,5,1,2)
        self.max1 = MaxPool2d(2)
        self.conv2 = Conv2d(32,32,5,1,2)
        self.max2 = MaxPool2d(2)
        self.conv3 = Conv2d(32,64,5,1,2)
        self.max3 = MaxPool2d(2)
        self.fla1 = Flatten() #flatten展平
        self.lin1 = Linear(in_features=1024,out_features=64)
        self.lin2 = Linear(64,10)
    def forward(self,x):
        x = self.conv1(x)
        x = self.max1(x)
        x = self.conv2(x)
        x = self.max2(x)
        x = self.conv3(x)
        x = self.max3(x)
        x = self.fla1(x)
        x = self.lin1(x)
        x = self.lin2(x)
        return x

#网络架构
zhen = Zhen()
#调用CUDA转移到GPU进行训练
if torch.cuda.is_available():
    zhen = zhen.cuda()

#损失函数
loss = nn.CrossEntropyLoss()
#对损失函数进行cuda加速
if torch.cuda.is_available():
    loss = loss.cuda()
#优化器
learing_rate = 0.01
optimizer = torch.optim.SGD(zhen.parameters(),lr=learing_rate)
#tensorboard
sw = SummaryWriter("logs1")

#设置训练网络的一些参数
#记录训练次数
total_train_step = 0
#记录测试次数
total_test_step = 0
#训练轮数
epoch = 10

for i in range(epoch):
    # 记录训练用时
    start_time = time.time()
    print("-------第{}轮训练开始-------".format(i))
    zhen.train()
    for data in train_dataloader:
        imgs,target = data
        #对数据进行cuda加速
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            target = target.cuda()

        output = zhen(imgs)
        losss = loss(output,target)#把预测的输出和真实的target放进去计算损失

    #优化器优化模型
        #梯度清零
        optimizer.zero_grad()
        #反向传播
        losss.backward()
        #开始优化
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数{}，loss:{}".format(total_train_step, losss.item()))
            sw.add_scalar("train_loss",losss,total_train_step)


    #测试步骤
    zhen.eval()
    total_test_loss = 0
    total_test_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets = data
            #对测试数据进行cuda加速
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = zhen(imgs)
            losss = loss(outputs,targets)
            total_test_loss += losss
            accuracy =  (outputs.argmax(1) == targets).sum()
            total_test_accuracy = total_test_accuracy + accuracy

    print("整体测试集上的loss：{}".format(total_test_loss))
    print("整体测试集上的正确率:{}".format(total_test_accuracy/test_len))
    sw.add_scalar("test_loss",total_test_loss,total_test_step)
    sw.add_scalar("test_accuracy",total_test_accuracy/test_len,total_test_step)
    total_test_step += 1


    #保存每epoch的训练结果

    torch.save(zhen,"model_zhen{}.pth".format(i))
    print("模型存储成功")
    end_time = time.time()
    print("第{}轮训练总用时为：{}".format(i,end_time-start_time))
sw.close()