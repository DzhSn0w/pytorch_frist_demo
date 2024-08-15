import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear


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

if __name__ == "__main__":
    zhen = Zhen()
    input = torch.ones((64,3,32,32))
    output = zhen(input)
    print(output.shape)