from torch.utils.data import Dataset,DataLoader
from PIL import Image
import os

#定义数据集类
class MyData(Dataset):
#__init__用于初始化class类

    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        #os.listdir 用于创建一个列表，包含此文件下的所有的文件
        self.img_path = os.listdir(self.path)

#用于获取图片地址,输入idx编号，返回img和label
    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir,self.label_dir,img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img,label
#用于返回此时数据集长度
    def __len__(self):
        return len(self.img_path)

root_dir = "dataset/train"
label_dir = "ants"
ants_dataset = MyData(root_dir,label_dir)

label_dir1= "bees"
bees_dataset = MyData(root_dir,label_dir1)

train_dataset = ants_dataset + bees_dataset

train_dataLoader = DataLoader(train_dataset,)

lens = len(ants_dataset)