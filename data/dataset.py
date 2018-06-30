import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import random


class DogCat(data.Dataset):
    def __init__(self, root, transforms=None, train=True, test=False):
        # root :image rootdir
        self.test = test

        # get the dataset,img为路径下的图片名称，imgs为图片的路径
        imgs = [os.path.join(root, img) for img in os.listdir(root)]

        if self.test:
            # sorted：对所有可迭代的对象进行排序操作，返回一个新的list
            imgs = sorted(imgs, key=lambda x: int(
                x.split('.')[-2].split('/')[-1]))
        else:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))

        # get sample number of the dataset
        imgs_num = len(imgs)

        # random.shuffle(imgs)

        if self.test:
            self.imgs = imgs
        # 分割train为训练集和验证集，比例为7：3
        elif train:
            # 前70%的数据作为trainset(注意中括号中的冒号)
            # self.imgs=imgs[:int(0.7*imgs_num)]
            self.imgs = imgs[:int(imgs_num)]
        else:
            # 后30%的数据作为valset
            self.imgs = imgs[int(0.7*imgs_num):]
        # 数据预处理
        if transforms is None:
            # 标准化至[-1,1],同时规定均值和标准差
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            # 是测试集或验证集(不是训练集)
            if self.test or not train:

                # 使用Compose将对数据的处理操作拼接起来，和nn.Sequential相似，该操作以对象的形式存在，需要调用他的__call__方法
                self.transforms = T.Compose([
                    T.Scale(224),       # 缩放图片，保持长宽比，最短边为224像素
                    T.CenterCrop(224),     # 从图片中裁剪出224×224的图片
                    T.ToTensor(),       # 将图片转成Tensor，归一化至[0,1]
                    normalize           # 标准化至[-1,1]，规定均值和方差
                ])
            # 对训练集处理
            else:
                self.transforms = T.Compose([
                    T.Scale(256),
                    # CenterCrop、RandomCrop、RandomSizedCrop：裁剪图片
                    T.RandomSizedCrop(224),
                    T.RandomHorizontalFlip(),   # 随机水平翻转，0.5概率翻转
                    T.ToTensor(),
                    normalize
                ])

    def __getitem__(self, index):
        """
        将文件读取等费时操作放在__getitem__函数中，利用多线程加速。一次调用该函数，只返回一个样本，在多进程中会并行地调用__getitem__函数，由此实现加速
        如果需要使用batch，打乱以及并行加速等操作，需要继续使用PyTorch的DataLoader
        DataLoader是一个可迭代对象，所以我们可以在for循环中使用它
        """
        img_path = self.imgs[index]
        # 获取图片的label
        # 如果是测试集数据
        if self.test:
            # 将img_path 先以'.'分割开，取倒数第二个字符，再以'/' 分割开，取倒数第一个字符
            # 得到图片的序号
            label = int(self.imgs[index].split('.')[-2].split('/')[-1])

        # 否则如果是训练集数据或验证集数据
        else:
            # 将img_path 以 '/' 符号分割开，取倒数第一个字符，得到类似于'cat.2.jpg'这样的字符
            # 如果’dog‘在该取出的字符中，则label取1,否则取0
            label = 0 if 'dog' in img_path.split('/')[-1] else 1
        # 获取数据
        data = Image.open(img_path)
        # 数据预处理
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)
