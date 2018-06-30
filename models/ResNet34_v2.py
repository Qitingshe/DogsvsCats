from .BasicModule import BasicModule
from torch import nn
from torch.nn import functional as F 

class ResidualBlock(nn.Module):
    '''
    残差块
    实现子module：Residual Block
    '''
    def __init__(self,inchannel,outchannel,stride=1,shortcut=None):
        super(ResidualBlock,self).__init__()
        # 构建左侧输出,v2版本将激活函数提前类，即pre-activation
        self.left=nn.Sequential(
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel,outchannel,3,stride,1,bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel,outchannel,3,1,1,bias=False)
        )
        # 构建右侧输出
        self.right=shortcut

    def forward(self,x):
        # 先计算左边的输出
        out=self.left(x)
        # 对shortcut处理，如果shortcut为None，则直接将原输入添加给输出out，实现论文中实线部分，否则调用_make_layer部分的shortcut实现
        residual=x if self.right is None else self.right(x)
        # 直接将原输入添加给输出out
        out+=residual
        # 输出部分不再使用relu激活
        return out

class ResNet34_v2(BasicModule):
    '''
    实现主module：ResNet34
    ResNet34包含多个layer，每个layer又包含多个residual block
    用子module实现residual block，用_make_layer函数实现layer
    '''
    def __init__(self,num_classes=2):
        super(ResNet34_v2,self).__init__()
        self.model_name='resnet34'

        # 图片预处理
        self.pre=nn.Sequential(
            nn.Conv2d(3,64,7,2,3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,1)
        )

        #重复的layer，分别有3,4,6,3个residual block
        # 除了第一层外，其他每一层都会将通道翻倍，feature map的尺寸减半
        self.layer1=self._make_layer(64,64,3)
        self.layer2=self._make_layer(64,128,4,stride=2)
        self.layer3=self._make_layer(128,256,6,stride=2)
        self.layer4=self._make_layer(256,512,3,stride=2)

        # 分类部分采用全连接，输出为类别数:num_classes
        self.fc=nn.Linear(512,num_classes)

    def _make_layer(self,inchannel,outchannel,block_num,stride=1):
        '''
        构建layer，包含多个residual block
        '''
        # 实现论文中虚线部分的shortcut
        shortcut=nn.Sequential(
            nn.Conv2d(inchannel,outchannel,1,stride,bias=False),
            nn.BatchNorm2d(outchannel)
        )
        # 初始化层
        layers=[]
        # 构建层
        #构建虚线部分的ResidualBlock
        layers.append(ResidualBlock(inchannel,outchannel,stride,shortcut))
        # 构建实线部分的ResidualBlock
        for i in range(1,block_num):
            # 给layer添加residual block
            layers.append(ResidualBlock(outchannel,outchannel))

        return nn.Sequential(*layers)

    # forward函数
    def forward(self,x):
        x=self.pre(x)   

        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        
        x=F.avg_pool2d(x,7)
        x=x.view(x.size(0),-1)
        # 返回输出类别
        return self.fc(x)