import torch
from torch import nn
from d2l import torch as d2l

#模型架构
net = nn.Sequential(
    #二维卷积--第一层 使用96个11x11的卷积核（kernel）和4个步幅（stride），输出通道数为96，padding为1,输出尺寸为((224+2x1-11)/4)+1=55
    nn.Conv2d(1,96,kernel_size=11,stride=4,padding=1),nn.ReLU(),
    #最大池化层：使用3x3的卷积核和2个步幅，输出尺寸为((55-3)/2)+1=27，即27x27x96的特征图
    nn.MaxPool2d(kernel_size=3,stride=2),
    #输入尺寸为27x27x96，使用256个5x5的卷积核和1个padding，输出通道数为256。输出尺寸为((27+2x1-5)/1)+1=27，即27x27x256的特征图。ReLU层：不改变特征图尺寸。
    nn.Conv2d(96,256,kernel_size=5,padding=2),nn.ReLU(),
    #最大池化层：使用3x3的卷积核和2个步幅，输出尺寸为((27-3)/2)+1=13，即13x13x256的特征图。
    nn.MaxPool2d(kernel_size=3,stride=2),

    #第三层卷积层：输入尺寸为13x13x256，使用3x3的卷积核和1个padding，输出通道数为384。输出尺寸为((13+2x1-3)/1)+1=13，即13x13x384的特征图。
    nn.Conv2d(256,384,kernel_size=3,padding=1),nn.ReLU(),
    #第四层卷积层：输入尺寸为13x13x384，使用3x3的卷积核和1个padding，输出通道数为384。输出尺寸为((13+2x1-3)/1)+1=13，即13x13x384的特征图。
    nn.Conv2d(384,384,kernel_size=3,padding=1),nn.ReLU(),
    #第五层卷积层：输入尺寸为13x13x384，使用3x3的卷积核和1个padding，输出通道数为256。输出尺寸为((13+2x1-3)/1)+1=13，即13x13x256的特征图。
    nn.Conv2d(384,256,kernel_size=3,padding=1),nn.ReLU(),
    #最大池化层：使用3x3的卷积核和2个步幅，输出尺寸为((13-3)/2)+1=6，即6x6x256的特征图
    nn.MaxPool2d(kernel_size=3,stride=2),
    #扁平化层：将6x6x256的特征图转换为1维向量，即9216维
    nn.Flatten(),

    #全连接层使用dropout层来减轻过拟合
    #第一层全连接层：输入大小为9216，输出大小为4096。
    nn.Linear(9216,4096),nn.ReLU(),
    nn.Dropout(p=0.5),
    # 第二次全连接层：输入4096，输出4096。
    nn.Linear(4096,4096),nn.ReLU(),
    nn.Dropout(p=0.5),
    # 第三次全连接层：输入4096，输出1000（1000类） 由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
    nn.Linear(4096,10)

)
#构造一个高度和宽度都为224的单通道数据，来观察每一层输出的形状
X = torch.randn(1,1,244,244)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t',X.shape)

#这里使用的是Fashion-MNIST数据集
batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

lr, num_epochs = 0.01, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())