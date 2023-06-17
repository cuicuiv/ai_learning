import  torch
from torch import nn
from d2l import torch as d2l

#定义了一个名为vgg_block的函数来实现一个VGG块

#卷积层的数量num_convs、输入通道的数量in_channels 和输出通道的数量out_channels.
def vgg_block(num_convs,in_channels,out_channles):
    layers = []
    #使用循环可以轻松实现不同卷积层数的VGG块
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels,out_channles,kernel_size=3,padding=1))
        layers.append(nn.ReLU())
        #in_channels被更新为out_channels的目的是确保每个卷积层的输入通道数与上一层的输出通道数一致
        in_channels=out_channles
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    #通过使用*layers，我们可以将列表中的每个元素作为独立的参数传递给nn.Sequential函数
    return nn.Sequential(*layers)

#超参数变量conv_arch。该变量指定了每个VGG块里卷积层个数和输出通道数
#原始VGG网络有5个卷积块，其中前两个块各有一个卷积层，后三个块各包含两个卷积层。
#第一个模块有64个输出通道，每个后续模块将输出通道数量翻倍，直到该数字达到512。由于该网络使用8个卷积层和3个全连接层，因此它通常被称为VGG-11
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

#VGG-11实现
def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    #卷积部分
    for (num_convs,out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs,in_channels,out_channels))
        #确保每个卷积层的输入通道数与上一层的输出通道数一致
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks,nn.Flatten(),
        #全连接层部分,3层
        nn.Linear(out_channels*7*7,4096),nn.ReLU(),nn.Dropout(0.5),
        nn.Linear(4096,4096),nn.ReLU(),nn.Dropout(0.5),
        nn.Linear(4096,10)
                         )
net = vgg(conv_arch)

#每个块的高度和宽度减半，最终高度和宽度都为7。最后再展平表示，送入全连接层处理
