import torchvision
import torchvision.transforms as transformers
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # 在pytharm中 防止出现图片闪退出错
fashion_train = torchvision.datasets.FashionMNIST(root=r'E:\LWW\real_learing\AlexNet\dataset',train=True,download=True,transform=transformers.ToTensor())
fashion_test = torchvision.datasets.FashionMNIST(root=r'E:\LWW\real_learing\AlexNet\dataset',train=False,download=True,transform=transformers.ToTensor())