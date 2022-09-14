import torch
from torch import nn


# 定义一个网络模型
class MyLeNet5(nn.Module):
    # 初始化网络
    def __init__(self):
        super(MyLeNet5, self).__init__()

        # in_channels：输入通道； out_channels：输出通道；kernel_size：卷积核（池化核）大小；padding：填充量
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)

        # 1、使用Sigmoid函数进行激活
        # self.activate = nn.Sigmoid()

        # 2、使用Relu函数进行激活
        self.activate = torch.nn.ReLU(inplace=False)

        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)

        # 平展
        self.flatten = nn.Flatten()
        self.f6 = nn.Linear(120, 84)
        self.output = nn.Linear(84, 10)

    def forward(self, x):
        x = self.activate(self.c1(x))
        x = self.s2(x)
        x = self.activate(self.c3(x))
        x = self.s4(x)
        x = self.c5(x)
        x = self.flatten(x)
        x = self.f6(x)
        x = self.output(x)
        return x


if __name__ == "__main__":
    x = torch.rand([1, 1, 28, 28])
    model = MyLeNet5()
    y = model(x)