import torch
from torch import nn
from net import MyLeNet5
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import os

# 数据转为tensor格式
data_transform = transforms.Compose([
    transforms.ToTensor()
])

# 加载训练数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=data_transform, download=True)
# 16张图片一组，将所有数据分成若干组，并且打乱数据集
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

# 加载测试数据集
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)

# 如果有显卡，可以转GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 调用net里面定义的模型，讲模型数据转到GPU
model = MyLeNet5().to(device)

# 定义一个损失函数（交叉熵损失）
loss_fn = nn.CrossEntropyLoss()

# 定义一个优化器 momentum：动量，在SGD（Stochastic Gradient Descent, 随机梯度下降）
# 的基础上加一个动量，如果当前收敛效果好，就可以加速收敛，如果不好，则减慢它的步伐
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

# 学习率每隔10轮，变为原来的0.1
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# 定义训练函数
def train(dataloader, model, loss_fn, optimizer):
    loss, current, n = 0.0, 0.0, 0
    # 从数据加载器中获取批次个数，图像数据和对应标签
    for batch, (X, y) in enumerate(dataloader):
        # 正向传播
        X, y = X.to(device), y.to(device)
        # 将图像数据通过训练模型
        output = model(X)
        # 计算预测值和真实值的误差大小
        cur_loss = loss_fn(output, y)
        _, pred = torch.max(output, axis=1)

        # 本轮精度  output.shape[0] 张量行数,这里是16
        cur_acc = torch.sum(y == pred) / output.shape[0]

        # 优化器梯度清零
        optimizer.zero_grad()
        # 反向传播
        # 就是将损失loss 向输入侧进行反向传播，同时对于需要进行梯度计算的所有变量 [公式] (requires_grad=True)，
        # 计算梯度，并将其累积到梯度 [公式] 中备用，即[公式] :x.grad = x.grad + d/dx loss
        cur_loss.backward()
        # 值更新
        # 是优化器对 [公式] 的值进行更新，以随机梯度下降SGD为例：学习率(learning rate, lr)来控制步幅，
        # 即[公式] : x = x - lr * x.grad (减号是由于要沿着梯度的反方向调整变量值以减少Cost)
        optimizer.step()

        loss += cur_loss.item()
        current += cur_acc.item()
        n += 1

    print("train_loss" + str(loss / n))
    print("train_acc" + str(current / n))


def val(dataloader, model, loss_fn):
    # 模型验证模式
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    # 不进行梯度计算
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            # 正向传播
            X, y = X.to(device), y.to(device)
            output = model(X)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]

            loss += cur_loss.item()
            current += cur_acc.item()
            n += 1

    print("val_loss" + str(loss / n))
    print("val_acc" + str(current / n))

    return current / n


# 开始训练
epoch = 10
min_acc = 0
for t in range(epoch):
    print(f'epoch{t + 1}\n------------------')
    train(train_dataloader, model, loss_fn, optimizer)
    a = val(test_dataloader, model, loss_fn)
    # 保存最好的模型权重
    if a > min_acc:
        folder = 'save_model'
        if not os.path.exists(folder):
            os.mkdir('save_model')
        min_acc = a
        print('save best model')
        torch.save(model.state_dict(), 'save_model/best_model_relu.pth')
print('Done!')
