import torch
from net import MyLeNet5
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage

# 数据转为tensor格式
data_transform = transforms.Compose([
    transforms.ToTensor()
])

# 加载测试数据集
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)

# 如果有显卡，可以转GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 调用net里面定义的模型，讲模型数据转到GPU
model = MyLeNet5().to(device)

model.load_state_dict(torch.load("/Users/liupeng/PycharmProjects/LeNet-5/save_model/best_model_sigmoid.pth"))

# 获取结果
classes = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]

# 把tensor转化成图片
show = ToPILImage()

# 进入验证
for i in range(20):
    X, y = test_dataset[i][0], test_dataset[i][1]
    show(X).show()

    X = Variable(torch.unsqueeze(X, dim=0).float(), requires_grad=False)
    with torch.no_grad():
        pred = model(X)

        predicted, actual = classes[torch.argmax(pred[0])], classes[y]

        print(f'predicted:"{predicted}",actual:"{actual}"')
