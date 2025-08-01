import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from torch import nn
from torch.utils.data import DataLoader
import time

# 定义训练的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="./data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="./data", train=False, transform=torchvision.transforms.ToTensor(),
                                          download=True)

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为： {}".format(train_data_size))
print("测试数据集的长度为： {}".format(test_data_size))

# 利用DataLoader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
import torch.nn as nn

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            # 卷积块1 (3通道 → 32通道)
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),  # 输入: 32×32, 输出: 32×32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 输出: 16×16
            nn.BatchNorm2d(32),  # 标准化32个特征图
            
            # 卷积块2 (32通道 → 32通道)
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),  # 输入: 16×16, 输出: 16×16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 输出: 8×8
            nn.BatchNorm2d(32),  # 标准化32个特征图
            
            # 卷积块3 (32通道 → 64通道)
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),  # 输入: 8×8, 输出: 8×8
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 输出: 4×4
            nn.BatchNorm2d(64),  # 标准化64个特征图
            
            # 卷积块4 (64通道 → 128通道)
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),  # 输入: 4×4, 输出: 4×4
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 输出: 2×2
            nn.BatchNorm2d(128),  # 标准化128个特征图
            
            # 准备全连接层
            nn.Flatten(),  # 将128×2×2展平为512维向量
            
            # 加入Dropout防止过拟合 (50%概率丢弃神经元)
            nn.Dropout(p=0.5),  # 训练时激活，测试时自动关闭
            
            # 全连接层1 (512 → 64)
            nn.Linear(in_features=128 * 2 * 2, out_features=64),  # 512 → 64
            nn.ReLU(),
            
            # 全连接层2 (64 → 10) 输出层
            nn.Linear(in_features=64, out_features=10)  # 10个类别输出
        )

    def forward(self, x):
        x = self.model(x)
        return x
tudui = Tudui()
tudui.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 64

writer = SummaryWriter("logs_train")
start_time = time.time()

for i in range(epoch):
    print("---------第{}轮训练开始---------".format(i + 1))

    # 训练步骤开始
    tudui.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print("使用时间：{}".format(end_time - start_time))
            print("训练次数：{}, loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率： {}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy",total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    torch.save(tudui, "tudui_{}.pth".format(i))
    print("模型已保存")

writer.close()