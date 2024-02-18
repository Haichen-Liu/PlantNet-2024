# encoding:utf-8
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.models import ResNet18_Weights
import matplotlib.pyplot as plt
# 设定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 将图片统一调整为ResNet所需的224x224大小
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载训练集和测试集
train_dataset = torchvision.datasets.ImageFolder(root='./dataset/train', transform=transform)
test_dataset = torchvision.datasets.ImageFolder(root='./dataset/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 加载预训练的ResNet模型
weights = ResNet18_Weights.IMAGENET1K_V1
model = models.resnet18(weights=weights)

# 替换最后的全连接层以适应植物分类任务
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)  # 假设有5个类别

# 将模型移动到设定的设备上
model = model.to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 用于保存最佳模型的变量
best_accuracy = 0.0
best_model_path = "best_model.pth"

# 初始化列表来存储准确率和损失
accuracy_list = []
loss_list = []

# 训练模型
num_epochs = 10  # 训练轮数
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loss_list.append(loss.item())  # 添加损失到列表

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

    # 测试模型
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        accuracy_list.append(accuracy)  # 添加准确率到列表
        print(f'Epoch [{epoch+1}/{num_epochs}], 测试集上的准确率: {accuracy}%')

        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"新的最佳模型已保存：{best_model_path}")

# 绘制损失曲线图
plt.figure(figsize=(10, 5))
plt.plot(loss_list, label='Training Loss')
plt.title('Loss Curve')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.savefig('./loss.png')

# 绘制准确率曲线图
plt.figure(figsize=(10, 5))
plt.plot(accuracy_list, label='Test Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.savefig('./acc.png')
