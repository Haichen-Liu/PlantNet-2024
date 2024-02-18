# encoding:utf-8
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# 加载模型
model = models.resnet18(weights=None)  # 不加载预训练权重
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 5)  # 假设有5个类别
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# 图像预处理
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 将图片调整为模型所需的尺寸
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# 预测图像
def predict_image(image_path, model):
    image_tensor = preprocess_image(image_path)
    outputs = model(image_tensor)
    _, predicted = torch.max(outputs.data, 1)
    return predicted.item()  # 返回预测的类别索引

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

def display_prediction_in_window(image_path, prediction):
    class_names = ['aeonium', 'aloe_vera', 'calendula_flower', 'mushroom', 'prickly_pear_cactus']
    predicted_class_name = class_names[prediction]

    # 创建窗口
    window = tk.Tk()
    window.title("TUTE Plant species recognition")  # 在这里改窗口名字

    # 加载图像
    image = Image.open(image_path)
    image.thumbnail((400, 400))  # 调整图像大小
    photo = ImageTk.PhotoImage(image)

    # 创建图像标签
    image_label = ttk.Label(window, image=photo)
    image_label.image = photo  # 防止垃圾回收
    image_label.pack(pady=10)

    # 创建文本标签
    text_label = ttk.Label(window, text=f"预测结果: {predicted_class_name}", font=("Helvetica", 16))
    text_label.pack(pady=10)

    window.mainloop()

# 测试图像路径
test_image_path = "./dataset/test/aeonium/aeonium_1.jpg"

# 进行预测并显示结果
predicted_class = predict_image(test_image_path, model)
display_prediction_in_window(test_image_path, predicted_class)


