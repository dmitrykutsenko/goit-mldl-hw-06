import os
import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# 2. Підготовка даних (завантаження, train/val/test, DataLoader)

# 2.1. Структура датасету Intel Image Classification

# 2.2. Трансформації (preprocessing + аугментація)
# Для train: аугментація + нормалізація
# Для val/test: тільки базові перетворення + нормалізація
# Нормалізація — під ImageNet (щоб ResNet18 працював коректно)

data_dir = "./data/raw/intel-image-classification/"

img_size = 150
batch_size = 64

train_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 2.3. Завантаження train/test через ImageFolder

train_dir = os.path.join(data_dir, "seg_train")
test_dir  = os.path.join(data_dir, "seg_test")

print("\nTrain dir exists:", os.path.exists(train_dir))
print("\nTest dir exists:", os.path.exists(test_dir))

print("\nTrain classes:", os.listdir(train_dir))
print("\nTest classes:", os.listdir(test_dir))

full_train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
test_dataset       = datasets.ImageFolder(root=test_dir, transform=val_test_transforms)

class_names = full_train_dataset.classes
num_classes = len(class_names)
print("\nКласи:", class_names, " | num_classes =", num_classes)

# 2.4. Розділення train → train + validation
# Наприклад, 80% train / 20% val:

val_ratio = 0.2
train_size = int((1 - val_ratio) * len(full_train_dataset))
val_size   = len(full_train_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

# ВАЖЛИВО: для val потрібні інші трансформації (без аугментації)
val_dataset.dataset.transform = val_test_transforms

# 2.5. DataLoader-и

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=4)

