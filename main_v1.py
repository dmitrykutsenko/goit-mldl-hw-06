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


# ------------------------------------------------------------
# 2. Підготовка даних (завантаження, train/val/test, DataLoader)
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 3. Варіант A — Проста CNN з нуля
# ------------------------------------------------------------

# 3.1. Архітектура Custom CNN
# Ціль — досягти ~75–85% accuracy, F1 ~0.73–0.83.
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 150 -> 75

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 75 -> 37

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 37 -> 18

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 18 -> 9
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 9 * 9, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ------------------------------------------------------------
# 4. Варіант B — Transfer Learning (ResNet18)
# ------------------------------------------------------------

# 4.1. Ініціалізація ResNet18
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# 4.2. «Заморожування» шарів (опціонально)
for param in resnet.parameters():
    param.requires_grad = False  # заморозити всі

# Розморозити тільки останній блок (опціонально, для кращого fine-tuning)
for param in resnet.layer4.parameters():
    param.requires_grad = True

# 4.3. Заміна останнього шару під 6 класів
in_features = resnet.fc.in_features
resnet.fc = nn.Linear(in_features, num_classes)
resnet = resnet.to(device)


# ------------------------------------------------------------
# 5. Функція втрат та оптимізатор
# ------------------------------------------------------------

# 5.1. Втрата: CrossEntropyLoss
# Обґрунтування:
#  -  Класифікація зображень з взаємовиключними класами (6 сцен).
#  -  Модель повертає логіти (N,C), а CrossEntropyLoss поєднує LogSoftmax + NLLLoss в одному стабільному виразі.
#  -  Стандарт де-факто для multi-class classification.
criterion = nn.CrossEntropyLoss()

# 5.2. Оптимізатори
#  -  Для SimpleCNN — Adam (швидка збіжність, зручно для експериментів)
#  - Для ResNet18 — SGD з momentum (часто стабільніший для fine-tuning)
simple_cnn = SimpleCNN(num_classes=num_classes).to(device)
optimizer_cnn = optim.Adam(simple_cnn.parameters(), lr=1e-3, weight_decay=1e-4)

optimizer_resnet = optim.SGD(
    filter(lambda p: p.requires_grad, resnet.parameters()),
    lr=1e-3,
    momentum=0.9,
    weight_decay=1e-4
)

scheduler_resnet = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_resnet, mode='min', factor=0.5, patience=3
)


# ------------------------------------------------------------
#6. Цикл навчання з відстеженням прогресу та збереженням найкращої моделі
# ------------------------------------------------------------

# 6.1. Загальна функція тренування (train + val)
def train_model(model, criterion, optimizer, train_loader, val_loader,
                num_epochs=20, scheduler=None, model_name="model"):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = np.inf
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 30)

        # --- TRAIN ---
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

        epoch_train_loss = running_loss / total
        epoch_train_acc = running_corrects.double().item() / total

        # --- VAL ---
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)
                val_total += labels.size(0)

        epoch_val_loss = val_running_loss / val_total
        epoch_val_acc = val_running_corrects.double().item() / val_total

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)

        if scheduler is not None:
            scheduler.step(epoch_val_loss)

        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")
        print(f"Val   Loss: {epoch_val_loss:.4f} | Val   Acc: {epoch_val_acc:.4f}")
        print(f"Epoch time: {time.time() - start_time:.1f} sec")

        # Збереження найкращої моделі по val_loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, f"best_{model_name}.pth")
            print(f"--> Saved best {model_name} (val_loss={best_val_loss:.4f})")

        print()

    model.load_state_dict(best_model_wts)
    return model, history

# 6.2. Запуск навчання
num_epochs_cnn = 25
simple_cnn, history_cnn = train_model(
    simple_cnn, criterion, optimizer_cnn,
    train_loader, val_loader,
    num_epochs=num_epochs_cnn,
    scheduler=None,
    model_name="simple_cnn"
)

num_epochs_resnet = 20
resnet, history_resnet = train_model(
    resnet, criterion, optimizer_resnet,
    train_loader, val_loader,
    num_epochs=num_epochs_resnet,
    scheduler=scheduler_resnet,
    model_name="resnet18"
)


# ------------------------------------------------------------
#7. Оцінка моделі на тесті + метрики (accuracy, F1)
# ------------------------------------------------------------

# 7.1. Функція оцінки
def evaluate_model(model, data_loader, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    f1_macro = f1_score(all_labels, all_preds, average='macro')

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test F1-macro: {f1_macro:.4f}")
    print("\nClassification report:\n")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    return acc, f1_macro, cm


# 7.2. Оцінка для обох моделей
print("\n=== Simple CNN on TEST ===")
acc_cnn, f1_cnn, cm_cnn = evaluate_model(simple_cnn, test_loader, class_names)

print("\n=== ResNet18 on TEST ===")
acc_resnet, f1_resnet, cm_resnet = evaluate_model(resnet, test_loader, class_names)

#Обґрунтування вибору F1-macro:
#  -  Класи можуть бути неідеально збалансовані.
#  -  accuracy може бути високою навіть при поганій роботі на рідкісних класах.
#  -  macro F1 усереднює F1 по класах, даючи кожному класу однакову вагу — це важливо для аналізу якості по всіх сценах.


# ------------------------------------------------------------
# 8. Візуалізація кривих навчання та аналіз помилок
# ------------------------------------------------------------

# 8.1. Криві навчання (loss + accuracy)
def plot_history(history, title_prefix="Model"):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} - Loss")
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title_prefix} - Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_history(history_cnn, title_prefix="Simple CNN")
plot_history(history_resnet, title_prefix="ResNet18")

# 8.2. Аналіз помилок класифікації
# Ідея: подивитися приклади, де модель помиляється, і зрозуміти, які класи плутаються.

def show_misclassified(model, data_loader, class_names, max_images=20):
    model.eval()
    mis_images = []
    mis_true = []
    mis_pred = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            mismatch = preds != labels
            if mismatch.any():
                for i in range(inputs.size(0)):
                    if mismatch[i]:
                        mis_images.append(inputs[i].cpu())
                        mis_true.append(labels[i].item())
                        mis_pred.append(preds[i].item())
                        if len(mis_images) >= max_images:
                            break
            if len(mis_images) >= max_images:
                break

    # Візуалізація
    n = len(mis_images)
    cols = 5
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(15, 3 * rows))
    for i in range(n):
        img = mis_images[i]
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"T: {class_names[mis_true[i]]}\nP: {class_names[mis_pred[i]]}")
    plt.tight_layout()
    plt.show()

print("Misclassified examples - Simple CNN")
show_misclassified(simple_cnn, test_loader, class_names, max_images=15)

print("Misclassified examples - ResNet18")
show_misclassified(resnet, test_loader, class_names, max_images=15)

