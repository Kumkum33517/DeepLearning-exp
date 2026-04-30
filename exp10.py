import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# VGG16 expects 224x224 RGB images and ImageNet normalization
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Using CIFAR-10 — 10-class RGB dataset (good match for VGG16)
train_dataset = datasets.CIFAR10(root="./data", train=True,  download=True, transform=transform_train)
test_dataset  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

print("Train size:", len(train_dataset))
print("Test size: ", len(test_dataset))
# Visualize sample images
images, labels = next(iter(train_loader))

plt.figure(figsize=(12, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    # Unnormalize using ImageNet stats
    img = images[i].numpy().transpose(1, 2, 0)
    img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.title(class_names[labels[i]])
    plt.axis("off")
plt.suptitle("Sample CIFAR-10 Images")
plt.tight_layout()
plt.show()
# Load pre-trained VGG16 and apply Transfer Learning
class TransferVGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(TransferVGG16, self).__init__()

        # Load VGG16 with pre-trained ImageNet weights
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        # FREEZE all convolutional layers (feature extractor stays fixed)
        for param in vgg16.features.parameters():
            param.requires_grad = False

        # Keep the pre-trained feature extractor as-is
        self.features = vgg16.features        # Conv layers (frozen)
        self.avgpool  = vgg16.avgpool         # Adaptive avg pool

        # REPLACE the classifier head for our 10-class problem
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)      # Output: 10 classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = TransferVGG16(num_classes=10).to(device)
print(model)
print("Device:", device)

# Count trainable vs frozen parameters
total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal Parameters:     {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}  (classifier only)")
print(f"Frozen Parameters:    {total_params - trainable_params:,}  (VGG16 conv layers)")
criterion = nn.CrossEntropyLoss()

# Only optimize the classifier — frozen layers have no gradients
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

# Reduce LR if val loss plateaus
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)
EPOCHS = 15
train_losses, val_losses, train_accs, val_accs = [], [], [], []

for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted  = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

    train_losses.append(running_loss / len(train_loader))
    train_accs.append(correct / total)

    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs  = model(images)
            loss     = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_correct += predicted.eq(labels).sum().item()
            val_total   += labels.size(0)

    val_losses.append(val_loss / len(test_loader))
    val_accs.append(val_correct / val_total)
    scheduler.step(val_losses[-1])

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {train_losses[-1]:.4f} | Train Acc: {train_accs[-1]:.4f} | Val Loss: {val_losses[-1]:.4f} | Val Acc: {val_accs[-1]:.4f}")
    model.eval()
correct, total = 0, 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        _, predicted = model(images).max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

print(f"Test Accuracy: {correct / total * 100:.2f}%")
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses,   label="Val Loss")
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label="Train Acc")
plt.plot(val_accs,   label="Val Acc")
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.legend()

plt.tight_layout()
plt.show()
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        _, predicted = model(images).max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix — VGG16 Transfer Learning")
plt.tight_layout()
plt.show()
model.eval()
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

with torch.no_grad():
    _, predicted = model(images).max(1)

plt.figure(figsize=(14, 6))
for i in range(10):
    plt.subplot(2, 5, i+1)
    img = images[i].cpu().numpy().transpose(1, 2, 0)
    img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    pred   = class_names[predicted[i]]
    actual = class_names[labels[i]]
    color  = "green" if pred == actual else "red"
    plt.title(f"P: {pred}\nA: {actual}", color=color, fontsize=8)
    plt.axis("off")
plt.suptitle("Green = Correct | Red = Wrong")
plt.tight_layout()
plt.show()