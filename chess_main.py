import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import imgaug.augmenters as iaa
import numpy as np

# Pfade und Parameter
DATA_DIR = "./data"
BATCH_SIZE = 32
IMG_SIZE = (128, 128)
NUM_CLASSES = 4
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Augmentation mit imgaug
seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Crop(percent=(0, 0.1)),
    iaa.Affine(rotate=(-20, 20)),
    iaa.Multiply((0.8, 1.2))
])

def imgaug_transform(img):
    img_np = np.array(img)
    img_aug = seq(images=[img_np])[0]
    return torch.tensor(img_aug).permute(2, 0, 1) / 255.0

# Datenaufbereitung ohne Augmentation
transform_no_aug = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Datenaufbereitung mit Augmentation
transform_with_aug = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.Lambda(imgaug_transform),  # Augmentierung anwenden
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Datensätze und Loader
dataset_no_aug = datasets.ImageFolder(DATA_DIR, transform=transform_no_aug)
dataset_with_aug = datasets.ImageFolder(DATA_DIR, transform=transform_with_aug)

# Trainings- und Validierungsaufteilung (75/25)
train_size = int(0.75 * len(dataset_no_aug))
val_size = len(dataset_no_aug) - train_size
train_dataset_no_aug, val_dataset_no_aug = random_split(dataset_no_aug, [train_size, val_size])
train_dataset_with_aug, val_dataset_with_aug = random_split(dataset_with_aug, [train_size, val_size])

train_loader_no_aug = DataLoader(train_dataset_no_aug, batch_size=BATCH_SIZE, shuffle=True)
val_loader_no_aug = DataLoader(val_dataset_no_aug, batch_size=BATCH_SIZE)

train_loader_with_aug = DataLoader(train_dataset_with_aug, batch_size=BATCH_SIZE, shuffle=True)
val_loader_with_aug = DataLoader(val_dataset_with_aug, batch_size=BATCH_SIZE)

# Modelldefinition
class SimpleCNN(nn.Module):
    def __init__(self, num_hidden_layers):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Dummy-Durchlauf zur Berechnung der Größe nach der Convolution
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, IMG_SIZE[0], IMG_SIZE[1])
            conv_output_size = self.conv(dummy_input).view(1, -1).size(1)

        # Erstellung der FC-Schichten mit Dropout
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            *(nn.Sequential(
                nn.Linear(conv_output_size, 128),
                nn.ReLU(),
                nn.Dropout(0.5)) for _ in range(num_hidden_layers)),
            nn.Linear(128, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc_layers(x)

# Trainingsfunktion
def train(model, train_loader, val_loader, name):
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        val_loss, val_accuracy = evaluate(model, val_loader)
        print(f"[{name}] Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss / len(train_loader):.4f}, Accuracy: {correct / total:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

# Evaluierungsfunktion
def evaluate(model, data_loader):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return total_loss / len(data_loader), correct / total

# Training und Auswertung
print("Training ohne Augmentation...")
model_no_aug = SimpleCNN(num_hidden_layers=1)
train(model_no_aug, train_loader_no_aug, val_loader_no_aug, "Ohne Augmentation")

print("Training mit Augmentation...")
model_with_aug = SimpleCNN(num_hidden_layers=3)
train(model_with_aug, train_loader_with_aug, val_loader_with_aug, "Mit Augmentation")
