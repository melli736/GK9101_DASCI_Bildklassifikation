import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# Importieren Sie die SimpleCNN-Klasse und andere notwendige Funktionen aus der train.py-Datei
from chess_training import SimpleCNN, IMG_SIZE, NUM_CLASSES, DEVICE, DATA_DIR

# Testdaten vorbereiten
test_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_dataset = datasets.ImageFolder(DATA_DIR, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def test_model(model, test_loader, model_name):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"{model_name} - Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    # Modell ohne Augmentation laden und testen
    model_no_aug = SimpleCNN(num_hidden_layers=1)
    model_no_aug.load_state_dict(torch.load('ohne_augmentation_model.pth'))
    model_no_aug.to(DEVICE)
    test_model(model_no_aug, test_loader, "Modell ohne Augmentation")

    # Modell mit Augmentation laden und testen
    model_with_aug = SimpleCNN(num_hidden_layers=3)
    model_with_aug.load_state_dict(torch.load('mit_augmentation_model.pth'))
    model_with_aug.to(DEVICE)
    test_model(model_with_aug, test_loader, "Modell mit Augmentation")