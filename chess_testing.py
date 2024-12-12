import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# Importieren Sie die SimpleCNN-Klasse und andere notwendige Funktionen aus der train.py-Datei
from chess_training import SimpleCNN, IMG_SIZE, NUM_CLASSES, DEVICE, DATA_DIR

# Testdaten vorbereiten
# Definiert die Transformationen für die Testdaten
test_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),  # Bildgröße auf IMG_SIZE anpassen
    transforms.ToTensor(),  # Umwandlung der Bilder in Tensoren
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalisierung der Werte
])

# Erstellen des Test-Datasets und DataLoader für das Testen
test_dataset = datasets.ImageFolder(DATA_DIR, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Funktion zur Modellbewertung
def test_model(model, test_loader, model_name):
    model.eval()  # Setzt das Modell in den Evaluierungsmodus (keine Gradientenberechnung)
    correct = 0
    total = 0
    with torch.no_grad():  # Deaktiviert die Berechnung von Gradienten
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)  # Verschiebt die Daten auf das gewünschte Gerät
            outputs = model(images)  # Modellvorhersage
            _, predicted = torch.max(outputs.data, 1)  # Vorhersage der Klasse mit der höchsten Wahrscheinlichkeit
            total += labels.size(0)  # Erhöht die Gesamtzahl der getesteten Bilder
            correct += (predicted == labels).sum().item()  # Zählt die korrekten Vorhersagen

    accuracy = correct / total  # Berechnet die Genauigkeit
    print(f"{model_name} - Test Accuracy: {accuracy:.4f}")

# Main-Block
if __name__ == "__main__":
    # Modell ohne Augmentation laden und testen
    model_no_aug = SimpleCNN(num_hidden_layers=1)
    model_no_aug.load_state_dict(torch.load('ohne_augmentation_model.pth', weights_only=True))  # Modellgewichte laden
    model_no_aug.to(DEVICE)  # Modell auf das richtige Gerät verschieben (CPU oder GPU)
    test_model(model_no_aug, test_loader, "Modell ohne Augmentation")

    # Modell mit Augmentation laden und testen
    model_with_aug = SimpleCNN(num_hidden_layers=1)
    model_with_aug.load_state_dict(torch.load('mit_augmentation_model.pth', weights_only=True))  # Modellgewichte laden
    model_with_aug.to(DEVICE)  # Modell auf das richtige Gerät verschieben
    test_model(model_with_aug, test_loader, "Modell mit Augmentation")