import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
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

# Verbesserte Augmentation mit imgaug
# Eine Reihe von Bildaugmentierungen, die zufällig auf die Bilder angewendet werden
seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # Zufälliges horizontales Spiegeln mit 50% Wahrscheinlichkeit
    iaa.Crop(percent=(0, 0.05)),  # Zufälliges Zuschneiden des Bildes
    iaa.Multiply((0.95, 1.05)),  # Helligkeit und Kontrast variieren
    iaa.GaussianBlur(sigma=(0, 0.5))  # Geringe Unschärfe auf die Bilder anwenden
])


# Funktion, um die Bildaugmentation auf ein Bild anzuwenden
def imgaug_transform(img):
    img_np = np.array(img)  # Konvertiere das Bild in ein NumPy-Array
    img_aug = seq(images=[img_np])[0]  # Augmentiere das Bild
    img_tensor = torch.tensor(img_aug).float().permute(2, 0, 1) / 255.0  # Konvertiere das Bild in einen Tensor und normalisiere
    return img_tensor

# Datenaufbereitung ohne Augmentation
# Transformationen für das Training ohne Augmentation
transform_no_aug = transforms.Compose([
    transforms.Resize(IMG_SIZE),  # Bildgröße anpassen
    transforms.ToTensor(),  # Umwandlung in Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalisierung
])

# Datenaufbereitung mit Augmentation
# Transformationen für das Training mit Augmentation
transform_with_aug = transforms.Compose([
    transforms.Resize(IMG_SIZE),  # Bildgröße anpassen
    transforms.Lambda(imgaug_transform),  # Anwenden der benutzerdefinierten Augmentation
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalisierung
])

# Datensätze und DataLoader für Training und Validierung
dataset_no_aug = datasets.ImageFolder(DATA_DIR, transform=transform_no_aug)  # Dataset ohne Augmentation
dataset_with_aug = datasets.ImageFolder(DATA_DIR, transform=transform_with_aug)  # Dataset mit Augmentation

# Trainings- und Validierungsaufteilung (75% Training, 25% Validierung)
train_size = int(0.75 * len(dataset_no_aug))  # 75% der Daten für Training
val_size = len(dataset_no_aug) - train_size  # Rest für Validierung
train_dataset_no_aug, val_dataset_no_aug = random_split(dataset_no_aug, [train_size, val_size])
train_dataset_with_aug, val_dataset_with_aug = random_split(dataset_with_aug, [train_size, val_size])

# DataLoader für Training und Validierung
train_loader_no_aug = DataLoader(train_dataset_no_aug, batch_size=BATCH_SIZE, shuffle=True)  # Training ohne Augmentation
val_loader_no_aug = DataLoader(val_dataset_no_aug, batch_size=BATCH_SIZE)  # Validierung ohne Augmentation
train_loader_with_aug = DataLoader(train_dataset_with_aug, batch_size=BATCH_SIZE, shuffle=True)  # Training mit Augmentation
val_loader_with_aug = DataLoader(val_dataset_with_aug, batch_size=BATCH_SIZE)  # Validierung mit Augmentation

# Modelldefinition
class SimpleCNN(nn.Module):
    def __init__(self, num_hidden_layers):
        super(SimpleCNN, self).__init__()
        # Convolutional Layer
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Convolutional Layer
            nn.ReLU(),  # Aktivierungsfunktion ReLU
            nn.MaxPool2d(kernel_size=2)  # Max-Pooling Layer
        )

        # Berechnung der Größe der Ausgabeschicht nach der Convolution
        conv_output_size = 32 * (IMG_SIZE[0] // 2) * (IMG_SIZE[1] // 2)

        # Fully Connected Layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # Flacht die Ausgaben aus der Convolution
            *(nn.Sequential(
                nn.Linear(conv_output_size if i == 0 else 128, 128),
                nn.ReLU(),
                nn.Dropout(0.5)) for i in range(num_hidden_layers)),  # Dropout für Regularisierung
            nn.Linear(128, NUM_CLASSES)  # Finaler Fully Connected Layer für die Klassifikation
        )

    def forward(self, x):
        x = self.conv(x)  # Convolutional Layer anwenden
        return self.fc_layers(x)  # Fully Connected Layer anwenden

# Trainingsfunktion
def train(model, train_loader, val_loader, name):
    model.to(DEVICE)  # Modell auf das richtige Gerät verschieben
    criterion = nn.CrossEntropyLoss()  # Verlustfunktion
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimierer

    for epoch in range(EPOCHS):
        model.train()  # Setzt das Modell in den Trainingsmodus
        running_loss = 0.0
        correct, total = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)  # Verschiebt Daten auf das Gerät
            optimizer.zero_grad()  # Setzt die Gradienten zurück
            outputs = model(images)  # Vorhersage des Modells
            loss = criterion(outputs, labels)  # Berechnet den Verlust
            loss.backward()  # Berechnet die Gradienten
            optimizer.step()  # Aktualisiert die Modellparameter
            running_loss += loss.item()  # Akkumuliert den Verlust
            _, predicted = torch.max(outputs.data, 1)  # Vorhersage der Klasse
            total += labels.size(0)  # Erhöht die Gesamtzahl der Bilder
            correct += (predicted == labels).sum().item()  # Zählt die korrekten Vorhersagen

        # Validierung des Modells nach jedem Epoch
        val_loss, val_accuracy = evaluate(model, val_loader)
        print(f"[{name}] Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss / len(train_loader):.4f}, Accuracy: {correct / total:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Speichern des Modells nach dem Training
    torch.save(model.state_dict(), f'{name.replace(" ", "_").lower()}_model.pth')

# Evaluierungsfunktion
def evaluate(model, data_loader):
    model.eval()  # Setzt das Modell in den Evaluierungsmodus
    total_loss, correct, total = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss()  # Verlustfunktion
    with torch.no_grad():  # Keine Gradientenberechnung
        for images, labels in data_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)  # Verschiebt die Daten auf das Gerät
            outputs = model(images)  # Modellvorhersage
            loss = criterion(outputs, labels)  # Berechnet den Verlust
            total_loss += loss.item()  # Akkumuliert den Verlust
            _, predicted = torch.max(outputs.data, 1)  # Vorhersage der Klasse
            total += labels.size(0)  # Erhöht die Gesamtzahl der Bilder
            correct += (predicted == labels).sum().item()  # Zählt die korrekten Vorhersagen
    return total_loss / len(data_loader), correct / total  # Gibt den durchschnittlichen Verlust und die Genauigkeit zurück

# Main-Block für das Training und Speichern der Modelle
if __name__ == "__main__":
    print("Training ohne Augmentation...")
    model_no_aug = SimpleCNN(num_hidden_layers=1)  # Modell ohne Augmentation
    train(model_no_aug, train_loader_no_aug, val_loader_no_aug, "Ohne Augmentation")  # Training ohne Augmentation

    print("Training mit Augmentation...")
    model_with_aug = SimpleCNN(num_hidden_layers=1)  # Modell mit Augmentation
    train(model_with_aug, train_loader_with_aug, val_loader_with_aug, "Mit Augmentation")  # Training mit Augmentation
