from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import os

def load_data(data_dir, batch_size=16):
    # Definiere Transformationen für die Bilder
    transform = transforms.Compose([
        transforms.Resize((150, 150)),  # Ändert die Größe der Bilder auf 150x150 Pixel
        transforms.ToTensor(),           # Konvertiert die Bilder in Tensoren (Werte zwischen 0 und 1)
    ])

    # Lade das Dataset aus einem Ordner, wobei die Struktur des Ordners den Klassen entspricht
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Filtere die Trainingsdaten, um nur die Ordner "bracketRight" und "exclamationmark" zu berücksichtigen
    train_indices = []
    for idx, (img, label) in enumerate(full_dataset):
        folder_name = full_dataset.classes[label]
        if folder_name in ["bracketRight", "exclamationmark"]:
            train_indices.append(idx)

    # Erstelle einen Subset für das Training, der nur die gewünschten Ordner enthält
    train_dataset = Subset(full_dataset, train_indices)

    # Erstelle einen Subset für die Testdaten, der alle Ordner enthält
    test_dataset = full_dataset  # Alle Ordner werden für den Test verwendet

    # Erstelle DataLoader für das Training und den Test
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Trainingsdaten in Batches
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)    # Testdaten ohne Shuffle

    return train_loader, test_loader  # Gebe die DataLoader für das Training und den Test zurück
