from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def load_data(data_dir, batch_size=16):
    # Definiere Transformationen für die Bilder
    transform = transforms.Compose([
        transforms.Resize((150, 150)),  # Ändert die Größe der Bilder auf 150x150 Pixel
        transforms.ToTensor(),           # Konvertiert die Bilder in Tensoren (Werte zwischen 0 und 1)
    ])

    # Lade das Dataset aus einem Ordner, wobei die Struktur des Ordners den Klassen entspricht
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Splitte das Dataset in Trainings- und Testdaten (67% Training, 33% Test)
    train_size = int(0.67 * len(dataset))  # Berechne die Größe des Trainingssets
    test_size = len(dataset) - train_size   # Berechne die Größe des Testsets
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])  # Teile das Dataset in zwei Teile

    # Erstelle DataLoader für das Training und den Test
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Trainingsdaten in Batches
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)    # Testdaten ohne Shuffle

    return train_loader, test_loader  # Gebe die DataLoader für das Training und den Test zurück
