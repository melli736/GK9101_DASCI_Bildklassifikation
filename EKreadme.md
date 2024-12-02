# Schachfiguren-Klassifikation mit PyTorch

Dieses Projekt implementiert ein Convolutional Neural Network (CNN) zur Klassifikation von Schachfiguren unter Verwendung von PyTorch. Es demonstriert den Einfluss von Datenaugmentation und unterschiedlichen Netzwerkarchitekturen auf die Modellleistung.

## Projektstruktur

```
/project-root
├── data/
│   ├── Queen/
│   ├── King/
│   ├── Knight/
│   └── Pawn/
├── mit_augmentation_model.pth
├── ohne_augmentation_model.pth
├── chess_training.py
├── chess_testing.py
└── requirements.txt
```

- `data/`: Enthält Bilddaten der Schachfiguren, unterteilt in Unterordner für jede Klasse.
- `mit_augmentation_model.pth`: Gespeichertes Modell, trainiert mit Datenaugmentation.
- `ohne_augmentation_model.pth`: Gespeichertes Modell, trainiert ohne Datenaugmentation.
- `chess_training.py`: Skript zum Training der Modelle.
- `chess_testing.py`: Skript zum Testen der trainierten Modelle.
- `requirements.txt`: Liste der benötigten Python-Pakete.

## Funktionsweise

### Training (chess_training.py)

1. Datenvorbereitung:
   - Lädt Bilder aus dem `data/` Verzeichnis.
   - Teilt Daten in Trainings- (75%) und Validierungssets (25%).
   - Erstellt zwei Datensätze: einen ohne und einen mit Datenaugmentation.

2. Modellarchitektur:
   - Definiert ein einfaches CNN (`SimpleCNN`) mit variabler Anzahl hidden Layers.

3. Training:
   - Trainiert zwei Modelle: eines ohne und eines mit Datenaugmentation.
   - Verwendet Adam-Optimizer und Cross-Entropy-Verlustfunktion.
   - Speichert die trainierten Modelle als `.pth`-Dateien.

4. Evaluation:
   - Bewertet die Modellleistung nach jeder Epoche auf dem Validierungsset.

### Testing (chess_testing.py)

1. Lädt die trainierten Modelle.
2. Bereitet einen Testdatensatz vor.
3. Evaluiert beide Modelle auf dem Testset und gibt die Genauigkeit aus.

## Datenaugmentation

Verwendet die `imgaug`-Bibliothek für Bildtransformationen wie horizontales Spiegeln, Zuschneiden, Rotation und Helligkeitsanpassungen.

## Modellvergleich

Das Projekt vergleicht zwei Ansätze:
1. Training ohne Augmentation (1 versteckte Schicht)
2. Training mit Augmentation (3 hidden Layers)

Dies ermöglicht eine Analyse des Einflusses von Datenaugmentation und Netzwerktiefe auf die Modellleistung.

## Erklärung des Codes
Gerne erkläre ich den Code detailliert mit Code-Snippets:

chess_training.py

### Importe und Konfiguration

```python
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
```

Hier werden die notwendigen Bibliotheken importiert und globale Parameter definiert.

### Datenaugmentation

```python
seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Crop(percent=(0, 0.1)),
    iaa.Affine(rotate=(-20, 20)),
    iaa.Multiply((0.8, 1.2)),
    iaa.Resize(IMG_SIZE)
])

def imgaug_transform(img):
    img_np = np.array(img)
    img_aug = seq(images=[img_np])[0]
    img_tensor = torch.tensor(img_aug).float().permute(2, 0, 1) / 255.0
    return img_tensor
```

Diese Funktion definiert die Augmentationssequenz und eine Funktion zur Anwendung der Augmentation auf einzelne Bilder.

### Datentransformationen

```python
transform_no_aug = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

transform_with_aug = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.Lambda(imgaug_transform),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
```

Hier werden zwei Transformationspipelines definiert: eine ohne und eine mit Augmentation.

### Datensätze und DataLoader

```python
dataset_no_aug = datasets.ImageFolder(DATA_DIR, transform=transform_no_aug)
dataset_with_aug = datasets.ImageFolder(DATA_DIR, transform=transform_with_aug)

train_size = int(0.75 * len(dataset_no_aug))
val_size = len(dataset_no_aug) - train_size
train_dataset_no_aug, val_dataset_no_aug = random_split(dataset_no_aug, [train_size, val_size])
train_dataset_with_aug, val_dataset_with_aug = random_split(dataset_with_aug, [train_size, val_size])

train_loader_no_aug = DataLoader(train_dataset_no_aug, batch_size=BATCH_SIZE, shuffle=True)
val_loader_no_aug = DataLoader(val_dataset_no_aug, batch_size=BATCH_SIZE)
train_loader_with_aug = DataLoader(train_dataset_with_aug, batch_size=BATCH_SIZE, shuffle=True)
val_loader_with_aug = DataLoader(val_dataset_with_aug, batch_size=BATCH_SIZE)
```

Dieser Abschnitt erstellt Datensätze und DataLoader für Training und Validierung, jeweils mit und ohne Augmentation.

### Modelldefinition

```python
class SimpleCNN(nn.Module):
    def __init__(self, num_hidden_layers):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        conv_output_size = 32 * (IMG_SIZE[0] // 2) * (IMG_SIZE[1] // 2)

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            *(nn.Sequential(
                nn.Linear(conv_output_size if i == 0 else 128, 128),
                nn.ReLU(),
                nn.Dropout(0.5)) for i in range(num_hidden_layers)),
            nn.Linear(128, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc_layers(x)
```

Diese Klasse definiert das CNN-Modell mit variabler Anzahl versteckter Schichten.

### Trainings- und Evaluierungsfunktionen

```python
def train(model, train_loader, val_loader, name):
    # ... (Trainingslogik)

def evaluate(model, data_loader):
    # ... (Evaluierungslogik)
```

Diese Funktionen implementieren die Trainings- und Evaluierungslogik für die Modelle.

### Hauptausführung

```python
if __name__ == "__main__":
    print("Training ohne Augmentation...")
    model_no_aug = SimpleCNN(num_hidden_layers=1)
    train(model_no_aug, train_loader_no_aug, val_loader_no_aug, "Ohne Augmentation")

    print("Training mit Augmentation...")
    model_with_aug = SimpleCNN(num_hidden_layers=3)
    train(model_with_aug, train_loader_with_aug, val_loader_with_aug, "Mit Augmentation")
```

Dieser Teil führt das Training für beide Modellvarianten durch.

chess_testing.py

### Importe und Datenvorbereitung

```python
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from chess_training import SimpleCNN, IMG_SIZE, NUM_CLASSES, DEVICE, DATA_DIR

test_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_dataset = datasets.ImageFolder(DATA_DIR, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```

Dieser Teil bereitet die Testdaten vor.

### Testfunktion

```python
def test_model(model, test_loader, model_name):
    # ... (Testlogik)
```

Diese Funktion führt den Test für ein gegebenes Modell durch.

### Hauptausführung

```python
if __name__ == "__main__":
    model_no_aug = SimpleCNN(num_hidden_layers=1)
    model_no_aug.load_state_dict(torch.load('ohne_augmentation_model.pth', weights_only=True))
    model_no_aug.to(DEVICE)
    test_model(model_no_aug, test_loader, "Modell ohne Augmentation")

    model_with_aug = SimpleCNN(num_hidden_layers=3)
    model_with_aug.load_state_dict(torch.load('mit_augmentation_model.pth', weights_only=True))
    model_with_aug.to(DEVICE)
    test_model(model_with_aug, test_loader, "Modell mit Augmentation")
```

Dieser Teil lädt die trainierten Modelle und führt den Test durch.

## Ergebnisse
Basierend auf den Testergebnissen können wir folgende Interpretationen und Verbesserungsvorschläge machen:

1. Interpretation der Ergebnisse:

   - Das Modell ohne Augmentation erreicht eine sehr hohe Testgenauigkeit von 98.89%, was auf eine gute Leistung hindeutet.
   - Das Modell mit Augmentation zeigt eine deutlich niedrigere Testgenauigkeit von 62.22%, was überraschend ist, da Augmentation normalerweise die Generalisierungsfähigkeit verbessern sollte.

2. Mögliche Gründe für die Unterschiede:

   - Overfitting beim Modell ohne Augmentation: Die perfekte Trainingsgenauigkeit (1.0000) in den späteren Epochen deutet auf Overfitting hin, obwohl die Testgenauigkeit hoch bleibt.
   - Unterfitting beim Modell mit Augmentation: Die niedrigen Trainings- und Validierungsgenauigkeiten deuten darauf hin, dass das Modell Schwierigkeiten hat, die augmentierten Daten zu lernen.

3. Verbesserungsvorschläge:

   a) Datenaugmentation anpassen:
      - Reduzieren Sie die Stärke der Augmentationen, z.B. geringere Rotationswinkel oder weniger aggressive Zuschnitte.
      - Fügen Sie subtilere Augmentationen hinzu, wie leichte Farbänderungen oder Helligkeitsanpassungen.

   ```python
   seq = iaa.Sequential([
       iaa.Fliplr(0.5),
       iaa.Crop(percent=(0, 0.05)),  # Weniger aggressives Zuschneiden
       iaa.Affine(rotate=(-10, 10)),  # Geringere Rotation
       iaa.Multiply((0.9, 1.1)),  # Subtilere Helligkeitsänderungen
       iaa.AddToBrightness((-10, 10)),  # Leichte Helligkeitsanpassungen
       iaa.Resize(IMG_SIZE)
   ])
   ```

   b) Modellarchitektur anpassen:
      - Erhöhen Sie die Komplexität des Modells für das augmentierte Training, z.B. durch Hinzufügen weiterer Convolutional Layer.

   ```python
   class ImprovedCNN(nn.Module):
       def __init__(self):
           super(ImprovedCNN, self).__init__()
           self.features = nn.Sequential(
               nn.Conv2d(3, 32, kernel_size=3, padding=1),
               nn.ReLU(),
               nn.MaxPool2d(kernel_size=2),
               nn.Conv2d(32, 64, kernel_size=3, padding=1),
               nn.ReLU(),
               nn.MaxPool2d(kernel_size=2)
           )
           
           self.classifier = nn.Sequential(
               nn.Flatten(),
               nn.Linear(64 * (IMG_SIZE[0] // 4) * (IMG_SIZE[1] // 4), 256),
               nn.ReLU(),
               nn.Dropout(0.5),
               nn.Linear(256, NUM_CLASSES)
           )

       def forward(self, x):
           x = self.features(x)
           return self.classifier(x)
   ```

   c) Hyperparameter-Tuning:
      - Experimentieren Sie mit verschiedenen Lernraten und Batch-Größen.
      - Verwenden Sie einen Learning Rate Scheduler, um die Lernrate im Laufe des Trainings anzupassen.

   ```python
   optimizer = optim.Adam(model.parameters(), lr=0.001)
   scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

   # In der Trainingsfunktion:
   scheduler.step(val_loss)
   ```

   d) Regularisierung:
      - Fügen Sie L2-Regularisierung (Weight Decay) hinzu, um Overfitting zu reduzieren.

   ```python
   optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
   ```

   e) Früher Stopp:
      - Implementieren Sie einen Early Stopping-Mechanismus, um das Training zu beenden, wenn sich die Validierungsleistung nicht mehr verbessert.

   f) Kreuzvalidierung:
      - Verwenden Sie K-Fold-Kreuzvalidierung, um eine robustere Einschätzung der Modellleistung zu erhalten.

   g) Datenanalyse:
      - Überprüfen Sie die Verteilung der Klassen in Ihrem Datensatz und stellen Sie sicher, dass er ausgewogen ist.
      - Analysieren Sie die falsch klassifizierten Bilder, um mögliche Muster oder Probleme zu identifizieren.

