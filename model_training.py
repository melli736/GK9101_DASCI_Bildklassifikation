import torch
import torch.nn as nn
import torch.optim as optim
from utils import load_data

# CNN-Modell Definition
class SymbolCNN(nn.Module):
    def __init__(self, hidden_layers):
        super(SymbolCNN, self).__init__()  # Initialisiere die Basisklasse
        # Definiere die Convolutional Layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # Erste Faltungsschicht
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # Zweite Faltungsschicht
        # Definiere die vollständig verbundenen Schichten
        self.fc1 = nn.Linear(32 * 37 * 37, hidden_layers)  # Versteckte Schicht, um die Ausgaben der Faltungsschichten zu verarbeiten
        self.fc2 = nn.Linear(hidden_layers, 2)  # Ausgabeschicht für 2 Klassen

    def forward(self, x):
        # Definiere den Vorwärtsdurchlauf des Modells
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))  # Faltung und max Pooling mit ReLU-Aktivierung
        x = torch.relu(torch.max_pool2d(self.conv2(x), 2))  # Faltung und max Pooling mit ReLU-Aktivierung
        x = x.view(x.size(0), -1)  # Flache die Tensoren für die vollständig verbundenen Schichten
        x = torch.relu(self.fc1(x))  # ReLU-Aktivierung auf der versteckten Schicht
        x = self.fc2(x)  # Ausgabe des Modells
        return x  # Gebe die Vorhersagen zurück

# Training-Funktion
def train_model(model, train_loader, num_epochs=100, learning_rate=0.001, model_name="model"):
    criterion = nn.CrossEntropyLoss()  # Verlustfunktion für Klassifikation
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam-Optimierer für Gewichtsaktualisierung

    for epoch in range(num_epochs):  # Schleife über die Epochen
        model.train()  # Setze das Modell in den Trainingsmodus
        running_loss = 0.0  # Zähle den Verlust pro Epoche
        for images, labels in train_loader:  # Iteriere durch die Trainingsdaten
            optimizer.zero_grad()  # Setze die Gradienten auf Null
            outputs = model(images)  # Führe Vorhersagen durch
            loss = criterion(outputs, labels)  # Berechne den Verlust
            loss.backward()  # Berechne die Gradienten
            optimizer.step()  # Aktualisiere die Gewichte
            running_loss += loss.item()  # Addiere den Verlust zur Gesamtzahl

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")  # Ausgabe des Verlusts pro Epoche

    print("Training abgeschlossen.")  # Meldung, wenn das Training beendet ist
    torch.save(model.state_dict(), f"{model_name}.pth")  # Speichere die Modellgewichte


if __name__ == "__main__":
    train_loader, _ = load_data("data", batch_size=16)  # Lade die Trainingsdaten

    # Training mit verschiedenen Größen der versteckten Schichten
    for hidden_layer_size in [64, 128, 256]:
        print(f"\nTraining with {hidden_layer_size} hidden units...")
        model = SymbolCNN(hidden_layers=hidden_layer_size)  # Erstelle das Modell mit der aktuellen Größe
        train_model(model, train_loader, model_name=f"symbol_cnn_{hidden_layer_size}")  # Trainiere das Modell und speichere es
