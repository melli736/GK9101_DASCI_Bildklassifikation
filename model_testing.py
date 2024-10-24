import torch
from utils import load_data
from model_training import SymbolCNN

def test_model(model, test_loader):
    model.eval()  # Setze das Modell in den Evaluationsmodus (keine Gradientenberechnung)
    correct = 0   # Zähle die korrekten Vorhersagen
    total = 0     # Zähle die Gesamtanzahl der Vorhersagen

    with torch.no_grad():  # Deaktiviere die Gradientenberechnung für den Test
        for images, labels in test_loader:
            outputs = model(images)  # Führe Vorhersagen für die Testbilder durch
            _, predicted = torch.max(outputs, 1)  # Hol das Symbol mit der höchsten Wahrscheinlichkeit
            total += labels.size(0)  # Zähle die Anzahl der Labels
            correct += (predicted == labels).sum().item()  # Zähle die korrekten Vorhersagen
            print(f"predicted: {predicted}, label: {labels}")  # Zeige die Vorhersagen und die echten Labels an

    # Berechne und drucke die Genauigkeit des Modells
    print(f"Test Accuracy: {100 * correct / total}%")  # Zeige die Genauigkeit in Prozent an

if __name__ == "__main__":
    # Lade die Testdaten (die Trainingsdaten werden hier ignoriert)
    _, test_loader = load_data("data", batch_size=16)

    # Teste die Modelle mit verschiedenen Größen der versteckten Schichten
    for hidden_layer_size in [64, 128, 256]:
        print(f"\nTesting model with {hidden_layer_size} hidden units...")
        model = SymbolCNN(hidden_layers=hidden_layer_size)  # Erstelle das Modell mit der aktuellen versteckten Schichtgröße
        model.load_state_dict(torch.load(f"symbol_cnn_{hidden_layer_size}.pth"))  # Lade die gespeicherten Gewichte des Modells
        test_model(model, test_loader)  # Teste das Modell mit den Testdaten