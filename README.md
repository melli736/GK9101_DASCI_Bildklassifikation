# GK9101 Bildklassifikation mit Neuronalen Netzwerken 

**Author:** Melissa Wallapch 5BHIT
**Version:** 2023-10-24

### Handwritten Symbol Recognition mit PyTorch

Dieses Projekt trainiert ein CNN-Modell zur Unterscheidung von handgeschriebenen Mathematiksymbolen (Ausrufezeichen und rechte Klammern). Es verwendet den PyTorch-Framework.

## Projektstruktur
#### Voraussetzungen
- **Python** installiert (mit PyTorch)
- Grundlegende Kenntnisse in neuronalen Netzwerken, CNNs, Tensors, Layern und Deep Learning
- Bilddatensatz von handgeschriebenen Symbolen aus Kaggle

Dieses Projekt trainiert ein CNN-Modell zur Unterscheidung von handgeschriebenen Mathematiksymbolen (Ausrufezeichen und rechte Klammern). Es verwendet den PyTorch-Framework.

``pip install torch torchvision
shell``

### Trainieren des Modells

Um das Modell zu trainieren, führe folgendes Skript aus:

``python model_training.py``

Dies trainiert das Modell mit verschiedenen Anzahlen von versteckten Schichten (64, 128, 256).

## Testen des Modells
``python model_testing.py``

Dies gibt die Genauigkeit des Modells auf den Testdaten aus.


## Übersicht 
In meinem Projekt geht es darum, ein Convolutional Neural Network (CNN) zu erstellen, das handgeschriebene Mathematiksymbole klassifizieren kann. Ich habe mehrere Python-Dateien erstellt, die zusammenarbeiten, um das Modell zu trainieren, zu testen und Daten aus einem Kaggle-Datensatz zu laden. Im Folgenden erkläre ich, wie jede Datei funktioniert und wie sie zu dem Gesamtprojekt beiträgt.

### Aufbau:
Das Projekt nutzt ein CNN, um handgeschriebene Mathematiksymbole zu klassifizieren. Dabei experimentiere ich mit verschiedenen Modellvarianten, die unterschiedliche Größen der versteckten Schichten haben. Die wichtigsten Schritte, die ich in meinem Code umsetze, umfassen:

1. **Daten laden:** Ich lade sowohl Trainings- als auch Testdaten aus einem Kaggle-Datensatz herunter.
2. **Modell definieren:** In `model_training.py` habe ich das CNN-Modell definiert, das aus zwei Faltungsschichten und zwei vollständig verbundenen Schichten besteht.
3. **Modell trainieren:** Ich trainiere das Modell für jede Konfiguration der versteckten Schichten und speichere das trainierte Modell.
4. **Modell testen:** Nachdem das Modell trainiert wurde, teste ich es, um zu sehen, wie gut es auf den Testdaten funktioniert.

### 1. **`model_testing.py`**
In dieser Datei geht es darum, das trainierte Modell zu testen:

- **Funktion `test_model`:**  
    Diese Funktion setzt das Modell in den Evaluationsmodus, sodass keine Gradientenberechnung erfolgt, und testet dann das Modell auf den Testdaten. Für jedes Bild, das durch das Modell läuft, berechne ich die Vorhersage und vergleiche sie mit dem echten Label. Die Anzahl der richtigen Vorhersagen wird gezählt und am Ende berechne ich die Genauigkeit des Modells in Prozent.
  
- **Main-Block:**  
    Hier lade ich die Testdaten und teste das Modell mit verschiedenen Größen der versteckten Schichten (64, 128 und 256). Für jede Konfiguration lade ich die gespeicherten Modellgewichte, die nach dem Training in `model_training.py` gespeichert wurden, und teste das Modell mit den entsprechenden Testdaten.

### 2. **`model_training.py`**
In dieser Datei definiere ich das CNN-Modell und die Trainingslogik:

- **Klasse `SymbolCNN`:**  
    Hier definiere ich die Architektur des CNN. Das Modell besteht aus zwei Faltungsschichten (Conv2d), gefolgt von zwei vollständig verbundenen Schichten (Linear). In der `forward`-Methode lege ich fest, wie die Eingabebilder durch das Netzwerk laufen. Jede Faltungsschicht wird durch Max-Pooling und eine ReLU-Aktivierung gefolgt, bevor die Ausgaben der Faltungsschichten in die vollständig verbundenen Schichten übergeben werden.

- **Funktion `train_model`:**  
    Diese Funktion trainiert das Modell über mehrere Epochen. Für jede Epoche berechne ich den Verlust (mit der `CrossEntropyLoss`-Funktion) und aktualisiere die Modellgewichte mithilfe des Adam-Optimierers. Am Ende jeder Epoche gebe ich den durchschnittlichen Verlust aus. Nachdem das Modell trainiert wurde, speichere ich die Modellgewichte auf der Festplatte, damit ich sie später in `model_testing.py` verwenden kann.

- **Main-Block:**  
    Hier lade ich die Trainingsdaten und trainiere das Modell mit verschiedenen Größen der versteckten Schichten. Ich speichere das Modell für jede Konfiguration der versteckten Schichten, damit ich sie später testen kann.

### 3. **`Kaggle_download.py`**
In dieser Datei lade ich den Datensatz von Kaggle herunter:

- **`kagglehub.dataset_download`:**  
    Diese Methode ermöglicht es mir, den Datensatz `xainano/handwrittenmathsymbols` von Kaggle herunterzuladen. Nachdem der Datensatz heruntergeladen wurde, wird der Pfad zu den Dateien ausgegeben, sodass ich weiß, wo die Daten liegen, um sie weiterzuverarbeiten.

### Wichtige Punkte:
- **Datenverarbeitung:** Ich gehe davon aus, dass die Daten bereits in einem geeigneten Format vorliegen. Es gibt eine `utils.py`-Datei, die in meinem Code importiert wird, aber sie ist nicht im Projekt enthalten. Ich gehe davon aus, dass diese Datei die Funktionalität zum Laden und Vorverarbeiten der Bilder enthält.
  
- **Modellvarianten:** Ich experimentiere mit verschiedenen Größen der versteckten Schichten, um herauszufinden, welche Konfiguration die besten Ergebnisse liefert. Dies ist eine gängige Methode, um die Hyperparameter eines Modells zu optimieren.

- **Testen und Auswertung:** Nachdem das Modell trainiert wurde, teste ich es mit den Testdaten, um zu überprüfen, wie gut es auf neuen, unbekannten Daten funktioniert. Dies ist ein wichtiger Schritt, um sicherzustellen, dass das Modell verallgemeinert und nicht nur auf den Trainingsdaten gut funktioniert.

### Weitere Überlegungen:
- Ich könnte eine **Visualisierung** der Trainings- und Testgenauigkeit hinzufügen, um zu sehen, wie sich das Modell im Laufe der Zeit verbessert.
- **Hyperparameter-Optimierung:** Ich könnte auch andere Hyperparameter wie die Lernrate anpassen oder zusätzliche Techniken wie Dropout ausprobieren, um das Modell weiter zu verbessern.

Insgesamt habe ich ein funktionierendes System entwickelt, das ein CNN nutzt, um handgeschriebene Mathematiksymbole zu klassifizieren. Der Code ist gut strukturiert und deckt alle wichtigen Schritte des Trainings und Testens ab.

