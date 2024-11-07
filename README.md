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

## Output - Format 
Die `.pth`-Dateien, die in deinem Projekt verwendet werden, sind **Modelldateien** von PyTorch. Sie enthalten die **gewonnenen Modellgewichte** (auch als Parameter bezeichnet), die das neuronale Netzwerk während des Trainings gelernt hat.

### Was ist eine `.pth`-Datei?

- **.pth** ist eine gängige Dateiendung in PyTorch, die für **"path"** oder **"PyTorch"** steht und verwendet wird, um gespeicherte Modellgewichte zu speichern. Es handelt sich dabei um ein **binäres Format**, das die Gewichte und Parameter des trainierten Modells enthält, damit dieses später wiederverwendet oder getestet werden kann, ohne das Modell erneut trainieren zu müssen.

### Warum sind `.pth`-Dateien wichtig?

- **Modell speichern:** Während des Trainings eines neuronalen Netzwerks werden die Modellgewichte immer wieder aktualisiert. Nach dem Abschluss des Trainings möchte man in der Regel den besten Zustand des Modells speichern, um das Training zu einem späteren Zeitpunkt fortzusetzen oder das Modell für Vorhersagen (Inference) zu verwenden. Dies wird in einer `.pth`-Datei gespeichert.
  
- **Wiederverwendung des Modells:** Wenn man das Modell erneut laden möcht, ohne das gesamte Training zu wiederholen, kann man einfach die `.pth`-Datei laden, um die gespeicherten Gewichte wieder in das Modell zu laden und das Modell für Tests oder Vorhersagen zu verwenden.

### In meinem Projekt:

- In der Datei **`model_training.py`** wird das Modell nach jedem Trainingsdurchgang gespeichert mit dem Befehl:
  
  ```python
  torch.save(model.state_dict(), f"{model_name}.pth")
  ```

  Hier wird das Modell nach jedem Training mit der spezifischen Anzahl an versteckten Schichten (z. B. `symbol_cnn_64.pth`, `symbol_cnn_128.pth`) gespeichert. Die `.pth`-Dateien enthalten die Parameter, die das Modell während des Trainings gelernt hat, basierend auf der Architektur (Anzahl und Größe der Schichten) und den Trainingsdaten.

- In **`model_testing.py`** lädst du dann diese gespeicherten `.pth`-Dateien, um das Modell zu testen:
  
  ```python
  model.load_state_dict(torch.load(f"symbol_cnn_{hidden_layer_size}.pth"))
  ```

  Damit werden die Gewichte aus der `.pth`-Datei in das Modell geladen, das du dann auf den Testdaten evaluierst.

### Zusammenfassung:
- **.pth-Dateien** sind PyTorch-Dateien, die die **trainierten Modellgewichte** enthalten.
- Du speicherst sie nach dem Training, um die Ergebnisse zu behalten und das Modell später zu testen oder zu verwenden.
- Durch das Laden der `.pth`-Dateien kannst du das Modell wiederherstellen und seine Vorhersagen auf neuen Daten testen, ohne das Modell erneut zu trainieren.

