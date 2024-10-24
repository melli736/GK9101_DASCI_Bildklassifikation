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
