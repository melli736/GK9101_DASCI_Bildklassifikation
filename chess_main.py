from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import imgaug.augmenters as iaa
import imageio

# Pfade
DATA_DIR = "./data"
BATCH_SIZE = 32
IMG_SIZE = (128, 128)

# Augmentations-Seq
seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # Horizontale Spiegelung
    iaa.Crop(percent=(0, 0.1)),  # Zufälliger Beschnitt
    iaa.Affine(rotate=(-20, 20)),  # Zufällige Rotation
    iaa.Multiply((0.8, 1.2))  # Helligkeitsanpassung
])

# Datenaufbereitung ohne Augmentation
datagen_no_aug = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.25
)

# Datenaufbereitung mit Augmentation
datagen_with_aug = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.25,
    preprocessing_function=lambda img: seq(images=[img])[0]  # Augmentierte Bilder
)

# Trainings- und Testdaten ohne Augmentation
train_data_no_aug = datagen_no_aug.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='training'
)

test_data_no_aug = datagen_no_aug.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='validation'
)

# Trainings- und Testdaten mit Augmentation
train_data_with_aug = datagen_with_aug.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='training'
)

test_data_with_aug = datagen_with_aug.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='validation'
)

# Modellfunktion
def create_model(num_hidden_layers):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten()
    ])
    for _ in range(num_hidden_layers):
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))  # 4 Klassen
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Training und Auswertung
def train_and_evaluate(model, train_data, test_data, name):
    early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
    history = model.fit(train_data, validation_data=test_data, epochs=20, callbacks=[early_stopping])
    test_loss, test_accuracy = model.evaluate(test_data)
    print(f"{name} - Test Accuracy: {test_accuracy:.2f}, Misclassification Rate: {(1 - test_accuracy):.2f}")
    return history

# Training der Modelle
print("Training ohne Augmentation...")
model_no_aug = create_model(1)  # 1 Hidden Layer
train_and_evaluate(model_no_aug, train_data_no_aug, test_data_no_aug, "Ohne Augmentation")

print("\nTraining mit Augmentation...")
model_with_aug = create_model(3)  # 3 Hidden Layers
train_and_evaluate(model_with_aug, train_data_with_aug, test_data_with_aug, "Mit Augmentation")
