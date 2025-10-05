import os
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import json

# Załaduj dane MNIST (do ewaluacji)
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
model_path = "model.keras"

# Sprawdź, czy istnieje zapisany model
if os.path.exists(model_path):
    print("Exist")
    model = keras.models.load_model(model_path)
else:
    print("Training new model...")
    # Budowa modelu
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    # Kompilacja modelu
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # Trenowanie modelu
    history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
    with open("history.json", "w") as f:
        json.dump(history.history, f)
    # Zapisz model
    model.save(model_path)

history = None
if os.path.exists("history.json"):
    with open("history.json", "r") as f:
        hist_data = json.load(f)

# Pobierz ścieżkę obrazu
if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    image_path = input("Podaj ścieżkę do obrazu (28x28 lub większy, grayscale): ")

# Wczytanie obrazu używając tf.keras.utils.load_img
img = tf.keras.utils.load_img(image_path, color_mode='grayscale', target_size=(28, 28))
img_array = tf.keras.utils.img_to_array(img) / 255.0
img_array = tf.expand_dims(img_array, 0)  # dodanie wymiaru batch

# Predykcja
predictions = model.predict(img_array)
predicted_class = tf.argmax(predictions[0]).numpy()
print(f"Przewidywana cyfra: {predicted_class}")

# Wyświetlenie obrazu
plt.imshow(img, cmap='gray')
plt.show()

# Krzywa uczenia
if hist_data:
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(hist_data['loss'], label='Loss (train)')
    plt.plot(hist_data['val_loss'], label='Loss (val)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(hist_data['accuracy'], label='Accuracy (train)')
    plt.plot(hist_data['val_accuracy'], label='Accuracy (val)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.show()
