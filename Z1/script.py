import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import numpy as np
import sys

if len(sys.argv) > 1:
    try:
        img_index = int(sys.argv[1])
    except ValueError:
        print("Podaj liczbę całkowitą jako parametr.")
        sys.exit(1)
else:
    img_index = 3

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

model.save("model.keras")
save_model = keras.models.load_model("model.keras")

save_model.evaluate(x_test, y_test)
model.evaluate(x_test, y_test)

img = x_test[img_index]

plt.imsave('img.png', img, cmap='gray')
img = tf.keras.preprocessing.image.load_img('img.png', color_mode='grayscale', target_size=(28, 28))
img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
predicted_class = tf.argmax(predictions[0]).numpy()

print(f"Wybrany indeks: {img_index}")
print(f"Predykcja: {predicted_class}")

plt.imshow(img, cmap='gray')
plt.show()
