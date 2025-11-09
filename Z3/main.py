import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import argparse
import sys
import pickle

DATA_FILE = "wine.data"
COLUMN_NAMES = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols', 'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280_OD315_ratio', 'Proline'
]

def load_and_prepare_data(filepath):
    data = pd.read_csv(filepath, header=None, names=COLUMN_NAMES)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    X = data.drop('Class', axis=1)
    y = data['Class'] - 1

    y_one_hot = to_categorical(y, num_classes=3)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_one_hot, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def build_model_1(lr, input_shape):
    model = Sequential(name="Model_Bazowy")
    model.add(Input(shape=input_shape, name="Warstwa_Wejsciowa"))
    model.add(Dense(16, activation='relu', kernel_initializer='he_uniform', name="Warstwa_Ukryta_2"))
    model.add(Dense(3, activation='softmax', name="Warstwa_Wyjsciowa"))
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_model_2(lr, input_shape):
    model = Sequential(name="Model_Zlozony_z_Dropout")
    model.add(Input(shape=input_shape, name="Warstwa_Wejsciowa"))
    model.add(Dense(64, activation='relu', kernel_initializer='glorot_uniform', name="Warstwa1"))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='tanh', name="Warstwa2"))
    model.add(Dense(3, activation='softmax', name="Warstwa_Wyjsciowa"))
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def plot_history(history, title):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Strata (trening)')
    plt.plot(history.history['val_loss'], label='Strata (walidacja)')
    plt.title(f'Strata (Loss) - {title}')
    plt.xlabel('Epoka')
    plt.ylabel('Strata')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(history.history['accuracy'], label='Dokładność (trening)')
    plt.plot(history.history['val_accuracy'], label='Dokładność (walidacja)')
    plt.title(f'Dokładność (Accuracy) - {title}')
    plt.xlabel('Epoka')
    plt.ylabel('Dokładność')
    plt.legend()
    plt.grid(True)
    plt.show()

def train_and_evaluate():
    prepared_data = load_and_prepare_data(DATA_FILE)
    if prepared_data is None:
        return

    X_train, X_test, y_train, y_test, scaler = prepared_data
    input_shape = (X_train.shape[1],)

    model_1 = build_model_1(0.001, input_shape)
    history_1 = model_1.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=60,
        batch_size=16,
        verbose=1
    )

    model_2 = build_model_2(0.001, input_shape)
    history_2 = model_2.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=60,
        batch_size=16,
        verbose=1
    )

    score_1 = model_1.evaluate(X_test, y_test, verbose=0)
    score_2 = model_2.evaluate(X_test, y_test, verbose=0)

    print(f"Model 1 - Test Loss: {score_1[0]:.4f}, Test Accuracy: {score_1[1]:.4f}")
    print(f"Model 2 - Test Loss: {score_2[0]:.4f}, Test Accuracy: {score_2[1]:.4f}")

    if score_1[1] >= score_2[1]:
        print(f"\nModel 1 jest lepszy. Zapisywanie...")
        best_model = model_1
    else:
        print(f"\nModel 2 jest lepszy. Zapisywanie...")
        best_model = model_2

    best_model.save('wine_model.keras')
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    plot_history(history_1, f"Model 1 (Bazowy)\nDokładność testowa: {score_1[1]:.4f}")
    plot_history(history_2, f"Model 2 (Złożony z Dropout)\nDokładność testowa: {score_2[1]:.4f}")

def predict_from_args():
    try:
        model = load_model('wine_model.keras')
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    except Exception as e:
        print(f"Błąd ładowania modelu lub scalera: {e}")
        return

    parser = argparse.ArgumentParser(description="Przewiduje klasę wina na podstawie 13 cech.")
    parser.add_argument('--alcohol', type=float, required=True)
    parser.add_argument('--malic_acid', type=float, required=True)
    parser.add_argument('--ash', type=float, required=True)
    parser.add_argument('--alcalinity', type=float, required=True)
    parser.add_argument('--magnesium', type=float, required=True)
    parser.add_argument('--phenols', type=float, required=True)
    parser.add_argument('--flavanoids', type=float, required=True)
    parser.add_argument('--nonflavanoid', type=float, required=True)
    parser.add_argument('--proanthocyanins', type=float, required=True)
    parser.add_argument('--color', type=float, required=True)
    parser.add_argument('--hue', type=float, required=True)
    parser.add_argument('--od_ratio', type=float, required=True)
    parser.add_argument('--proline', type=float, required=True)

    args = parser.parse_args(sys.argv[1:])

    input_data = np.array([[
        args.alcohol, args.malic_acid, args.ash, args.alcalinity,
        args.magnesium, args.phenols, args.flavanoids, args.nonflavanoid,
        args.proanthocyanins, args.color, args.hue, args.od_ratio, args.proline
    ]])

    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    predicted_index = np.argmax(prediction, axis=1)[0]
    predicted_class = predicted_index + 1

    print(f"Przewidywana kategoria wina: {predicted_class}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict_from_args()
    else:
        train_and_evaluate()
