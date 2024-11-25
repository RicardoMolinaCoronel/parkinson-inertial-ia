import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import Global.config as config
x=0


# Función para preprocesar datos
def preprocess_data(X_train, X_test):
    # Aplicar StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Función para cargar y procesar archivos JSON
def load_features_from_folders(folder_0, folder_1):
    features = []
    labels = []

    # Función para filtrar y limpiar los registros
    def process_file(file_path, label):
        global x
        with open(file_path, 'r') as f:
            data = json.load(f)

            for window in data:
                # Filtrar solo si el sensor es "derecha"
                if window.get("sensor") == "derecha":
                    # Eliminar campos innecesarios
                    window.pop("sensor", None)
                    window.pop("segment_index", None)
                    window.pop("window_index", None)
                    window.pop("start_time", None)
                    window.pop("end_time", None)

                    # Agregar las características y el label
                    features.append(list(window.values()))
                    labels.append(label)
                if(x==0):
                    print(file_path)
                    print(features)
                x+=1

    # Leer archivos de la carpeta con label 0
    for file in os.listdir(folder_0):
        file_path = os.path.join(folder_0, file)
        if file.endswith('.json'):
            process_file(file_path, 0)

    # Leer archivos de la carpeta con label 1
    for file in os.listdir(folder_1):
        file_path = os.path.join(folder_1, file)
        if file.endswith('.json'):
            process_file(file_path, 1)

    return np.array(features), np.array(labels)


# Función para crear y entrenar la MLP
def train_mlp(X_train, X_test, y_train, y_test, labels, model_path):

    # Crear el modelo MLP
    model = Sequential([
        Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')  # Clasificación binaria
    ])

    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    # Entrenamiento
    model.fit(X_train, y_train, epochs=700, batch_size=32, validation_split=0.2)

    # Evaluar el modelo en el conjunto de prueba
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Guardar el modelo
    model.save(model_path)
    print(f"Modelo guardado en {model_path}")

    return model


# Configuración de carpetas
folder_0 = config.p_walking_data_path_no_parkinson  # Reemplaza con la ruta de la carpeta con label 0
folder_1 = config.p_walking_data_path_parkinson  # Reemplaza con la ruta de la carpeta con label 1



# Procesar datos
features, labels = load_features_from_folders(folder_0, folder_1)

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
print(X_train.shape)


# Preprocesar datos (fit en entrenamiento, transform en prueba)
X_train_preprocessed, X_test_preprocessed = preprocess_data(X_train, X_test)


# Entrenar MLP
model = train_mlp(X_train_preprocessed, X_test_preprocessed, y_train, y_test, labels, model_path='MODELS/model2.keras')

