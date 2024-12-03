import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.models import load_model
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import joblib

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import MinMaxScaler
import Global.config as config
from EvaluationMetrics import evaluate_model
x=0

# Función para preprocesar datos
def preprocess_data(X_train, X_test):

    # Aplicar StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Guarda el escalador en un archivo
    joblib.dump(scaler, "MODELS/scaler_mlp.pkl")
    # Aplicar MinMaxScaler
    #normalizer = MinMaxScaler()
    #X_train_scaled= normalizer.fit_transform(X_train)
    #X_test_scaled = normalizer.transform(X_test)
    return X_train_scaled, X_test_scaled

# Función para cargar y procesar archivos JSON
def load_features_from_folders_sensors(folder_0, folder_1):
    features = []
    labels = []

    # Mapeo de sensores a valores numéricos
    sensor_map = {"derecha": 0, "izquierda": 1, "espina_base": 2}

    # Función para filtrar y limpiar los registros
    def process_file(file_path, label):
        with open(file_path, 'r') as f:
            data = json.load(f)
            for window in data:
                # Verificar si el campo "sensor" está en el registro
                if "sensor" in window:
                    # Convertir el sensor a su valor numérico
                    sensor_value = sensor_map.get(window["sensor"])
                    print(sensor_value)
                    if sensor_value is None:
                        raise ValueError(f"Sensor desconocido: {window['sensor']}")

                    # Eliminar campos innecesarios
                    window.pop("sensor", None)
                    window.pop("segment_index", None)
                    window.pop("window_index", None)
                    window.pop("start_time", None)
                    window.pop("end_time", None)

                    # Agregar el valor del sensor como característica
                    feature_values = list(window.values())
                    feature_values.append(sensor_value)

                    # Agregar las características y el label
                    features.append(feature_values)
                    labels.append(label)

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

    # Convertir a numpy arrays y validar
    features_array = np.array(features, dtype=float)
    labels_array = np.array(labels, dtype=int)

    return features_array, labels_array
def load_features_from_folders(folder_0, folder_1):
    features = []
    labels = []

    # Función para filtrar y limpiar los registros
    def process_file(file_path, label):
        global x
        with open(file_path, 'r') as f:
            data = json.load(f)
            print(file_path)
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
    x=0

    return np.array(features), np.array(labels)


# Función para crear y entrenar la MLP
def train_mlp(X_train, X_test, y_train, y_test, model_path):
    #Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    #Dropout(0.3),
    # Crear el modelo MLP
    '''
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(256, activation='relu', kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(128, activation='relu', kernel_regularizer=l1(0.005), bias_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)),
        Dropout(0.5),

        Dense(1, activation='sigmoid', kernel_regularizer=l1_l2(l1=0.005, l2=0.001))  # Clasificación binaria
    ])
    '''
    #CONFIGURACION RESULTADOS B
    '''
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(256, activation='relu', kernel_regularizer=l2(0.005), bias_regularizer=l2(0.005)),
        Dropout(0.5),
        Dense(128, activation='relu', kernel_regularizer=l1(0.005), bias_regularizer=l2(0.005)),
        Dropout(0.5),
        Dense(64, activation='relu', kernel_regularizer=l2(0.005), bias_regularizer=l2(0.001)),
        Dropout(0.5),

        Dense(1, activation='sigmoid', kernel_regularizer=l1_l2(l1=0.005, l2=0.005))  # Clasificación binaria
    ])
    '''
    '''
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(256, activation='relu', kernel_regularizer=l2(0.005), bias_regularizer=l2(0.005)),
        Dropout(0.5),
        Dense(128, activation='relu', kernel_regularizer=l1(0.005), bias_regularizer=l2(0.005)),
        Dropout(0.5),
        Dense(64, activation='relu', kernel_regularizer=l2(0.005), bias_regularizer=l2(0.001)),
        Dropout(0.5),

        Dense(1, activation='sigmoid', kernel_regularizer=l1_l2(l1=0.005, l2=0.005))  # Clasificación binaria
    ])
    '''
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(256, activation='relu', kernel_regularizer=l2(0.005), bias_regularizer=l2(0.005)),
        Dropout(0.5),
        Dense(128, activation='relu', kernel_regularizer=l1(0.005), bias_regularizer=l2(0.005)),
        Dropout(0.5),
        Dense(64, activation='relu', kernel_regularizer=l2(0.006), bias_regularizer=l2(0.003)),
        Dropout(0.5),

        Dense(1, activation='sigmoid', kernel_regularizer=l1_l2(l1=0.006, l2=0.006))  # Clasificación binaria
    ])
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    checkpointer = ModelCheckpoint(filepath=model_path, verbose=1,
                                   save_best_only=True)
    # Entrenamiento

    X_val, X_final_test, y_val, y_final_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    model.fit(X_train, y_train, epochs=2000, batch_size=32, validation_data=(X_val, y_val), callbacks=[checkpointer])
    #model.fit(X_train, y_train, epochs=250, batch_size=32, callbacks=[checkpointer])
    # Evaluar el modelo en el conjunto de prueba
    model = load_model(model_path)
    test_loss, test_accuracy = model.evaluate(X_final_test, y_final_test)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    y_pred_mlp = model.predict(X_final_test)
    y_pred_mlp = (y_pred_mlp > 0.5).astype(int).flatten()
    #y_pred_classes = np.argmax(y_pred, axis=1)

    # Evaluar la MLP con otras métricas
    evaluate_model(y_final_test, y_pred_mlp, "MLP 2s ambos (Keras)")

    # Guardar el modelo
    #model.save(model_path)
    #print(f"Modelo guardado en {model_path}")

    return model


# Configuración de carpetas
folder_0 = config.p_walking_data_path_no_parkinson  # Reemplaza con la ruta de la carpeta con label 0
folder_1 = config.p_walking_data_path_parkinson  # Reemplaza con la ruta de la carpeta con label 1



# Procesar datos
features, labels = load_features_from_folders(folder_0, folder_1)
folder_2 = config.test_p_walking_data_path_no_parkinson  # Reemplaza con la ruta de la carpeta con label 0
folder_3 = config.test_p_walking_data_path_parkinson  # Reemplaza con la ruta de la carpeta con label 1
print(features.shape)

print("DATOS CARPETAS DE TESTING")
features_test, labels_test = load_features_from_folders(folder_2, folder_3)
print(features_test.shape)

features_total = np.concatenate((features, features_test), axis=0)
labels_total = np.concatenate((labels, labels_test), axis=0)
'''
# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
print(X_train.shape)


# Preprocesar datos (fit en entrenamiento, transform en prueba)
X_train_preprocessed, X_test_preprocessed = preprocess_data(X_train, X_test)


# Entrenar MLP
model = train_mlp(X_train_preprocessed, X_test_preprocessed, y_train, y_test, model_path='MODELS/mlp_model_2s_derecha2.best79.keras')
'''
# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(features_total, labels_total, shuffle=False, test_size=0.36797454931)
print(X_train.shape)
print(X_test.shape)
# Desordenar los datos de entrenamiento
train_indices = np.random.permutation(len(X_train))
X_train_shuffled = X_train[train_indices]
y_train_shuffled = y_train[train_indices]

# Desordenar los datos de prueba
test_indices = np.random.permutation(len(X_test))
X_test_shuffled = X_test[test_indices]
y_test_shuffled = y_test[test_indices]


# Preprocesar datos (fit en entrenamiento, transform en prueba)
X_train_preprocessed, X_test_preprocessed = preprocess_data(X_train_shuffled, X_test_shuffled)

'''
rf = RandomForestClassifier()
rf.fit(X_train_preprocessed, y_train_shuffled)
importances = rf.feature_importances_

# Seleccionar las 20 características más importantes
indices = np.argsort(importances)[-144:]
X_train_reduced = X_train_preprocessed[:, indices]
X_test_reduced = X_test_preprocessed[:, indices]
'''
# Entrenar MLP
model = train_mlp(X_train_preprocessed, X_test_preprocessed, y_train_shuffled, y_test_shuffled, model_path='MODELS/mlp_model_2s_derecha2.bestS2.keras')
#model = train_mlp(X_train_reduced, X_test_reduced, y_train_shuffled, y_test_shuffled, model_path='MODELS/mlp_model_2s_derecha2.bestS2.keras')
