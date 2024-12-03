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
import Global.config as config
import joblib
from EvaluationMetrics import evaluate_model

x=0
# Configuración de carpetas
folder_0 = config.ftesting_data_path_no_parkinson  # Reemplaza con la ruta de la carpeta con label 0
folder_1 = config.ftesting_data_path_parkinson # Reemplaza con la ruta de la carpeta con label 1
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

def preprocess_data(X_test):
    # Aplicar StandardScaler
    scaler = joblib.load('MODELS/scaler_mlp.pkl')
    X_test_scaled = scaler.transform(X_test)

    # Aplicar MinMaxScaler
    #normalizer = MinMaxScaler()
    #X_train_scaled= normalizer.fit_transform(X_train)
    #X_test_scaled = normalizer.transform(X_test)
    return X_test_scaled
# Procesar datos
features, labels = load_features_from_folders(folder_0, folder_1)

X_test, X_t, y_test, y_t = train_test_split(features, labels, shuffle=True, test_size=0.9)
print("Shape entrenamiento:", X_t.shape)
X_test_prep = preprocess_data(X_t)


model = load_model('MODELS/mlp_model_2s_derecha2.bestS.keras')
test_loss, test_accuracy = model.evaluate(X_test_prep, y_t)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
y_pred_mlp = model.predict(X_test_prep)
y_pred_mlp = (y_pred_mlp > 0.5).astype(int).flatten()
# y_pred_classes = np.argmax(y_pred, axis=1)

# Evaluar la MLP con otras métricas
evaluate_model(y_t, y_pred_mlp, "MLP 2s ambos (Keras)")