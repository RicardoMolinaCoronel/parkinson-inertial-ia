import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
import Global.config as config
import json
def load_windows_from_folder(folder_path, label, sensor="derecha"):
    """
    Carga ventanas de una carpeta, filtra por sensor y asigna etiquetas.

    Args:
        folder_path (str): Ruta de la carpeta con los archivos JSON.
        label (int): Etiqueta (1 para Parkinson, 0 para no Parkinson).
        sensor (str): Sensor a filtrar ("derecha").

    Returns:
        list: Lista de ventanas con sus etiquetas.
    """
    data = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as f:
                json_data = json.load(f)

            if sensor in json_data:
                for segment in json_data[sensor]:
                    for window in segment:
                        df = pd.DataFrame(window['data'])
                        # Seleccionar columnas relevantes y descartar otras
                        features = df[['a', 'b', 'g', 'x', 'y', 'z']].values
                        data.append((features, label))
    return data


# Cargar datos de ambas carpetas
parkinson_data = load_windows_from_folder(config.p_walking_data_path_parkinson, label=1)
non_parkinson_data = load_windows_from_folder(config.p_walking_data_path_no_parkinson, label=0)
test_parkinson_data = load_windows_from_folder(config.test_p_walking_data_path_parkinson, label=1)
test_non_parkinson_data = load_windows_from_folder(config.test_p_walking_data_path_no_parkinson, label=0)
# Combinar y barajar
all_training_data = parkinson_data + non_parkinson_data
all_test_data = test_parkinson_data + test_non_parkinson_data
np.random.shuffle(all_training_data)
np.random.shuffle(all_test_data)
print(len(all_training_data))
print(len(all_test_data))
print(len(all_test_data)+len(all_training_data))
print(len(all_test_data)/(len(all_test_data)+len(all_training_data)))
all_data = all_training_data+all_test_data
# Separar características y etiquetas
X, y = zip(*all_data)
X = np.array(X)  # Convertir a array numpy (n_samples, n_timesteps, n_features)
y = np.array(y)

# Normalizar características
scaler = StandardScaler()
X = np.array([scaler.fit_transform(x) for x in X])  # Escalar cada ventana

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3075291622481442, shuffle=False)
X_val, X_test_final, y_val, y_test_final = train_test_split(X_test, y_test, test_size=0.5, shuffle=True)

'''
model = Sequential([

    LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]),
         return_sequences=True,
         kernel_regularizer=l2(0.001),
         dropout=0.3,
         ),
    BatchNormalization(),
    LSTM(64, return_sequences=False,
             dropout=0.3,
             kernel_regularizer=l2(0.001)),
        BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binaria: Parkinson (1) o no Parkinson (0)

    ])
'''

model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]),
         return_sequences=False,
         ),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Binaria: Parkinson (1) o no Parkinson (0)

    ])


checkpointer = ModelCheckpoint(filepath='MODELS/LSTMV1-2.keras', verbose=1,
                               save_best_only=True)
model.compile(optimizer= 'adam', loss='binary_crossentropy', metrics=['accuracy'])
#early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=16, callbacks=[checkpointer])
# Evaluar en el conjunto de prueba
model = load_model('MODELS/LSTMV1-2.keras')
test_loss, test_accuracy = model.evaluate(X_test_final, y_test_final)
print(f"Precisión en prueba: {test_accuracy * 100:.2f}%")
