import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
import Global.config as config


def load_matrices_from_json(folder, label, sensor_name="derecha"):
    """
    Lee archivos JSON de una carpeta, extrae las matrices de las ventanas del sensor especificado y asigna etiquetas.

    Args:
        folder (str): Ruta a la carpeta que contiene los archivos JSON.
        label (int): Etiqueta para las matrices generadas (1 o 0).
        sensor_name (str): Nombre del sensor a procesar.

    Returns:
        matrices (list): Lista de matrices generadas a partir de las ventanas.
        labels (list): Lista de etiquetas correspondientes.
    """
    matrices = []
    labels = []

    for file_name in os.listdir(folder):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder, file_name)

            # Cargar el archivo JSON
            with open(file_path, 'r') as file:
                data = json.load(file)

            if sensor_name not in data:
                continue  # Ignorar si el sensor no está en el archivo

            # Procesar cada segmento del sensor
            for segment in data[sensor_name]:
                for window in segment:
                    # Crear una matriz con las señales a, b, g, x, y, z
                    signals = ['a','b','g','x','y','z']
                    matrix = np.array([[reading[signal] for signal in signals] for reading in window['data']])
                    matrices.append(matrix.flatten())  # Aplanar la matriz para la entrada de la MLP
                    labels.append(label)

    return matrices, labels


def prepare_data(folder_label_1, folder_label_0, sensor_name="derecha"):
    """
    Combina datos de ambas carpetas y genera un conjunto completo de matrices y etiquetas.

    Args:
        folder_label_1 (str): Carpeta que contiene archivos con label 1.
        folder_label_0 (str): Carpeta que contiene archivos con label 0.
        sensor_name (str): Nombre del sensor a procesar.

    Returns:
        matrices (numpy.ndarray): Matrices combinadas.
        labels (numpy.ndarray): Etiquetas correspondientes.
    """
    matrices_1, labels_1 = load_matrices_from_json(folder_label_1, label=1, sensor_name=sensor_name)
    matrices_0, labels_0 = load_matrices_from_json(folder_label_0, label=0, sensor_name=sensor_name)

    matrices = np.array(matrices_1 + matrices_0)
    labels = np.array(labels_1 + labels_0)

    return matrices, labels


def train_mlp(matrices, labels):
    """
    Entrena una red MLP con las matrices y etiquetas proporcionadas.

    Args:
        matrices (numpy.ndarray): Datos de entrada.
        labels (numpy.ndarray): Etiquetas de salida.
    """
    normalizer = MinMaxScaler()
    X_normalized = normalizer.fit_transform(matrices)
    # Preprocesamiento: Escalado
    scaler = StandardScaler()
    matrices_scaled = scaler.fit_transform(X_normalized)

    # División en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(matrices_scaled, labels, test_size=0.3, random_state=42)

    # Construcción del modelo MLP
    model = tf.keras.Sequential([
        #tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=l2(0.01), input_shape=(X_train.shape[1],)),
        #tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=l2(0.01)),  # Capa oculta
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(0.01)),  # Capa oculta
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Capa de salida para clasificación binaria
    ])
    '''
     tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=l2(0.01), input_shape=(X_train.shape[1],)),
     tf.keras.layers.Dropout(0.5),
     tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=l2(0.01)),  # Capa oculta
     tf.keras.layers.Dropout(0.5),
     tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(0.01)),  # Capa oculta
     tf.keras.layers.Dropout(0.5),
     tf.keras.layers.Dense(1, activation='sigmoid')  # Capa de salida para clasificación binaria
     '''

    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    # Entrenamiento
    model.fit(X_train, y_train, epochs=500, batch_size=32, validation_split=0.2)

    # Evaluación
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Pérdida en el conjunto de prueba: {loss:.4f}")
    print(f"Precisión en el conjunto de prueba: {accuracy:.4f}")

    # Guardar el modelo entrenado
    model.save("mlp_sensor_derecha_3.h5")
    print("Modelo guardado como 'mlp_sensor_derecha.h5'")


# Rutas a las carpetas
folder_label_1 = config.p_walking_data_path_parkinson  # Carpeta con archivos JSON con label 1
folder_label_0 = config.p_walking_data_path_no_parkinson  # Carpeta con archivos JSON con label 0

# Preparar datos
matrices0, labels0 = prepare_data(folder_label_1, folder_label_0, sensor_name="derecha")
#matrices1, labels1 = prepare_data(folder_label_1, folder_label_0, sensor_name="izquierda")
#matrices_combined = np.concatenate((matrices0, matrices1), axis=0)
#labels_combined = np.concatenate((labels0, labels1), axis=0)
print(matrices0.shape)
# Entrenar la MLP
train_mlp(matrices0, labels0)
