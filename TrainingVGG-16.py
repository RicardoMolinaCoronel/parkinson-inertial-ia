import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from PIL import Image
from sklearn.model_selection import train_test_split
import Global.config as config
# Configuración general
IMG_SIZE = (224, 224)  # Tamaño de las imágenes compatibles con VGG-16
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001

# Rutas de las carpetas
WAVELET_DIR = config.prep_data_path # Carpeta raíz que contiene las imágenes
LABELS = {
    "PAC-PARKINSON/WAVELETS": 1,
    "CTL-PSANO/WAVELETS": 0
}

# Función para cargar imágenes y combinarlas en 3 canales
def load_wavelet_triplet(wavelet_dir, file_names):
    """
    Carga y combina tres imágenes (b, g, z) en un único tensor de 3 canales.

    Args:
        wavelet_dir (str): Carpeta donde se encuentran las imágenes.
        file_names (dict): Diccionario con los nombres de las imágenes de cada canal (b, g, z).

    Returns:
        np.ndarray: Tensor 3D con los tres canales combinados (formato RGB).
    """
    channels = []
    for channel in ['b', 'g', 'z']:
        file_path = file_names[channel]
        img = Image.open(file_path).resize(IMG_SIZE)  # Asegurarse de que todas las imágenes tienen el mismo tamaño
        img = np.array(img) / 255.0  # Normalizar a [0, 1]
        channels.append(img)

    # Combinar los tres canales
    combined_image = np.stack(channels, axis=-1)  # Combina como (224, 224, 3)
    #print(combined_image.size)
    return combined_image

# Función para preparar los datos
def prepare_data(wavelet_dir, labels):
    """
    Prepara los datos y etiquetas combinando las imágenes de tres canales por ventana.

    Args:
        wavelet_dir (str): Carpeta que contiene las imágenes.
        labels (dict): Diccionario con las clases y sus etiquetas.

    Returns:
        tuple: (X, y) donde X son las imágenes combinadas y y son las etiquetas.
    """
    X = []
    y = []

    for label, class_value in labels.items():
        label_dir = os.path.join(wavelet_dir, label)
        for side in ["derecha", "izquierda"]:
            side_dir = os.path.join(label_dir, side)
            files = os.listdir(side_dir)

            # Agrupar las imágenes por ventana (con base en el nombre del archivo)
            wavelet_groups = {}
            for file_name in files:
                # Extraer el nombre base de la ventana y el canal
                base_name, channel = file_name.rsplit("_", 1)
                channel = channel.split(".")[0]  # Eliminar la extensión
                if base_name not in wavelet_groups:
                    wavelet_groups[base_name] = {}
                wavelet_groups[base_name][channel] = os.path.join(side_dir, file_name)

            # Combinar las imágenes en tres canales
            for base_name, file_names in wavelet_groups.items():
                if all(c in file_names for c in ['b', 'g', 'z']):  # Verificar que están los tres canales
                    combined_image = load_wavelet_triplet(side_dir, file_names)
                    X.append(combined_image)
                    y.append(class_value)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y

# Preparar los datos
X, y = prepare_data(WAVELET_DIR, LABELS)

# Dividir los datos en entrenamiento (70%), validación (20%) y prueba (10%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)  # 70% train, 30% temp
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.333, random_state=42)  # 20% val, 10% test

print(f"Tamaño del conjunto de entrenamiento: {X_train.shape[0]}")
print(f"Tamaño del conjunto de validación: {X_val.shape[0]}")
print(f"Tamaño del conjunto de prueba: {X_test.shape[0]}")

# Crear el modelo VGG-16 preentrenado
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Congelar las capas preentrenadas
for layer in base_model.layers:
    layer.trainable = False

# Añadir capas personalizadas
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="binary_crossentropy", metrics=["accuracy"])

# Entrenar el modelo
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS
)

# Evaluar el modelo en el conjunto de prueba
test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
print(f"Pérdida en el conjunto de prueba: {test_loss}")
print(f"Precisión en el conjunto de prueba: {test_accuracy}")

# Guardar el modelo
model.save("vgg16_parkinson_triplet_model.h5")
print("Modelo guardado como 'vgg16_parkinson_triplet_model.h5'")
