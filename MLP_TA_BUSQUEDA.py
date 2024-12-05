import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.regularizers import l1, l2, l1_l2
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV

# 1. Definir el modelo como una función para adaptarlo a la búsqueda
def create_model(
    learning_rate=0.0001,
    dropout_rate=0.5,
    neurons_1=256,
    neurons_2=128,
    neurons_3=64,
    kernel_reg_1=l2(0.005),
    kernel_reg_2=l1(0.005),
    kernel_reg_3=l2(0.006),
    bias_reg_1=l2(0.005),
    bias_reg_2=l2(0.005),
    bias_reg_3=l2(0.001),
    final_reg=l1_l2(l1=0.005, l2=0.005),
    optimizer="adam",
):
    if optimizer == "adam":
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == "rmsprop":
        opt = RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(neurons_1, activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=kernel_reg_1, bias_regularizer=bias_reg_1),
        Dropout(dropout_rate),
        Dense(neurons_2, activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=kernel_reg_2, bias_regularizer=bias_reg_2),
        Dropout(dropout_rate),
        Dense(neurons_3, activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=kernel_reg_3, bias_regularizer=bias_reg_3),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid', kernel_regularizer=final_reg),
    ])
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 2. Crear el wrapper para el modelo
model = KerasClassifier(model=create_model, verbose=0)

# 3. Definir el espacio de búsqueda para los hiperparámetros
param_grid = {
    'model__learning_rate': [0.0001, 0.001],
    'model__dropout_rate': [0.3, 0.5],
    'model__neurons_1': [128, 256],
    'model__neurons_2': [64, 128],
    'model__neurons_3': [32, 64],
    'model__kernel_reg_1': [l2(0.001), l2(0.005)],
    'model__kernel_reg_2': [l1(0.001), l1(0.005)],
    'model__kernel_reg_3': [l2(0.003), l2(0.006)],
    'model__optimizer': ['adam', 'rmsprop'],
    'batch_size': [16, 32],
    'epochs': [10, 20],
}

# 4. Configurar la búsqueda con GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=2, n_jobs=-1)

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


# 5. Ejecutar la búsqueda
grid_result = grid.fit(X_train, y_train)

# 6. Imprimir resultados
print(f"Mejor puntuación: {grid_result.best_score_} usando {grid_result.best_params_}")
