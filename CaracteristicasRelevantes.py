import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import shap
import seaborn as sns
import matplotlib.pyplot as plt
import Global.config as config

# Paso 1: Cargar datos desde las carpetas
def load_imu_data(folder_path, label):
    """
    Cargar datos de una carpeta y asignar una etiqueta a cada archivo.
    :param folder_path: Ruta de la carpeta con archivos JSON.
    :param label: Etiqueta para los datos (1: Parkinson, 0: No Parkinson).
    :return: DataFrame con señales y etiquetas.
    """
    data = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as file:
                file_data = json.load(file)
                for sensor, readings in file_data.items():
                    for reading in readings:
                        data.append({
                            'sensor': sensor,
                            'a': reading['a'],
                            'b': reading['b'],
                            'g': reading['g'],
                            'x': reading['x'],
                            'y': reading['y'],
                            'z': reading['z'],
                            'target': label  # Etiqueta: 1 (Parkinson), 0 (No Parkinson)
                        })
    return pd.DataFrame(data)


# Paso 2: Calcular magnitudes totales
def calculate_magnitudes(data):
    """
    Calcula las magnitudes de aceleración y velocidad angular.
    :param data: DataFrame con las señales.
    :return: DataFrame con columnas adicionales para magnitudes.
    """
    data['magnitude_accel'] = np.sqrt(data['x'] ** 2 + data['y'] ** 2 + data['z'] ** 2)
    data['magnitude_gyro'] = np.sqrt(data['a'] ** 2 + data['b'] ** 2 + data['g'] ** 2)
    return data


# Paso 3: Análisis de importancia de características (Random Forest)
def analyze_feature_importance(data):
    """
    Analiza la importancia de las señales mediante Random Forest.
    :param data: DataFrame con señales y target.
    """
    # Variables independientes y objetivo
    X = data[['a', 'b', 'g', 'x', 'y', 'z', 'magnitude_accel', 'magnitude_gyro']]
    y = data['target']

    # Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Entrenar Random Forest
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Importancia de características
    importances = model.feature_importances_
    features = X.columns

    # Visualizar importancia
    sns.barplot(x=importances, y=features)
    plt.title("Importancia de Características (Random Forest)")
    plt.xlabel("Importancia")
    plt.ylabel("Señales")
    plt.show()


# Paso 4: Reducir dimensionalidad con PCA
def apply_pca(data):
    """
    Aplica PCA para reducir dimensionalidad y encontrar patrones.
    :param data: DataFrame con señales.
    :return: Imprime la varianza explicada por los componentes principales.
    """
    X = data[['a', 'b', 'g', 'x', 'y', 'z', 'magnitude_accel', 'magnitude_gyro']]

    # Aplicar PCA
    pca = PCA(n_components=2)  # Reducir a 2 componentes principales
    principal_components = pca.fit_transform(X)

    print("Varianza explicada por los componentes principales:", pca.explained_variance_ratio_)
    return principal_components


# Paso 5: Explicaciones con SHAP
def explain_with_shap(data):
    """
    Explica el modelo con SHAP para entender el impacto de las señales.
    :param data: DataFrame con señales y target.
    """
    X = data[['a', 'b', 'g', 'x', 'y', 'z', 'magnitude_accel', 'magnitude_gyro']]
    y = data['target']

    # Entrenar modelo
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    # Crear explicador SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Resumen de SHAP
    shap.summary_plot(shap_values[1], X, plot_type="bar")
    shap.summary_plot(shap_values[1], X)

def plot_pca(data, principal_components):
    """
    Graficar los datos en los dos primeros componentes principales.
    :param data: DataFrame con etiquetas (target).
    :param principal_components: Resultados del PCA.
    """
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        principal_components[:, 0], principal_components[:, 1],
        c=data['target'], cmap='coolwarm', alpha=0.7
    )
    plt.colorbar(scatter, label="Target (0: No Parkinson, 1: Parkinson)")
    plt.title("Datos proyectados en los dos primeros componentes principales")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.grid()
    plt.show()


# Paso 6: Ejecutar el análisis
if __name__ == "__main__":
    # Carpetas de datos
    parkinson_folder = config.rdced_walking_data_path_parkinson  # Cambia a la ruta de tu carpeta con Parkinson
    no_parkinson_folder = config.walking_data_path_no_parkinson  # Cambia a la ruta de tu carpeta sin Parkinson

    # Cargar datos y etiquetar
    parkinson_data = load_imu_data(parkinson_folder, label=1)
    no_parkinson_data = load_imu_data(no_parkinson_folder, label=0)

    # Combinar datos
    all_data = pd.concat([parkinson_data, no_parkinson_data], ignore_index=True)

    # Calcular magnitudes
    all_data = calculate_magnitudes(all_data)

    # Analizar importancia de características
    analyze_feature_importance(all_data)

    # Aplicar PCA
    #apply_pca(all_data)

    # Explicar modelo con SHAP
    #explain_with_shap(all_data)

    # Ejecutar la gráfica
    principal_components = apply_pca(all_data)
    plot_pca(all_data, principal_components)