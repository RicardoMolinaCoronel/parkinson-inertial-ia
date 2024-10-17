import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import Global.config as config

# Función para calcular las medias de a, b, g, x, y, z en una sección
def calcular_media(seccion):
    # Extraer valores de 'a', 'b', 'g', 'x', 'y', 'z'
    valores_a = [item['a'] for item in seccion]
    valores_b = [item['b'] for item in seccion]
    valores_g = [item['g'] for item in seccion]
    valores_x = [item['x'] for item in seccion]
    valores_y = [item['y'] for item in seccion]
    valores_z = [item['z'] for item in seccion]

    # Calcular las medias
    media_a = np.mean(valores_a)
    media_b = np.mean(valores_b)
    media_g = np.mean(valores_g)
    media_x = np.mean(valores_x)
    media_y = np.mean(valores_y)
    media_z = np.mean(valores_z)

    return media_a, media_b, media_g, media_x, media_y, media_z


# Función para procesar cada archivo JSON
def procesar_archivo(json_file):
    with open(json_file, 'r') as archivo:
        datos = json.load(archivo)

    # Calcular las medias para 'derecha', 'izquierda' y 'espina'
    medias = {}

    for seccion in ['derecha', 'izquierda', 'espina_base']:
        if seccion in datos:
            medias[seccion] = calcular_media(datos[seccion])
        else:
            medias[seccion] = (None, None, None, None, None, None)  # Si no existe la sección

    return medias


# Directorio donde están los archivos JSON de los pacientes
directorio = config.raw_data_path_parkinson

# Diccionarios para almacenar las medias de cada variable por sección
medias_derecha = {'a': [], 'b': [], 'g': [], 'x': [], 'y': [], 'z': []}
medias_izquierda = {'a': [], 'b': [], 'g': [], 'x': [], 'y': [], 'z': []}
medias_espina = {'a': [], 'b': [], 'g': [], 'x': [], 'y': [], 'z': []}

# Iterar sobre todos los archivos JSON en el directorio y almacenar las medias
for filename in os.listdir(directorio):
    if filename.endswith('.json'):
        filepath = os.path.join(directorio, filename)
        medias = procesar_archivo(filepath)

        # Almacenar las medias en las listas correspondientes
        for seccion, valores in medias.items():
            if seccion == 'derecha':
                medias_derecha['a'].append(valores[0])
                medias_derecha['b'].append(valores[1])
                medias_derecha['g'].append(valores[2])
                medias_derecha['x'].append(valores[3])
                medias_derecha['y'].append(valores[4])
                medias_derecha['z'].append(valores[5])
            elif seccion == 'izquierda':
                medias_izquierda['a'].append(valores[0])
                medias_izquierda['b'].append(valores[1])
                medias_izquierda['g'].append(valores[2])
                medias_izquierda['x'].append(valores[3])
                medias_izquierda['y'].append(valores[4])
                medias_izquierda['z'].append(valores[5])
            elif seccion == 'espina_base':
                medias_espina['a'].append(valores[0])
                medias_espina['b'].append(valores[1])
                medias_espina['g'].append(valores[2])
                medias_espina['x'].append(valores[3])
                medias_espina['y'].append(valores[4])
                medias_espina['z'].append(valores[5])

# Crear diagramas de cajas para cada variable en las secciones 'derecha', 'izquierda', y 'espina'

variables = ['a', 'b', 'g', 'x', 'y', 'z']

# Crear los datos para los diagramas de cajas
datos_derecha = [medias_derecha[var] for var in variables]
datos_izquierda = [medias_izquierda[var] for var in variables]
datos_espina = [medias_espina[var] for var in variables]

# Gráfico de caja para las medias de la sección 'derecha'
plt.figure(figsize=(10, 6))
sns.boxplot(data=datos_derecha)
plt.title('Diagramas de caja para sección "derecha"')
plt.xticks(ticks=range(6), labels=variables)
plt.ylabel('Valor de la media')
plt.show()

# Gráfico de caja para las medias de la sección 'izquierda'
plt.figure(figsize=(10, 6))
sns.boxplot(data=datos_izquierda)
plt.title('Diagramas de caja para sección "izquierda"')
plt.xticks(ticks=range(6), labels=variables)
plt.ylabel('Valor de la media')
plt.show()

# Gráfico de caja para las medias de la sección 'espina'
plt.figure(figsize=(10, 6))
sns.boxplot(data=datos_espina)
plt.title('Diagramas de caja para sección "espina"')
plt.xticks(ticks=range(6), labels=variables)
plt.ylabel('Valor de la media')
plt.show()


