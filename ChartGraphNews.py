import json
import matplotlib.pyplot as plt
import os

# Definir la ruta del archivo JSON
directorio = 'RAW_DATA/CTL-PSANO'  # Reemplaza con el nombre de la carpeta donde está el archivo
archivo_json = os.path.join(directorio, 'archivo_filtrado.json')  # Reemplaza con el nombre del archivo

# Cargar los datos desde el archivo JSON
with open(archivo_json, 'r') as file:
    data_json = json.load(file)

# Extraer los datos para "derecha" y "izquierda"
def extract_data(sensor_data):
    t_values = [entry["t"] for entry in sensor_data]
    x_values = [entry["x"] for entry in sensor_data]
    y_values = [entry["y"] for entry in sensor_data]
    z_values = [entry["z"] for entry in sensor_data]
    a_values = [entry["a"] for entry in sensor_data]
    b_values = [entry["b"] for entry in sensor_data]
    g_values = [entry["g"] for entry in sensor_data]
    return t_values, x_values, y_values, z_values, a_values, b_values, g_values

# Obtener los datos para ambos sensores
t_derecha, x_derecha, y_derecha, z_derecha, a_derecha, b_derecha, g_derecha = extract_data(data_json["derecha"])
t_izquierda, x_izquierda, y_izquierda, z_izquierda, a_izquierda, b_izquierda, g_izquierda = extract_data(data_json["izquierda"])

# Graficar cada eje (x, y, z, a, b, g) respecto a t para ambos sensores
plt.figure(figsize=(14, 16))

# Eje X
plt.subplot(6, 1, 1)
plt.plot(t_derecha, x_derecha, label='Derecha X')
plt.plot(t_izquierda, x_izquierda, label='Izquierda X', linestyle='--')
plt.title('Eje X vs Tiempo')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Valor de X')
plt.legend()

# Eje Y
plt.subplot(6, 1, 2)
plt.plot(t_derecha, y_derecha, label='Derecha Y')
plt.plot(t_izquierda, y_izquierda, label='Izquierda Y', linestyle='--')
plt.title('Eje Y vs Tiempo')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Valor de Y')
plt.legend()

# Eje Z
plt.subplot(6, 1, 3)
plt.plot(t_derecha, z_derecha, label='Derecha Z')
plt.plot(t_izquierda, z_izquierda, label='Izquierda Z', linestyle='--')
plt.title('Eje Z vs Tiempo')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Valor de Z')
plt.legend()

# Eje A
plt.subplot(6, 1, 4)
plt.plot(t_derecha, a_derecha, label='Derecha A')
plt.plot(t_izquierda, a_izquierda, label='Izquierda A', linestyle='--')
plt.title('Eje A vs Tiempo')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Valor de A')
plt.legend()

# Eje B
plt.subplot(6, 1, 5)
plt.plot(t_derecha, b_derecha, label='Derecha B')
plt.plot(t_izquierda, b_izquierda, label='Izquierda B', linestyle='--')
plt.title('Eje B vs Tiempo')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Valor de B')
plt.legend()

# Eje G
plt.subplot(6, 1, 6)
plt.plot(t_derecha, g_derecha, label='Derecha G')
plt.plot(t_izquierda, g_izquierda, label='Izquierda G', linestyle='--')
plt.title('Eje G vs Tiempo')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Valor de G')
plt.legend()

plt.tight_layout()
plt.show()
