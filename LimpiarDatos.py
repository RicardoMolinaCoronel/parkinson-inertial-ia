import json
import os

# Definir la ruta del archivo JSON
directorio = 'RAW_DATA/CTL-PSANO'  # Reemplaza con el nombre de la carpeta donde está el archivo
archivo_json = os.path.join(directorio, 'datos_filtrados.json')  # Reemplaza con el nombre del archivo

# Cargar los datos desde el archivo JSON
with open(archivo_json, 'r') as file:
    data_json = json.load(file)

# Función para filtrar registros
def filtrar_registros(sensor_data):
    return [
        entry for entry in sensor_data
        if all(key in entry for key in ["millis", "t", "x", "y", "z", "a", "b", "g"])
    ]

# Aplicar el filtro a ambos sensores
data_filtrada = {
    "derecha": filtrar_registros(data_json.get("derecha", [])),
    "izquierda": filtrar_registros(data_json.get("izquierda", []))
}

# Guardar el archivo filtrado en un nuevo archivo JSON
archivo_filtrado = os.path.join(directorio, 'archivo_filtrado.json')
with open(archivo_filtrado, 'w') as file:
    json.dump(data_filtrada, file, indent=4)

print(f"Archivo filtrado guardado en: {archivo_filtrado}")