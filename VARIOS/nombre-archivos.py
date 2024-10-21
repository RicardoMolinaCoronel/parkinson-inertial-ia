import os

# Especifica el directorio donde están los archivos
directorio = 'C:/Users/Ricardo/PycharmProjects/TESIS/NORMALIZED_DATA/normalizadoSANO'

# Especifica la ruta donde quieres guardar el archivo con los nombres
archivo_salida = 'no_parkinson.txt'

# Obtener la lista de archivos en el directorio
archivos = os.listdir(directorio)

# Filtrar para obtener solo los archivos (y no carpetas)
archivos = [f for f in archivos if os.path.isfile(os.path.join(directorio, f))]

# Obtener los primeros 128 archivos
archivos = archivos[-11:]

# Escribir los nombres de archivos en el archivo de salida
with open(archivo_salida, 'w') as salida:
    for archivo in archivos:
        salida.write(archivo + '\n')

print(f'Se han escrito {len(archivos)} nombres de archivo en {archivo_salida}.')