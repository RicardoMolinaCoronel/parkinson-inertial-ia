def transform_txt(input_file, output_file):
    """
    Transforma un archivo de texto de la estructura original a la estructura deseada.
    :param input_file: Ruta del archivo de entrada.
    :param output_file: Ruta del archivo de salida.
    """
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        lines = infile.readlines()

        for line in lines:
            line = line.strip()
            if ',' not in line:
                # Es un identificador UUID
                if line:
                    outfile.write(f"{line}\n")
            elif line:
                # Es una línea de rangos
                ranges = line.split(', ')
                for r in ranges:
                    outfile.write(f"{r.strip()}\n")
                outfile.write("\n")  # Agregar una línea en blanco entre bloques


# Rutas de los archivos
input_file = "../NORMALIZED_DATA/segmentar.txt"  # Cambia esto por la ruta de tu archivo de entrada
output_file = "../NORMALIZED_DATA/segmentar_1.txt"  # Cambia esto por la ruta de tu archivo de salida

# Ejecutar la transformación
transform_txt(input_file, output_file)

print(f"Archivo transformado guardado en {output_file}")
