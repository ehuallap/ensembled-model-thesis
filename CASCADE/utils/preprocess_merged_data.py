import os
import numpy as np
import h5py
from time import time
from PIL import Image

splits = ['train', 'test']

# Definir las rutas de las carpetas para 'images' y 'masks' dentro del merged dataset
base_path = '../merged_dataset/'  # Ruta al dataset base (fuera de CASCADE)
images_path = os.path.join(base_path, 'images')
masks_path = os.path.join(base_path, 'masks')
save_path = './data/merged_dataset/'

# Crear directorios para almacenar los datos procesados si no existen
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Rango de intensidades para la normalización
upper = 275
lower = -125

start_time = time()

# Procesar tanto el conjunto de entrenamiento como el de prueba
for split in splits:
    # Crear subdirectorios para 'train' y 'test' si no existen
    split_save_path = os.path.join(save_path, split)
    if not os.path.exists(split_save_path):
        os.makedirs(split_save_path)

    print(f"Procesando {split}...")

    # Asignar las rutas correctas de imágenes y máscaras para entrenamiento y prueba
    ct_path = os.path.join(images_path, split)
    seg_path = os.path.join(masks_path, split)

    # Iterar sobre los archivos en la carpeta de imágenes
    for img_file in os.listdir(ct_path):
        if not img_file.endswith(('.png', '.jpg', '.jpeg')):  # Verificar extensiones válidas
            continue

        img_path = os.path.join(ct_path, img_file)
        mask_path = os.path.join(seg_path, img_file)

        if not os.path.exists(mask_path):
            print(f"Advertencia: No se encuentra máscara para {img_file}. Saltando...")
            continue

        # Cargar la imagen y la máscara
        img = Image.open(img_path)
        mask = Image.open(mask_path)

        # Convertir las imágenes y máscaras a arrays numpy
        img_array = np.array(img, dtype=np.float32)
        mask_array = np.array(mask, dtype=np.uint8)

        # Normalizar las imágenes (ajustar a [0, 1] usando el rango)
        img_array = np.clip(img_array, lower, upper)
        img_array = (img_array - lower) / (upper - lower)

        # Añadir dimensión para imágenes 2D
        img_array = np.expand_dims(img_array, axis=0)

        # Guardar en formato .npz
        base_name = os.path.splitext(img_file)[0]
        np.savez(os.path.join(split_save_path, base_name + '.npz'), image=img_array, label=mask_array)

        print(f"Procesado: {img_file}")

    print(f"Tiempo de procesamiento para {split}: {(time() - start_time) / 60:.2f} minutos")
    print("-----------")
