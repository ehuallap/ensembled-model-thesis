import os
import shutil
import random

# Configuración de rutas
base_path = './merged_dataset'  # Ruta principal del dataset
images_path = os.path.join(base_path, 'images')  # Carpeta de imágenes
masks_path = os.path.join(base_path, 'masks')    # Carpeta de máscaras

# Directorios de salida dentro de images y masks
images_train = os.path.join(images_path, 'train')
images_test = os.path.join(images_path, 'test')
images_val = os.path.join(images_path, 'val')

masks_train = os.path.join(masks_path, 'train')
masks_test = os.path.join(masks_path, 'test')
masks_val = os.path.join(masks_path, 'val')

# Crear los directorios si no existen
for folder in [images_train, images_test, images_val, masks_train, masks_test, masks_val]:
    os.makedirs(folder, exist_ok=True)

# Obtener las listas de archivos
image_files = [f for f in os.listdir(images_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
mask_files = [f for f in os.listdir(masks_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Verificar que las imágenes y máscaras coincidan
image_files.sort()
mask_files.sort()

if len(image_files) != len(mask_files):
    raise ValueError("El número de imágenes y máscaras no coincide.")

# Mezclar aleatoriamente las imágenes y máscaras
combined = list(zip(image_files, mask_files))
random.shuffle(combined)

# Calcular los tamaños para train, test y validation
total_files = len(combined)
train_split = int(0.8 * total_files)
test_split = int(0.1 * total_files)
val_split = total_files - train_split - test_split

# Dividir en train, test y validation
train_set = combined[:train_split]
test_set = combined[train_split:train_split + test_split]
val_set = combined[train_split + test_split:]

# Función para mover archivos a su destino
def move_files(file_set, source_images, source_masks, dest_images, dest_masks):
    for img_file, mask_file in file_set:
        # Mover imágenes
        shutil.move(os.path.join(source_images, img_file), os.path.join(dest_images, img_file))
        # Mover máscaras
        shutil.move(os.path.join(source_masks, mask_file), os.path.join(dest_masks, mask_file))

# Mover archivos a las carpetas correspondientes
print("Dividiendo archivos en train, test y val...")
move_files(train_set, images_path, masks_path, images_train, masks_train)
move_files(test_set, images_path, masks_path, images_test, masks_test)
move_files(val_set, images_path, masks_path, images_val, masks_val)

# Mensajes de finalización
print("División completada:")
print(f" - Entrenamiento: {len(train_set)} imágenes")
print(f" - Test: {len(test_set)} imágenes")
print(f" - Validación: {len(val_set)} imágenes")
print("Estructura final:")
print(f" - Imágenes: {images_path}")
print(f" - Máscaras: {masks_path}")
