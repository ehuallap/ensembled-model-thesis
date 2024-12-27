import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuración de rutas de datos
data_dir = './merged_dataset'
image_dir = os.path.join(data_dir, 'images')
mask_dir = os.path.join(data_dir, 'masks')

# Configuración de generadores de datos
def create_data_generator(image_path, mask_path, batch_size=32):
    image_datagen = ImageDataGenerator(rescale=1./255)
    mask_datagen = ImageDataGenerator(rescale=1./255)

    image_generator = image_datagen.flow_from_directory(
        image_path,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode=None,
        seed=42
    )

    mask_generator = mask_datagen.flow_from_directory(
        mask_path,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode=None,
        seed=42
    )

    return zip(image_generator, mask_generator)

# Cargar modelo base desde archivos .npz
def load_npz_model(filepath, input_shape):
    data = np.load(filepath)
    model = Sequential([  # Este modelo es solo para cargar los pesos
        Dense(64, activation='relu', input_shape=input_shape),
        Dense(1, activation='sigmoid')
    ])
    model.set_weights([data['weights_0'], data['weights_1']])
    return model

# Parte 1: Entrenamiento
def train_meta_model():
    train_generator = create_data_generator(
        os.path.join(image_dir, 'train'),
        os.path.join(mask_dir, 'train')
    )

    # Carga de los modelos base desde .npz
    model1 = load_npz_model('cascade_trained.npz', (128,))
    model2 = load_npz_model('meganet_trained.npz', (128,))
    model3 = load_npz_model('transnetr_trained.npz', (128,))

    # Generar predicciones de los modelos base y etiquetas reales
    stacked_train = []
    y_train = []
    for images, masks in train_generator:
        pred1 = model1.predict(images)
        pred2 = model2.predict(images)
        pred3 = model3.predict(images)
        stacked_train.append(np.concatenate([pred1, pred2, pred3], axis=1))
        y_train.append(masks)

        if len(stacked_train) >= 1000 // 32:  # Limitar a 1000 muestras para este ejemplo
            break

    stacked_train = np.vstack(stacked_train)
    y_train = np.vstack(y_train).reshape(-1, 1)

    # Construir el metamodelo (red neuronal simple)
    meta_model = Sequential([
        Dense(64, activation='relu', input_shape=(stacked_train.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    meta_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Entrenar el metamodelo
    meta_model.fit(stacked_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    # Guardar el metamodelo
    meta_model.save('ensembled_trained.h5')

# Ejecutar
if __name__ == "__main__":
    print("Entrenando el metamodelo...")
    train_meta_model()
