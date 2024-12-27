import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import ndimage
from scipy.ndimage.interpolation import zoom


def random_rot_flip(image, label):
    """Realiza una rotación aleatoria (0, 90, 180, 270) y un flip aleatorio."""
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    """Realiza una rotación aleatoria entre -20 y 20 grados."""
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=1, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    """
    Clase para aplicar transformaciones aleatorias, como:
    - Rotación y flip aleatorio.
    - Redimensionamiento a un tamaño especificado.
    """
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # Aplicar transformaciones aleatorias
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        # Redimensionamiento
        x, y = image.shape[:2]
        if x != self.output_size[0] or y != self.output_size[1]:
            if len(image.shape) == 3:  # Imagen RGB (3D)
                image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=1)  # Interpolación bilineal
            else:  # Escala de grises (2D)
                image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=1)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        # Convertir a tensor
        if len(image.shape) == 3:  # Imagen RGB
            image = torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32))  # (C, H, W)
        else:  # Imagen en escala de grises
            image = torch.from_numpy(image[None, :, :].astype(np.float32))  # (1, H, W)
        
        label = torch.from_numpy(label.astype(np.float32)).long()  # (H, W)

        return {'image': image, 'label': label}


class ColonoscopyDataset(Dataset):
    """
    Dataset para cargar imágenes y máscaras en formato .npz.
    Se espera que cada archivo .npz contenga dos claves:
        - 'image': La imagen en formato (H, W, C) o escala de grises.
        - 'label': La máscara correspondiente en formato (H, W).
    """
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Directorio raíz donde se encuentran los archivos .npz.
            split (str): División de datos, puede ser 'train' o 'val'.
            transform (callable, optional): Transformaciones a aplicar en los datos.
        """
        self.data_dir = os.path.join(root_dir, split)  # Ej: root_dir/train
        self.file_list = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.npz')])
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Cargar archivo .npz
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        data = np.load(file_path)

        # Extraer imagen y máscara
        image = data['image']  # (H, W, C) o (H, W)
        label = data['label']  # (H, W)

        # Asegurar que la imagen tenga 3 canales (RGB) si está en escala de grises
        if len(image.shape) == 2:  # Si la imagen es en escala de grises
            image = np.expand_dims(image, axis=-1)  # (H, W, 1)
            image = np.repeat(image, 3, axis=-1)    # Convertir a (H, W, 3)

        sample = {'image': image, 'label': label}

        # Aplicar transformaciones si están definidas
        if self.transform:
            sample = self.transform(sample)

        return sample
