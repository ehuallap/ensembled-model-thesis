import torch

# Verificar si CUDA está disponible
if torch.cuda.is_available():
    print("CUDA está disponible.")
    print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
    print(f"Número de GPUs disponibles: {torch.cuda.device_count()}")
else:
    print("CUDA no está disponible. PyTorch no detecta una GPU.")
