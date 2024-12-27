import argparse
import logging
import os
import random
import sys
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.dataset_custom import ColonoscopyDataset, RandomGenerator
from utils.utils import DiceLoss

def inference(args, model, best_performance):
    test_dataset = ColonoscopyDataset(args.root_path, split='test', transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    logging.info("Number of test iterations: {}".format(len(test_loader)))
    
    model.eval()
    dice_score = 0.0
    for image, mask in tqdm(test_loader):
        image, mask = image.cuda(), mask.cuda()
        with torch.no_grad():
            outputs = model(image)
            preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            dice_score += DiceLoss.dice_coeff(preds, mask)
    
    mean_dice = dice_score / len(test_loader)
    logging.info(f"Mean Dice Score: {mean_dice:.4f}, Best Dice: {best_performance:.4f}")
    return mean_dice

def train_colon_model(args, model, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/train.log", level=logging.INFO,
                        format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # Parámetros
    batch_size = args.batch_size
    lr = args.base_lr
    max_epochs = args.max_epochs

    # Dataset y DataLoader
    train_dataset = ColonoscopyDataset(args.root_path, split='train',
                                    transform=transforms.Compose([
                                        RandomGenerator(output_size=[args.img_size, args.img_size]),
                                        transforms.ToTensor()
                                    ]))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Configuración del modelo
    model = model.cuda()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(n_classes=2)

    writer = SummaryWriter(snapshot_path + '/logs')
    best_performance = 0.0
    iter_num = 0

    # Bucle de entrenamiento
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}"):
            images, masks = images.cuda(), masks.cuda()

            outputs = model(images)
            loss_ce = ce_loss(outputs, masks.long())
            loss_dice = dice_loss(outputs, masks, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            iter_num += 1

            writer.add_scalar('train/loss', loss.item(), iter_num)

        logging.info(f"Epoch {epoch+1} Loss: {epoch_loss/len(train_loader):.4f}")

        # Validación después de cada epoch
        performance = inference(args, model, best_performance)
        if performance > best_performance:
            best_performance = performance
            torch.save(model.state_dict(), os.path.join(snapshot_path, 'best_model.pth'))
            logging.info("Model saved as best_model.pth")

    writer.close()
    logging.info("Training Completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data", help="Path to dataset folder")
    parser.add_argument("--snapshot_path", type=str, default="./output", help="Path to save model and logs")
    parser.add_argument("--img_size", type=int, default=256, help="Input image size")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--base_lr", type=float, default=1e-4, help="Base learning rate")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of epochs")
    args = parser.parse_args()

    os.makedirs(args.snapshot_path, exist_ok=True)
    from models.unet import UNet  # Importa tu modelo aquí

    model = UNet(in_channels=3, out_channels=2)  # Ejemplo para U-Net, 2 clases (fondo y pólipo)
    print(args)
    train_colon_model(args, model, args.snapshot_path)
