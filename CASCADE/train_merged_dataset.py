import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from lib.networks import TransCASCADE, PVT_CASCADE
from lib.cnn_vit_backbone import CONFIGS as CONFIGS_ViT_seg
from trainer import train_colon_model  # Esto debe ser actualizado si el trainer se llama diferente

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./data/merged_dataset', help='root dir for data')
parser.add_argument('--volume_path', type=str,
                    default='./data/merged_dataset', help='root dir for validation volume data')
parser.add_argument('--dataset', type=str,
                    default='merged_dataset', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_merged_dataset', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=1, help='output channel of network')  # Sólo una clase para segmentación binaria
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.0001,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=2222, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    # Cambiar la configuración a merged_dataset
    dataset_name = args.dataset
    dataset_config = {
        'merged_dataset': {  # Cambiar aquí el nombre del dataset
            'root_path': args.root_path,
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
            'z_spacing': 1,
        },
    }
    
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    args.exp = 'TransCASCADE_' + dataset_name + str(args.img_size)
    
    snapshot_path = "model_pth/{}/{}".format(args.exp, 'TransCASCADE')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size != 16 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.max_iterations)[0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.img_size)
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    # Configuración del modelo ViT
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    
    # Inicialización del modelo
    net = TransCASCADE(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()  # modelo TransCASCADE
    net.load_from(weights=np.load(config_vit.pretrained_path))  # cargar pesos preentrenados

    # Selección del trainer basado en el dataset
    trainer = {'merged_dataset': train_colon_model,}  # Actualiza esto si tienes un nuevo trainer
    trainer[dataset_name](args, net, snapshot_path)
