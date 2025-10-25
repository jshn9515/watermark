import argparse
import multiprocessing
import os

import lightning as pl
import torch.utils.data as utils
from datasets import load_dataset
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from models import StegaStampModule
from utils import parse_resolution

os.environ['WANDB_API_KEY'] = 'your-api-key'  # Replace with your actual Wandb API key

parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset',
    type=str,
    required=True,
    help='Name of the dataset (e.g., CIFAR10, MNIST, LSUN).',
)
parser.add_argument(
    '--input_dir',
    type=str,
    required=True,
    help='Directory containing training images.',
)
parser.add_argument(
    '--output_dir',
    type=str,
    required=True,
    help='Directory to save results to.',
)
parser.add_argument(
    '--resolution',
    type=parse_resolution,
    default=32,
    help='Image resolution as int or tuple (height, width). Default: 32.',
)
parser.add_argument(
    '--mode',
    type=str,
    choices=['resize', 'crop'],
    default='resize',
    help='Image preprocessing mode: "resize" or "crop". Default: "resize".',
)
parser.add_argument(
    '--bit_length',
    type=int,
    default=64,
    help='Number of bits in the fingerprint. Default: 64.',
)
parser.add_argument(
    '--num_epochs',
    type=int,
    default=20,
    help='Number of training epochs. Default: 20.',
)
parser.add_argument(
    '--batch_size',
    type=int,
    default=64,
    help='Batch size. Default: 64.',
)
parser.add_argument(
    '--lr',
    type=float,
    default=0.0001,
    help='Learning rate. Default: 0.0001.',
)
parser.add_argument(
    '--accelerator',
    type=str,
    default='cpu',
    choices=['cpu', 'gpu'],
    help='Device to use. Default: "cpu".',
)
parser.add_argument(
    '--l2_loss_await',
    type=int,
    default=1000,
    help='Train without L2 loss for the first x iterations. Default: 1000.',
)
parser.add_argument(
    '--l2_loss_weight',
    type=float,
    default=10,
    help='L2 loss weight for image fidelity. Default: 10.',
)
parser.add_argument(
    '--l2_loss_ramp',
    type=int,
    default=3000,
    help='Linearly increase L2 loss weight over x iterations. Default: 3000.',
)
parser.add_argument(
    '--bce_loss_weight',
    type=float,
    default=1,
    help='BCE loss weight for fingerprint reconstruction. Default: 1.',
)
args = parser.parse_args()

if __name__ == '__main__':
    dataset = load_dataset(
        name=args.dataset,
        root=args.input_dir,
        resolution=args.resolution,
        mode=args.mode,
    )
    dataloader = utils.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=multiprocessing.cpu_count(),
    )

    net = StegaStampModule(args)
    logger = WandbLogger(
        project='your-project',  # Replace with your Wandb project name
        save_dir=args.output_dir,
        log_model=True,
        checkpoint_name='stegastamp',
        config=vars(args),
    )
    checkpoints = ModelCheckpoint(
        dirpath=args.output_dir,
        filename='stegastamp',
        monitor='loss',
        mode='min',
        save_on_exception=True,
    )
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        max_epochs=args.num_epochs,
        logger=logger,
        callbacks=checkpoints,
        deterministic=True,
        benchmark=False,
    )
    trainer.fit(net, dataloader)
