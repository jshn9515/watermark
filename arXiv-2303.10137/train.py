import multiprocessing as mp
import os
import sys
from typing import Any

import lightning as pl
import rich_click as click
import torch.utils.data as utils
from datastore import load_dataset
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from models import StegaStampModule
from utils import AttributeDict, IntOrTuple

os.environ['WANDB_API_KEY'] = 'your-api-key'  # Replace with your actual Wandb API key


@click.command()
@click.option(
    '--dataset',
    type=str,
    required=True,
    help='Name of the dataset (e.g., CIFAR10, MNIST, LSUN).',
)
@click.option(
    '--input_dir',
    type=click.Path(),
    required=True,
    help='Directory containing training images.',
)
@click.option(
    '--output_dir',
    type=click.Path(),
    required=True,
    help='Directory to save results to.',
)
@click.option(
    '--resolution',
    type=IntOrTuple(),
    default=32,
    help='Image resolution as int or tuple (height, width). Default: 32.',
    show_default=True,
)
@click.option(
    '--mode',
    type=click.Choice(['resize', 'crop']),
    default='resize',
    help='Image preprocessing mode: "resize" or "crop".',
    show_default=True,
)
@click.option(
    '--bit_length',
    type=int,
    default=64,
    help='Number of bits in the fingerprint.',
    show_default=True,
)
@click.option(
    '--num_epochs',
    type=int,
    default=20,
    help='Number of training epochs.',
    show_default=True,
)
@click.option(
    '--batch_size',
    type=int,
    default=64,
    help='Batch size.',
    show_default=True,
)
@click.option(
    '--lr',
    type=float,
    default=0.0001,
    help='Learning rate.',
    show_default=True,
)
@click.option(
    '--accelerator',
    type=click.Choice(['cpu', 'gpu']),
    default='cpu',
    help='Device to use.',
    show_default=True,
)
@click.option(
    '--l2_loss_await',
    type=int,
    default=1000,
    help='Train without L2 loss for the first x iterations.',
    show_default=True,
)
@click.option(
    '--l2_loss_weight',
    type=float,
    default=10,
    help='L2 loss weight for image fidelity.',
    show_default=True,
)
@click.option(
    '--l2_loss_ramp',
    type=int,
    default=3000,
    help='Linearly increase L2 loss weight over x iterations.',
    show_default=True,
)
@click.option(
    '--bce_loss_weight',
    type=float,
    default=1,
    help='BCE loss weight for fingerprint reconstruction.',
    show_default=True,
)
def main(**args: dict[str, Any]) -> None:
    args = AttributeDict(args)
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
        num_workers=mp.cpu_count() if sys.platform != 'win32' else 0,
        pin_memory=True if args.accelerator != 'cpu' else False,
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


if __name__ == '__main__':
    click.rich_click.THEME = 'nord-box'
    main()
