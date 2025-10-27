import csv
import multiprocessing as mp
import pathlib
import sys
from typing import Any

import rich_click as click
import torch
import torch.utils.data as utils
import tqdm
from datastore import load_dataset
from models import StegaStampModule
from utils import AttributeDict, IntOrTuple, generate_random_fingerprints


@click.command()
@click.option(
    '--checkpoint_path',
    type=click.Path(exists=True),
    required=True,
    help='Path to trained StegaStamp encoder/decoder.',
)
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
    help='Path to save watermarked images to.',
)
@click.option(
    '--resolution',
    type=IntOrTuple(),
    default=32,
    help='Image resolution as int or tuple (height, width). Default: 32.',
)
@click.option(
    '--mode',
    type=click.Choice(['resize', 'crop']),
    default='resize',
    help='Image preprocessing mode: "resize" or "crop". Default: "resize".',
)
@click.option(
    '--identical_fingerprints',
    is_flag=True,
    help='If this option is provided use identical fingerprints. Otherwise sample arbitrary fingerprints. Default: False.',
)
@click.option(
    '--batch_size',
    type=int,
    default=64,
    help='Batch size. Default: 64.',
)
@click.option(
    '--device',
    type=click.Choice(['cpu', 'cuda', 'xpu']),
    default='cpu',
    help='Device to use. Default: cpu.',
)
@click.option(
    '--seed',
    type=int,
    default=42,
    help='Random seed to sample fingerprints. Default: 42.',
)
def main(**args: dict[str, Any]) -> None:
    args = AttributeDict(args)
    net = StegaStampModule.load_from_checkpoint(
        args.checkpoint_path,
        map_location=args.device,
    )

    all_fingerprints = []
    bitwise_accuracy = 0

    args.bit_length = net.hparams['bit_length']
    gt_fingerprints = generate_random_fingerprints(args.bit_length, 1)
    gt_fingerprints = torch.squeeze(gt_fingerprints)
    fingerprint_size = len(gt_fingerprints)
    z = torch.zeros(args.batch_size, fingerprint_size, dtype=torch.float)
    for i, bit in enumerate(gt_fingerprints):
        z[:, i] = bit.item()
    z = z.to(args.device)

    dataset = load_dataset(
        name=args.dataset,
        root=args.input_dir,
        resolution=args.resolution,
        mode=args.mode,
    )
    dataloader = utils.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=mp.cpu_count() if sys.platform != 'win32' else 0,
        pin_memory=True if args.device != 'cpu' else False,
    )

    for images, _ in tqdm.tqdm(
        dataloader,
        unit='images',
        desc='Detecting fingerprints',
    ):
        images = torch.as_tensor(images, device=args.device)
        fingerprints = net.decoder(images)
        fingerprints = torch.as_tensor(fingerprints > 0, dtype=torch.long)

        size = images.size(0)
        correct = fingerprints[:size] == z[:size]
        mean = torch.mean(correct.float(), dim=1)
        bitwise_accuracy += torch.sum(mean).item()

        all_fingerprints.append(fingerprints.cpu())

    all_fingerprints = torch.concat(all_fingerprints, dim=0).cpu()
    bitwise_accuracy = bitwise_accuracy / len(all_fingerprints)

    # non-corrected
    print(f'Bitwise accuracy on fingerprinted images: {bitwise_accuracy}')

    # write in file
    path = pathlib.Path(args.output_dir)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / 'detected_fingerprints.csv', 'w', newline='') as fp:
        fieldnames = ['filename', 'fingerprint']
        writer = csv.DictWriter(fp, fieldnames)
        writer.writeheader()
        for idx in range(len(all_fingerprints)):
            fingerprint = all_fingerprints[idx]
            fingerprint = ''.join([str(bit.item()) for bit in fingerprint])
            filename = str(idx + 1) + '.png'
            writer.writerow({'filename': filename, 'fingerprint': fingerprint})


if __name__ == '__main__':
    click.rich_click.THEME = 'nord-box'
    main()
