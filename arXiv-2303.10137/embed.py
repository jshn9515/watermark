import csv
import io
import multiprocessing as mp
import pathlib
import sys
import tarfile
from typing import Any

import rich_click as click
import torch
import torch.utils.data as utils
import torchvision.transforms.v2.functional as F
import tqdm
from datastore import load_dataset
from models import StegaStampModule
from torchvision.utils import make_grid, save_image
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
    help='Image resolution as int or tuple (height, width).',
    show_default=True,
)
@click.option(
    '--mode',
    type=click.Choice(['resize', 'crop']),
    default='resize',
    help='Image preprocessing mode: "resize" or "crop". Default: "resize".',
    show_default=True,
)
@click.option(
    '--identical_fingerprints',
    is_flag=True,
    help='If this option is provided use identical fingerprints. Otherwise sample arbitrary fingerprints.',
)
@click.option(
    '--check',
    is_flag=True,
    help='Validate fingerprint detection accuracy.',
)
@click.option(
    '--batch_size',
    type=int,
    default=64,
    help='Batch size.',
    show_default=True,
)
@click.option(
    '--device',
    type=click.Choice(['cpu', 'cuda', 'xpu']),
    default='cpu',
    help='Device to use.',
    show_default=True,
)
@click.option(
    '--seed',
    type=int,
    default=42,
    help='Random seed to sample fingerprints.',
    show_default=True,
)
def main(**args: dict[str, Any]) -> None:
    args = AttributeDict(args)
    net = StegaStampModule.load_from_checkpoint(
        args.checkpoint_path,
        map_location=args.device,
    )

    all_encoded_imgs = []
    all_fingerprints = []

    print('Fingerprinting the images...')
    torch.manual_seed(args.seed)

    # generate identical fingerprints
    args.bit_length = net.hparams['bit_length']
    fingerprints = generate_random_fingerprints(args.bit_length, 1)
    fingerprints = (
        fingerprints.view(1, args.bit_length)
        .expand(args.batch_size, args.bit_length)
        .to(args.device)
    )

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

    bitwise_accuracy = 0
    test_img = torch.tensor(0)
    test_encoded_img = torch.tensor(0)

    with torch.inference_mode():
        for images, _ in tqdm.tqdm(
            dataloader,
            unit='images',
            desc='Fingerprinting images',
        ):
            # generate arbitrary fingerprints
            if not args.identical_fingerprints:
                fingerprints = (
                    generate_random_fingerprints(args.bit_length, args.batch_size)
                    .view(args.batch_size, args.bit_length)
                    .to(args.device)
                )

            images = torch.as_tensor(images, device=args.device)
            encoded_imgs = net.encoder(fingerprints[: images.size(0)], images)
            all_encoded_imgs.append(encoded_imgs.cpu())
            all_fingerprints.append(fingerprints[: images.size(0)].cpu())

            test_img = images[:49]
            test_encoded_img = encoded_imgs[:49]

            if args.check:
                decoded_imgs = net.decoder(encoded_imgs)
                decoded_imgs = torch.as_tensor(decoded_imgs > 0, dtype=torch.long)
                size = images.size(0)
                correct = decoded_imgs[:size] == fingerprints[:size]
                mean = torch.mean(correct.float(), dim=1)
                bitwise_accuracy += torch.sum(mean).item()

    all_encoded_imgs = torch.concat(all_encoded_imgs, dim=0).cpu()
    all_fingerprints = torch.concat(all_fingerprints, dim=0).cpu()

    path = pathlib.Path(args.output_dir)
    path.mkdir(parents=True, exist_ok=True)

    with (
        tarfile.open(path / 'watermarked_images.tar', 'w:gz') as tar,
        open(path / 'embedded_fingerprints.csv', 'w', newline='') as fp,
    ):
        fieldnames = ['filename', 'fingerprint']
        writer = csv.DictWriter(fp, fieldnames)
        writer.writeheader()
        for idx in range(len(all_encoded_imgs)):
            image = all_encoded_imgs[idx]
            fingerprint = all_fingerprints[idx]
            fingerprint = ''.join([str(bit.item()) for bit in fingerprint])
            filename = str(idx + 1) + '.png'

            buffer = io.BytesIO()
            image = F.to_pil_image(make_grid(image))
            image.save(buffer, format='PNG')
            buffer.seek(0)

            info = tarfile.TarInfo(filename)
            info.size = buffer.getbuffer().nbytes
            tar.addfile(info, buffer)
            writer.writerow({'filename': filename, 'fingerprint': fingerprint})

    if args.check:
        bitwise_accuracy = bitwise_accuracy / len(all_fingerprints)
        print(f'Bitwise accuracy on fingerprinted images: {bitwise_accuracy}')

        save_image(
            test_img,
            path / 'test_samples_clean.png',
            nrow=7,
        )
        save_image(
            test_encoded_img,
            path / 'test_samples_fingerprinted.png',
            nrow=7,
        )
        save_image(
            torch.abs(test_img - test_encoded_img),
            path / 'test_samples_residual.png',
            normalize=True,
            nrow=7,
        )


if __name__ == '__main__':
    click.rich_click.THEME = 'nord-box'
    main()
