import argparse
import csv
import io
import pathlib
import tarfile

import torch
import torch.utils.data as utils
import tqdm
from datasets import load_dataset
from models import StegaStampModule
from torchvision.transforms.v2.functional import to_pil_image
from torchvision.utils import make_grid, save_image
from utils import generate_random_fingerprints, parse_resolution

parser = argparse.ArgumentParser()
parser.add_argument(
    '--checkpoint_path',
    type=str,
    required=True,
    help='Path to trained StegaStamp encoder/decoder.',
)
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
    help='Path to save watermarked images to.',
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
    '--identical_fingerprints',
    action='store_true',
    help='If this option is provided use identical fingerprints. Otherwise sample arbitrary fingerprints. Default: False.',
)
parser.add_argument(
    '--check',
    action='store_true',
    help='Validate fingerprint detection accuracy. Default: False.',
)
parser.add_argument(
    '--batch_size',
    type=int,
    default=64,
    help='Batch size. Default: 64.',
)
parser.add_argument(
    '--device',
    type=str,
    default='cpu',
    choices=['cpu', 'cuda', 'xpu'],
    help='Device to use. Default: cpu.',
)
parser.add_argument(
    '--seed',
    type=int,
    default=42,
    help='Random seed to sample fingerprints. Default: 42.',
)
args = parser.parse_args()

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
    num_workers=0,
)

bitwise_accuracy = 0
test_img = torch.tensor(0)
test_encoded_img = torch.tensor(0)

with torch.inference_mode():
    for images, _ in tqdm.tqdm(dataloader):
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

        test_img = images
        test_encoded_img = encoded_imgs

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
        image = to_pil_image(make_grid(image))
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
        test_img[:49],
        path / 'test_samples_clean.png',
        nrow=7,
    )
    save_image(
        test_encoded_img[:49],
        path / 'test_samples_fingerprinted.png',
        nrow=7,
    )
    save_image(
        torch.abs(test_img - test_encoded_img)[:49],
        path / 'test_samples_residual.png',
        normalize=True,
        nrow=7,
    )
