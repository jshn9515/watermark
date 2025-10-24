import argparse
import csv
import pathlib

import torch
import torch.utils.data as utils
import tqdm
from datasets import load_dataset
from models import StegaStampModule
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
    num_workers=0,
)

for images, _ in tqdm.tqdm(dataloader):
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
print(f'Bitwise accuracy on fingerprinted images: {bitwise_accuracy}')  # non-corrected

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
