import math
from itertools import chain
from typing import Any

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from lightning.pytorch.utilities import AttributeDict
from torch import Tensor
from torchvision.utils import make_grid
from utils import generate_random_fingerprints

__all__ = ['StegaStampModule']

plot_points = (
    list(range(0, 1000, 100))
    + list(range(1000, 3000, 200))
    + list(range(3000, 100000, 1000))
)


class StegaStampEncoder(nn.Module):
    def __init__(
        self,
        resolution: int = 32,
        image_channels: int = 1,
        fingerprint_size: int = 100,
        return_residual: bool = False,
    ):
        super().__init__()
        self.fingerprint_size = fingerprint_size
        self.image_channels = image_channels
        self.return_residual = return_residual
        self.secret_dense = nn.Linear(self.fingerprint_size, 16 * 16 * image_channels)

        log_resolution = int(math.log(resolution, 2))
        if resolution != 2**log_resolution:
            raise AssertionError(
                f'Image resolution must be a power of 2, got {resolution}.'
            )

        self.fingerprint_upsample = nn.Upsample(
            scale_factor=(2 ** (log_resolution - 4), 2 ** (log_resolution - 4))
        )

        self.conv1 = nn.Conv2d(2 * image_channels, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 2, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv4 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv5 = nn.Conv2d(128, 256, 3, 2, 1)

        self.pad6 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up6 = nn.Conv2d(256, 128, 2, 1)
        self.upsample6 = nn.Upsample(scale_factor=(2, 2))
        self.conv6 = nn.Conv2d(128 + 128, 128, 3, 1, 1)
        self.pad7 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up7 = nn.Conv2d(128, 64, 2, 1)
        self.upsample7 = nn.Upsample(scale_factor=(2, 2))
        self.conv7 = nn.Conv2d(64 + 64, 64, 3, 1, 1)
        self.pad8 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up8 = nn.Conv2d(64, 32, 2, 1)
        self.upsample8 = nn.Upsample(scale_factor=(2, 2))
        self.conv8 = nn.Conv2d(32 + 32, 32, 3, 1, 1)
        self.pad9 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up9 = nn.Conv2d(32, 32, 2, 1)
        self.upsample9 = nn.Upsample(scale_factor=(2, 2))

        self.conv9 = nn.Conv2d(32 + 32 + 2 * image_channels, 32, 3, 1, 1)
        self.conv10 = nn.Conv2d(32, 32, 3, 1, 1)
        self.residual = nn.Conv2d(32, image_channels, 1)

    def forward(self, fingerprint: Tensor, image: Tensor) -> Tensor:
        fingerprint = F.relu(self.secret_dense(fingerprint))
        fingerprint = fingerprint.view(-1, self.image_channels, 16, 16)
        fingerprint_enlarged = self.fingerprint_upsample(fingerprint)

        inputs = torch.concat([fingerprint_enlarged, image], dim=1)
        conv1 = F.relu(self.conv1(inputs))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        conv4 = F.relu(self.conv4(conv3))
        conv5 = F.relu(self.conv5(conv4))

        up6 = F.relu(self.up6(self.pad6(self.upsample6(conv5))))
        merge6 = torch.concat([conv4, up6], dim=1)
        conv6 = F.relu(self.conv6(merge6))
        up7 = F.relu(self.up7(self.pad7(self.upsample7(conv6))))
        merge7 = torch.concat([conv3, up7], dim=1)
        conv7 = F.relu(self.conv7(merge7))
        up8 = torch.relu(self.up8(self.pad8(self.upsample8(conv7))))
        merge8 = torch.concat([conv2, up8], dim=1)
        conv8 = F.relu(self.conv8(merge8))
        up9 = F.relu(self.up9(self.pad9(self.upsample9(conv8))))
        merge9 = torch.concat([conv1, up9, inputs], dim=1)
        conv9 = F.relu(self.conv9(merge9))
        conv10 = F.relu(self.conv10(conv9))

        residual = self.residual(conv10)
        if not self.return_residual:
            residual = F.sigmoid(residual)
        return residual


class StegaStampDecoder(nn.Module):
    def __init__(
        self,
        resolution: int = 32,
        image_channels: int = 1,
        fingerprint_size: int = 1,
    ):
        super().__init__()
        self.resolution = resolution
        self.image_channels = image_channels
        self.decoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, (3, 3), 2, 1),  # 16
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # 8
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1),  # 4
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),  # 2
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), 2, 1),
            nn.ReLU(),
        )
        self.dense = nn.Sequential(
            nn.Linear(resolution * resolution * 128 // 32 // 32, 512),
            nn.ReLU(),
            nn.Linear(512, fingerprint_size),
        )

    def forward(self, image: Tensor) -> Tensor:
        x = self.decoder(image)
        x = x.view(-1, self.resolution * self.resolution * 128 // 32 // 32)
        return self.dense(x)


class StegaStampModule(pl.LightningModule):
    def __init__(self, args: dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(args)
        if isinstance(args, dict):
            args = AttributeDict(args)
        self.encoder = StegaStampEncoder(
            resolution=args.resolution,
            image_channels=3,
            fingerprint_size=args.bit_length,
        )
        self.decoder = StegaStampDecoder(
            resolution=args.resolution,
            image_channels=3,
            fingerprint_size=args.bit_length,
        )
        self.args = args
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.steps_since_l2_loss_activated = -1

    def training_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        clean_img = batch[0].to(self.device)
        batch_size = min(self.args.batch_size, clean_img.size(0))
        fingerprints = generate_random_fingerprints(
            self.args.bit_length,
            batch_size,
        ).to(self.device)

        l2_loss_weight = min(
            max(
                0,
                self.args.l2_loss_weight
                * (self.steps_since_l2_loss_activated - self.args.l2_loss_await)
                / self.args.l2_loss_ramp,
            ),
            self.args.l2_loss_weight,
        )

        encoded_img = self.encoder(fingerprints, clean_img)
        residual = encoded_img - clean_img
        decoded_img = self.decoder(encoded_img)

        l2_loss = self.mse_loss(clean_img, encoded_img)
        bce_loss = self.bce_loss(decoded_img.view(-1), fingerprints.view(-1))
        loss = l2_loss_weight * l2_loss + self.args.bce_loss_weight * bce_loss

        predicted = torch.as_tensor(decoded_img > 0, dtype=torch.float)
        bitwise_accuracy = 1.0 - torch.mean(torch.abs(fingerprints - predicted))
        if self.steps_since_l2_loss_activated == -1:
            if bitwise_accuracy > 0.9:
                self.steps_since_l2_loss_activated = 0
        else:
            self.steps_since_l2_loss_activated += 1

        self.log('loss', loss, prog_bar=True)
        self.log_dict(
            {
                'BCE_loss': bce_loss,
                'bitwise_accuracy': bitwise_accuracy,
                'clean_statistics/min': clean_img.min(),
                'clean_statistics/max': clean_img.max(),
                'fingerprinted_statistics/min': encoded_img.min(),
                'fingerprinted_statistics/max': encoded_img.max(),
                'residual_statistics/min': residual.min(),
                'residual_statistics/max': residual.max(),
                'residual_statistics/mean_abs': residual.abs().mean(),
                'loss_weights/l2_loss_weight': l2_loss_weight,
                'loss_weights/BCE_loss_weight': self.args.bce_loss_weight,
            },
        )
        if self.global_step in plot_points:
            wandb.log(
                {
                    'clean_images': [wandb.Image(make_grid(clean_img, normalize=True))],
                    'residuals': [
                        wandb.Image(
                            make_grid(residual, normalize=True, scale_each=True)
                        )
                    ],
                    'encoded_images': [
                        wandb.Image(make_grid(encoded_img, normalize=True))
                    ],
                },
                step=self.global_step,
            )
        return loss

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(
            chain(self.encoder.parameters(), self.decoder.parameters()),
            lr=self.args.lr,
        )
