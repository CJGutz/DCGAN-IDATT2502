import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self,
                 noise_dim: int,
                 first_out_channel: int,
                 output_channels: int,
                 number_of_layers: int = 3,
                 **kwargs):
        super().__init__(**kwargs)

        self.main = nn.Sequential(
            # Project and reshape the noise vector
            nn.Linear(noise_dim, first_out_channel * 4 * 4 * 4),
            nn.Unflatten(1, (first_out_channel, 4, 4)),
            nn.ReLU(inplace=True)
        )

        for i in range(number_of_layers):
            self.main.add_module(
                f'conv_{i}',
                nn.ConvTranspose2d(first_out_channel * 2 ** i,
                                   first_out_channel * 2 ** (i + 1),
                                   kernel_size=4, stride=2, padding=1, bias=False)
            )
            self.main.add_module(
                nn.BatchNorm2d(first_out_channel * 2 ** (i + 1))
            )
            self.main.add_module(nn.ReLU(inplace=True))

        # Output layer
        self.main.add_module('conv_out',
                             nn.ConvTranspose2d(first_out_channel * 2 ** number_of_layers,
                                                output_channels,
                                                kernel_size=4, stride=2, padding=1, bias=False))

    def forward(self, x):
        return torch.tanh(self.main(x))
