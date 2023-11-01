import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self,
                 noise,
                 ngf,
                 numb_color_channels,
                 number_of_layers=4,
                 **kwargs):
        super().__init__(**kwargs)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(noise,
                               ngf,
                               kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
        )

        for i in range(number_of_layers):
            self.main.add_module(
                f'conv_{i}',
                nn.ConvTranspose2d(ngf * 2 ** i,
                                   ngf * 2 ** (i + 1),
                                   kernel_size=4, stride=2, padding=1, bias=False)
            )
            self.main.add_module(
                f'batch_norm_{i}',
                nn.BatchNorm2d(ngf * 2 ** (i + 1)))

            self.main.add_module(f'relu_{i}',
                                 nn.ReLU(inplace=True))

        self.main.add_module('conv_out',
                             nn.ConvTranspose2d(ngf * 2 ** number_of_layers,
                                                numb_color_channels,
                                                kernel_size=4, stride=2, padding=1, bias=False))

    def forward(self, x):
        return torch.tanh(self.main(x))
