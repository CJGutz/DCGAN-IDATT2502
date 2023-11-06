import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self,
                 noise,
                 ngf,
                 numb_color_channels,
                 number_of_layers=3,
                 **kwargs):
        super().__init__(**kwargs)

        layers = [
            nn.ConvTranspose2d(noise,
                               ngf * 8,
                               kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),
        ]

        current_channels = ngf * 8

        for _ in range(number_of_layers):
            layers.extend(
                [
                    nn.ConvTranspose2d(current_channels,
                                       current_channels // 2,
                                       kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(current_channels // 2),
                    nn.ReLU(inplace=True)
                ]
            )
            current_channels //= 2

        layers.append(
            nn.ConvTranspose2d(current_channels,
                               numb_color_channels,
                               kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

