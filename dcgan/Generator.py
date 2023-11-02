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

        self.main = nn.Sequential(
            nn.ConvTranspose2d(noise,
                               ngf * 8,
                               kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),
        )

        current_channels = ngf * 8

        for i in range(number_of_layers):
            self.main.extend(
                [
                    nn.ConvTranspose2d(current_channels,
                                       current_channels // 2,
                                       kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(current_channels // 2),
                    nn.ReLU(inplace=True)
                ]
            )
            current_channels //= 2

        self.main.add_module('conv_out',
                             nn.ConvTranspose2d(current_channels,
                                                numb_color_channels,
                                                kernel_size=4, stride=2, padding=1, bias=False))
        self.main.add_module('tanh', nn.Tanh())

    def forward(self, x):
        return self.main(x)
