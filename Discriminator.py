# CNN laget for dicriminatore basert på fagartikelen til Alec Radford
import torch.nn as nn
import torch


class CommonConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(kernel_size=4, stride=2, padding=1, bias=False,
                         *args, **kwargs)


class Discriminator(nn.Module):
    def __init__(self,
                 color_channels: int,
                 first_out_channel: int,
                 number_of_layers: int = 4,
                 **kwargs):
        super().__init__(**kwargs)

        self.main = nn.Sequential(
            CommonConv(color_channels, first_out_channel),
            nn.LeakyReLU(0.2, inplace=True)
        )
        for i in range(number_of_layers - 1):
            self.main.extend(
                [
                    CommonConv(first_out_channel * 2 ** i,
                               first_out_channel * 2 ** (i + 1)),
                    nn.BatchNorm2d(first_out_channel * 2 ** (i + 1)),
                    nn.LeakyReLU(0.2, inplace=True)
                ]
            )
        self.main.append(nn.AdaptiveAvgPool2d(1))
        self.main.append(CommonConv(first_out_channel *
                         2 ** (number_of_layers - 1), 1,
            kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        return torch.sigmoid(self.main(x))
