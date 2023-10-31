import torch.nn as nn
import torch


class CommonConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        default_params = dict(kernel_size=4, stride=2, padding=1, bias=False)
        default_params.update(kwargs)
        super().__init__(*args, **default_params)


class Discriminator(nn.Module):
    def __init__(self,
                 numb_color_channels,
                 ndf,
                 number_of_layers=4,
                 **kwargs):
        super().__init__(**kwargs)

        self.main = nn.Sequential(
            CommonConv(numb_color_channels, ndf),
            nn.LeakyReLU(0.2, inplace=True)
        )
        for i in range(number_of_layers - 1):
            self.main.extend(
                [
                    CommonConv(ndf * 2 ** i,
                               ndf * 2 ** (i + 1)),
                    nn.BatchNorm2d(ndf * 2 ** (i + 1)),
                    nn.LeakyReLU(0.2, inplace=True)
                ]
            )

        self.main.append(nn.AdaptiveAvgPool2d(1))
        # self.main.append(nn.Flatten()) see if flatten helps, according to papers it should
        self.main.append(CommonConv(ndf *
                                    2 ** (number_of_layers - 1), 1,
                                    kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        return torch.sigmoid(self.main(x))
