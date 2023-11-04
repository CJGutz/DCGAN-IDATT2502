import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self,
                 numb_color_channels,
                 ndf,
                 number_of_layers=3,
                 **kwargs):
        super().__init__(**kwargs)

        layers = [
            nn.Conv2d(numb_color_channels, ndf,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        for i in range(number_of_layers):
            layers.extend(
                [
                    nn.Conv2d(ndf * 2 ** i,
                              ndf * 2 ** (i + 1),
                              kernel_size=4, stride=2, padding=1, bias=False),

                    nn.BatchNorm2d(ndf * 2 ** (i + 1),
                                   affine=True, track_running_stats=True),

                    nn.LeakyReLU(0.2, inplace=True)
                ]
            )

        layers.append(nn.Conv2d(ndf * 2 ** number_of_layers, 1,
                                kernel_size=4, stride=1, padding=0))
        layers.append(nn.Sigmoid())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
