import torch.nn as nn
import torch


class lsDiscriminator(nn.Module):
    def __init__(self, numb_color_channels, ndf, number_of_layers=3, **kwargs):
        super(lsDiscriminator, self).__init__(**kwargs)

        layers = []
        in_dim = numb_color_channels

        for i in range(number_of_layers):
            out_dim = ndf * 2 ** i
            layers.append(self.stand_layer(in_dim, out_dim,
                                           kernel_size=4, stride=2, padding=1))
            in_dim = out_dim

        # Output layer
        layers.append(nn.Conv2d(out_dim, 1,
                                kernel_size=4, stride=1, padding=0, bias=False))

        # AdaptiveAvgPool2d layer to dynamically adjust spatial dimensions
        layers.append(nn.AdaptiveAvgPool2d(1))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        out = self.main(x)
        return out

    def accuracy(self, x, y):
        return torch.mean(torch.less(torch.abs(x - y), 0.5).float())

    def stand_layer(self, in_dim, out_dim, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size,
                      stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(0.2,
                         inplace=True)
        )
