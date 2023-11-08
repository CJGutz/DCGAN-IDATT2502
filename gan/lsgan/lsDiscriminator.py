import torch.nn as nn
import torch

class lsDiscriminator(nn.Module):
    def __init__(self, numb_color_channels, ndf, n_downsamplings=3, **kwargs):
        super(lsDiscriminator, self).__init__(**kwargs)

        layers = []

        d = ndf
        layers.append(nn.Conv2d(numb_color_channels, d, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2))

        for i in range(n_downsamplings):
            d_last = d
            d = min(ndf * 2 ** (i + 1), ndf * 8)
            layers.append(self.stand_layer(d_last, d, kernel_size=4, stride=2, padding=1))

        # 2: logit
        layers.append(nn.Conv2d(d, 1, kernel_size=4, stride=1, padding=0))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        y = self.net(x)
        return y

    def accuracy(self, x, y):
        return torch.mean(torch.less(torch.abs(x - y), 0.5).float())

    def stand_layer(self, in_dim, out_dim, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding,
                      bias=False),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(0.2)
        )
