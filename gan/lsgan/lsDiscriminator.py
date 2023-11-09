import torch.nn as nn
import torch


class lsDiscriminator(nn.Module):
    def __init__(self, numb_color_channels, ndf, n_downsamplings=3, **kwargs):
        super(lsDiscriminator, self).__init__(**kwargs)

        self.main = nn.Sequential(
            # 3 x 128 x 128
            nn.Conv2d(numb_color_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.01, True),
            # ndf * 64 * 64
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.01, True),
            # 2ndf * 32 * 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.01, True),
            # 4ndf * 16 * 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.01, True),
            # 8ndf * 8 * 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.01, True),
            # 16ndf * 4 * 4
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=True)

        )

    def forward(self, x):
        return self.main(x)

    def accuracy(self, x, y):
        return torch.mean(torch.less(torch.abs(x - y), 0.5).float())

    def stand_layer(self, in_dim, out_dim, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding,
                      bias=False),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(0.2)
        )
