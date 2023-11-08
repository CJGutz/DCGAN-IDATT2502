import torch
import torch.nn as nn


class lsGenerator(nn.Module):
    def __init__(self, nz, ngf, n_channels):
        super(lsGenerator, self).__init__()

        self.main = nn.Sequential(
            self.stand_layer(nz, ngf * 4,
                             kernel_size=3, stride=1, padding=0),
            self.stand_layer(ngf * 4, ngf * 4,
                             kernel_size=4, stride=2, padding=1),
            self.stand_layer(ngf * 4, ngf * 4,
                             kernel_size=4, stride=1, padding=1),
            self.stand_layer(ngf * 4, ngf * 4,
                             kernel_size=4, stride=2, padding=1),
            self.stand_layer(ngf * 4, ngf * 2,
                             kernel_size=4, stride=1, padding=1),
            self.stand_layer(ngf * 2, ngf,
                             kernel_size=4, stride=2, padding=1),
            self.stand_layer(ngf, n_channels,
                             kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(n_channels, n_channels,
                               kernel_size=4, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

    def accuracy(self, x, y):
        return torch.mean(torch.less(torch.abs(x - y), 0.5).float())

    def stand_layer(self, in_dim, out_dim, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size,
                               stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(True)
        )
