import torch
import torch.nn as nn


class lsGenerator(nn.Module):
    def __init__(self, nz, ngf, n_channels=3, num_layers=4):
        super(lsGenerator, self).__init__()

        layers = []

        d = min(nz * 2 ** (num_layers - 1), ngf * 8)
        layers.append(self.stand_layer(nz, d, kernel_size=4, stride=1, padding=0))

        for i in range(num_layers - 1):
            d_last = d
            d = min(ngf * 2 ** (num_layers - 2 - i), ngf * 8)
            layers.append(self.stand_layer(d_last, d, kernel_size=4, stride=2, padding=1))

        layers.append(nn.ConvTranspose2d(d, n_channels, kernel_size=4, stride=2, padding=1))
        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

    def forward(self, z):
        x = self.net(z)
        return x

    def accuracy(self, x, y):
        return torch.mean(torch.less(torch.abs(x - y), 0.5).float())

    def stand_layer(self, in_dim, out_dim, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(True)
        )

    def conv_layer(self, in_dim, out_dim, kernel_size, stride, padding, bias):
        return nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding, bias=bias)
