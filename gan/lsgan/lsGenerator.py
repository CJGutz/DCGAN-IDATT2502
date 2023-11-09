import torch
import torch.nn as nn


def conv_layer(in_dim, out_dim, kernel_size, stride, padding, bias):
    return nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding, bias=bias)


def stand_layer(in_dim, out_dim, kernel_size, stride, padding):
    return nn.Sequential(
        conv_layer(in_dim, out_dim, kernel_size, stride, padding, False),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(True)
    )


class lsGenerator(nn.Module):
    def __init__(self, nz, ngf, n_channels=3, num_layers=4):
        super(lsGenerator, self).__init__()

        self.main = nn.Sequential(
            conv_layer(nz, ngf * 16, 4, 1, 0, False),
            stand_layer(ngf * 16, ngf * 8, 4, 2, 1),
            stand_layer(ngf * 8, ngf * 4, 4, 2, 1),
            stand_layer(ngf * 4, ngf * 2, 4, 2, 1),
            stand_layer(ngf * 2, ngf * 1, 4, 2, 1),
            conv_layer(ngf * 1, 3, 4, 2, 1, False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

    def accuracy(self, x, y):
        return torch.mean(torch.less(torch.abs(x - y), 0.5).float())
