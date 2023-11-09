import torch
import torch.nn as nn


def conv_layer(in_dim, out_dim, kernel_size, stride, padding, bias):
    return


class lsGenerator(nn.Module):
    def __init__(self, nz, ngf, numb_channels=3):
        super(lsGenerator, self).__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 16, kernel_size=4, stride=1, padding=0, bias=False),
            self.stand_layer(ngf * 16, ngf * 8, 4, 2, 1),
            self.stand_layer(ngf * 8, ngf * 4, 4, 2, 1),
            self.stand_layer(ngf * 4, ngf * 2, 4, 2, 1),
            self.stand_layer(ngf * 2, ngf * 1, 4, 2, 1),
            nn.ConvTranspose2d(ngf, numb_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

    def stand_layer(self, in_dim, out_dim, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim,
                               kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(True)
        )

    def accuracy(self, x, y):
        return torch.mean(torch.less(torch.abs(x - y), 0.5).float())
