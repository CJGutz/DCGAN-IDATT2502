import torch
import torch.nn as nn


class lsGenerator(nn.Module):
    def __init__(self, nz, ngf, n_channels=3, num_layers=4):
        super(lsGenerator, self).__init__()

        self.main = nn.Sequential(
            # 1 x latent_size
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.LeakyReLU(0.01, True),
            nn.BatchNorm2d(ngf * 16),
            # 16ngf * 4 * 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.01, True),
            nn.BatchNorm2d(ngf * 8),
            # 8ngf * 8 * 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.01, True),
            nn.BatchNorm2d(ngf * 4),
            # 4ngf * 16 * 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.01, True),
            nn.BatchNorm2d(ngf * 2),
            # 2ngf * 32 * 32
            nn.ConvTranspose2d(ngf * 2, ngf * 1, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.01, True),
            nn.BatchNorm2d(ngf * 1),
            # ngf * 64 * 64
            nn.ConvTranspose2d(ngf * 1, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # 3 * 128 * 128
        )

    def forward(self, x):
        return self.main(x)

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
