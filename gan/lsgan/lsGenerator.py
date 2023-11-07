import torch
import torch.nn as nn


class lsGenerator(nn.Module):
    def __init__(self, nz, ngf, nChannels):
        super(lsGenerator, self).__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4,
                               kernel_size=4, stride=2, padding=1, bias=False
                               ),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nChannels,
                               kernel_size=4, stride=2, padding=1, bias=False
                               ),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

    def accuracy(self, x, y):
        return torch.mean(torch.less(torch.abs(x - y), 0.5).float())
