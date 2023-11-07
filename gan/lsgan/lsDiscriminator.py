import torch.nn as nn
import torch


class lsDiscriminator(nn.Module):
    def __init__(self, numb_color_channels, ndf, number_of_layers=3, **kwargs):
        super(lsDiscriminator, self).__init__(**kwargs)

        self.main = nn.Sequential(
            # Layer 1
            nn.Conv2d(numb_color_channels, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2
            nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3
            nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4
            nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 5
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
        )

    def forward(self, x):
        return self.main(x)

    def accuracy(self, x, y):
        return torch.mean(torch.less(torch.abs(x - y), 0.5).float())
