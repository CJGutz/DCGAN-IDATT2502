# CNN laget for Generatoren basert på fagartikelen til Alec Radford
import torch.nn as nn
import torch


class CommonConv(nn.ConvTranspose2d):
    def __init__(self, *args, **kwargs):
        super().__init__(kernel_size=4, stride=2, padding=1, bias=False,
                         *args, **kwargs)


class Generator(nn.Module):
    def __init__(self,
                 generator_input_z: int,
                 color_channels: int,
                 last_out_channel: int,
                 number_of_layers: int = 4,
                 **kwargs):
        super().__init__(**kwargs)

        channel_sizes = [generator_input_z]
        channel_sizes.extend([last_out_channel * 2 ** i
                              for i in range(number_of_layers, 0, -1)])

        self.main = nn.Sequential()
        for i in range(len(channel_sizes) - 1):
            self.main.extend(
                [
                    CommonConv(channel_sizes[i], channel_sizes[i+1]),
                    nn.BatchNorm2d(channel_sizes[i+1]),
                    nn.ReLU(True),
                ]
            )

        first_conv = self.main[0]
        first_conv.stride = torch.tensor([1, 1])
        first_conv.padding = torch.tensor([0, 0])

        self.main.append(CommonConv(last_out_channel, color_channels))

    def forward(self, x):
        return torch.tanh(self.main(x))
