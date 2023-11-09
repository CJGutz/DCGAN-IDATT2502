import torch.nn as nn
from torch import Tensor
import torch

ZERO_SUBSTITUTE = 0.00001


class Discriminator(nn.Module):
    def __init__(self,
                 numb_color_channels,
                 ndf,
                 number_of_layers=3,
                 **kwargs):
        super().__init__(**kwargs)

        layers = [
            nn.Conv2d(numb_color_channels, ndf,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        for i in range(number_of_layers):
            layers.extend(
                [
                    nn.Conv2d(ndf * 2 ** i,
                              ndf * 2 ** (i + 1),
                              kernel_size=4, stride=2, padding=1, bias=False),

                    nn.BatchNorm2d(ndf * 2 ** (i + 1),
                                   affine=True, track_running_stats=True),

                    nn.LeakyReLU(0.2, inplace=True)
                ]
            )

        layers.append(nn.Conv2d(ndf * 2 ** number_of_layers, 1,
                                kernel_size=4, stride=1, padding=0))
        layers.append(nn.Sigmoid())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

    def calc_f1_score(self,
                      preds_real: Tensor,
                      preds_fake: Tensor,
                      real_labels: Tensor,
                      fake_labels: Tensor
                      ) -> float:
        true_positive = self.times_correct(preds_real, real_labels)
        false_positive = self.times_correct(preds_real, fake_labels)
        false_negative = self.times_correct(preds_fake, real_labels)
        precision = self.calc_precision(true_positive, false_positive)
        recall = self.calc_recall(true_positive, false_negative)
        return 2 * ((precision * recall) / max(precision + recall, ZERO_SUBSTITUTE))

    def times_correct(self,
                      predictions: torch.Tensor,
                      actual: torch.Tensor
                      ) -> int:
        return int(torch.less(
            torch.abs(predictions - actual), 0.5
        ).sum().item())

    def calc_precision(self, true_positive, false_positive):
        return true_positive / max(true_positive + false_positive, ZERO_SUBSTITUTE)

    def calc_recall(self, true_positive, false_negative):
        return true_positive / max(true_positive + false_negative, ZERO_SUBSTITUTE)
