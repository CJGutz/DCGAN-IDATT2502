import torch.nn as nn
import torch
from torch import Tensor

ZERO_SUBSTITUTE = 0.00001


class lsDiscriminator(nn.Module):
    def __init__(self, numb_color_channels, ndf, **kwargs):
        super(lsDiscriminator, self).__init__(**kwargs)

        self.main = nn.Sequential(
            self.stand_layer(numb_color_channels, ndf, 4, 2, 1),
            self.stand_layer(ndf, ndf * 2, 4, 2, 1),
            self.stand_layer(ndf * 2, ndf * 4, 4, 2, 1),
            self.stand_layer(ndf * 4, ndf * 8, 4, 2, 1),
            self.stand_layer(ndf * 8, ndf * 16, 4, 2, 1),
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=True)
        )

    def forward(self, x):
        return self.main(x)

    def accuracy(self, x, y):
        return torch.mean(torch.less(torch.abs(x - y), 0.5).float())

    def stand_layer(self, in_dim, out_dim, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(0.2)
        )

    def times_correct(self,
                      predictions: torch.Tensor,
                      actual: torch.Tensor
                      ) -> int:
        return int(torch.less(
            torch.abs(predictions - actual), 0.5
        ).sum().item())

    def calc_recall(self, true_positive, false_negative):
        return true_positive / max(true_positive + false_negative, ZERO_SUBSTITUTE)

    def calc_precision(self, true_positive, false_positive):
        return true_positive / max(true_positive + false_positive, ZERO_SUBSTITUTE)

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
