# Selve DCGAN klassen
import torch
import random
import torch.nn as nn


# based in the paper by Alec Radford the, the team concluded that the weight should be distributed in this maner
def weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class DCGAN:
    def __init__(self, num_epochs, dataloader, nz, nc):
        super(DCGAN, self).__init__()

        self.num_epochs = num_epochs
        self.dataloader = dataloader
        self.nz = nz
        self.nc = nc

    def pre_training(self):
        # Set random seed for reproducibility
        manualSeed = random.randint(1, 10000)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        torch.use_deterministic_algorithms(True)

    def train(self):
        return 0

    def save_model(self):
        return 0

    def load_model(self):
        return 0
