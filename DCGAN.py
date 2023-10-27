# Selve DCGAN klassen
import torch
import random


class DCGAN:
    def __init__(self, num_epochs, dataloader, netD, netG, device, nz, nc):
        super(DCGAN, self).__init__()

        self.num_epochs = num_epochs
        self.dataloader = dataloader
        self.netD = netD
        self.netG = netG
        self.device = device
        self.nz = nz
        self.nc = nc

    def train(self):
        # Set random seed for reproducibility
        manualSeed = random.randint(1, 10000)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        torch.use_deterministic_algorithms(True)

    def save_model(self):
        return 0

    def load_model(self):
        return 0
