# Selve DCGAN klassen
import torch
import random
import torch.nn as nn
from torch import optim
from Generator import Generator as generator
from Discriminator import Discriminator as discriminator


# based in the paper by Alec Radford the, the team concluded that the weight should be distributed in this maner
def weights(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


class DCGAN:
    def __init__(self, num_epochs, dataloader, nc, device,
                 batch_size=32, lr=0.0002, beta1=0.5, nz=100):
        super(DCGAN, self).__init__()

        self.device = device
        # find a better way to apply gen and dis. Maby create them in execution and then just pass them here
        # self.generator = generator.apply(weights).to(device=self.device)
        # self.discriminator = discriminator.apply(weights).to(device=self.device)
        # self.optim_gen = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
        # self.optim_disc = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.dataloader = dataloader
        self.nz = nz
        self.numb_channels = nc

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
