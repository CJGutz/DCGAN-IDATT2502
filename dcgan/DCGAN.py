import torch
import random
import torch.nn as nn
from torch import optim
import torchvision.utils as vutils
from enum import Enum

from Visualization import print_epoch_images


class Label(Enum):
    real_label = 1.
    fake_label = 0.


# based in the paper by Alec Radford the, the team concluded that the weight should be distributed in this manner

def weights(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


class DCGAN:
    def __init__(self, num_epochs, dataloader, nc, device, generator, discriminator,
                 batch_size=128, lr=0.0002, beta1=0.5, nz=100, visualize=True):
        super(DCGAN, self).__init__()

        self.device = device
        self.generator = generator.apply(weights).to(device=self.device)
        self.discriminator = discriminator.apply(
            weights).to(device=self.device)
        self.optim_gen = optim.Adam(
            generator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optim_disc = optim.Adam(
            discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.dataloader = dataloader
        self.nz = nz
        self.numb_channels = nc
        self.visualize = visualize

    def pre_training(self):
        # Set random seed for reproducibility
        manualSeed = random.randint(1, 10000)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        torch.use_deterministic_algorithms(True)

        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        fixed_noise = torch.randn(64, self.nz, 1, 1, device=self.device)

        # Hyperparams
        numb_episodes = len(self.dataloader)
        criterion = nn.BCELoss().to(self.device)

        return manualSeed, fixed_noise, numb_episodes, criterion

    def train(self):
        (manual_seed, fixed_noise, numb_episodes,
         criterion) = self.pre_training()

        img_list = []
        G_losses = []
        D_losses = []

        for epoch in range(self.num_epochs):
            for i, data_batch in enumerate(self.dataloader, 0):
                # Training netG in real_samples
                self.discriminator.zero_grad()
                # real samples created form batches
                real_samples = data_batch[0].to(self.device)
                # fills tabel with real label(1)
                labels_real = torch.full((real_samples.size(0),), Label.real_label.value,
                                         dtype=torch.float, device=self.device)

                # calculate the loss and predicted value from real samples
                netD_predictions_real = self.discriminator.forward(
                    real_samples).view(-1)
                netD_loss_real = criterion(netD_predictions_real, labels_real)
                netD_loss_real.backward()

                # Testing discriminator on fake samples
                noise = torch.randn(real_samples.size(
                    0), self.nz, 1, 1, device=self.device)
                # fake bact of samples created with generator
                fake_samples = self.generator(noise)
                labels_fake = torch.full((real_samples.size(0),), Label.fake_label.value,
                                         dtype=torch.float, device=self.device)

                # calculate the predicted value and loss from fake samples
                netD_predictions_fake = self.discriminator.forward(
                    fake_samples.detach()).view(-1)
                netD_loss_fake = criterion(netD_predictions_fake, labels_fake)
                netD_loss_fake.backward()

                # Calculate the total loss of discriminator
                netD_loss = netD_loss_real + netD_loss_fake
                self.optim_disc.step()

                # Train netG
                self.generator.zero_grad()
                labels_real.fill_(Label.real_label.value)
                # loss of generator
                netG_output = self.discriminator(fake_samples).view(-1)
                netG_loss = criterion(netG_output, labels_real)
                netG_loss.backward()

                # optimizes generator using BCELoss
                self.optim_gen.step()

                # Print and save losses and generated images
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                          % (epoch, self.num_epochs, i, len(self.dataloader),
                             netD_loss.item(), netG_loss.item()))

                if ((i + 1) % 500 == 0) or ((epoch == self.num_epochs - 1) and (i == len(self.dataloader) - 1)):
                    with torch.no_grad():
                        fake = self.generator(fixed_noise).detach().cpu()
                    img_list.append(vutils.make_grid(
                        fake, padding=2, normalize=True))
                    if self.visualize:
                        print_epoch_images(self.dataloader, img_list)

                # save loss of both D(x) and G(x) for further visualization
                D_losses.append(netD_loss.item())
                G_losses.append(netG_loss.item())

    def save_model(self):
        return 0

    def load_model(self):
        return 0