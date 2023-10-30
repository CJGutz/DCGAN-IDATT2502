# Selve DCGAN klassen
import torch
import random
import torch.nn as nn
from torch import optim


# based in the paper by Alec Radford the, the team concluded that the weight should be distributed in this maner
def weights(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


class DCGAN:
    def __init__(self, num_epochs, dataloader, nc, device, generator, discriminator,
                 batch_size=128, lr=0.0002, beta1=0.5, nz=100):
        super(DCGAN, self).__init__()

        self.device = device
        self.generator = generator.apply(weights).to(device=self.device)
        self.discriminator = discriminator.apply(weights).to(device=self.device)
        self.optim_gen = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optim_disc = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
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

        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        fixed_noise = torch.randn(64, self.nz, 1, 1, device=self.device)

        # Hyperparams
        numb_episodes = len(self.dataloader)
        criterion = nn.BCELoss().to(self.device)

        # Establish convention for real and fake labels during training
        real_label = 1.
        fake_label = 0.

        return manualSeed, fixed_noise, numb_episodes, criterion, real_label, fake_label

    def train(self):
        (manual_seed, fixed_noise, numb_episodes,
         criterion, real_label, fake_label) = self.pre_training()

        img_list = []
        G_losses = []
        D_losses = []

        for epoch in range(self.num_epochs):
            for i, data_batch in enumerate(self.dataloader, 0):
                # create a batch of real samples
                self.discriminator.zero_grad()
                samples = data_batch[0].to(self.device)
                samples_size = samples.size(0)
                labels = torch.full((samples_size,), real_label, dtype=torch.float, device=self.device)

                # forward pass through D
                netD_predictions = self.discriminator.forward(samples).view(-1)

                # calculate loss log(D(x))
                netD_err = criterion(netD_predictions, labels)
                netD_err.backward()

                # create batch of fake samples with G
                noise = torch.randn(samples_size, self.nz, 1, 1, device=self.device)
                fake_samples = self.generator(noise)
                labels.fill_(fake_label)

                # forward pass this patch to D
                netD_predictions_fake = self.discriminator.forward(fake_samples.detach()).view(-1)

                # calculate loss log(1 - D(G(z)))
                netD_fake_err = criterion(netD_predictions_fake, labels)
                netD_fake_err.backward()

                netD_err += netD_fake_err
                self.optim_disc.step()

                self.generator.zero_grad()
                labels.fill_(real_label)
                netG_output = self.discriminator(fake_samples).view(-1)
                netG_err = criterion(netG_output, labels)
                netG_err.backward()
                self.optim_gen.step()

                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                          % (epoch, self.num_epochs, i, len(self.dataloader),
                             netD_err.item(), netG_err.item()))

                D_losses.append(netD_err.item())
                G_losses.append(netG_err.item())

    def save_model(self):
        return 0

    def load_model(self):
        return 0
