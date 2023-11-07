import torch
import math
import os
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

from Visualization import save_img_generated


class Label:
    REAL = 1
    FAKE = 0


class GAN:
    LSGAN = "lsgan"
    DCGAN = "dcgan"


# based in the paper by Alec Radford the, the team concluded that the weight should be distributed in this manner
def weights(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


class DCGAN:
    def __init__(self, generator, discriminator, num_epochs, dataloader, model_name, nc, device, gan,
                 batch_size=128, lr=0.0002, beta1=0.5, nz=100, load=False):
        super(DCGAN, self).__init__()

        self.gan = gan
        self.device = device
        self.model_name = model_name

        self.generator = generator.apply(
            weights)
        self.discriminator = discriminator.apply(
            weights)
        self.optim_gen = optim.Adam(
            generator.parameters(),
            lr=lr, betas=(beta1, 0.999))

        self.optim_disc = optim.Adam(
            discriminator.parameters(),
            lr=lr, betas=(beta1, 0.999))

        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.dataloader = dataloader
        self.nz = nz
        self.numb_channels = nc

        self.G_losses = []
        self.D_losses = []
        self.G_accuracies = []
        self.D_accuracies = []
        self.img_list = []

        if load:
            print("loading model")
            self.load_model()

    def pre_training(self):
        # Create batch of latent vectors that we will use to visualize
        # the progression of the generator
        fixed_noise = torch.randn(64, self.nz, 1, 1, device=self.device)

        # Hyperparams
        numb_episodes = len(self.dataloader)
        if self.gan == GAN.LSGAN:
            criterion = nn.MSELoss().to(device=self.device)
        else:
            criterion = nn.BCELoss().to(device=self.device)

        return fixed_noise, numb_episodes, criterion

    def train(self):
        (fixed_noise, numb_episodes,
         criterion) = self.pre_training()

        for epoch in range(1, self.num_epochs+1):
            for i, data_batch in enumerate(self.dataloader, 0):
                # training netG in real_samples
                self.discriminator.zero_grad()
                # real samples created form batches
                real_samples = data_batch[0].to(self.device)
                # fills tabel with real label(1)
                labels = torch.full((real_samples.size(0),), Label.REAL,
                                    dtype=torch.float, device=self.device)

                # calculate the loss and predicted value from real samples
                netD_predictions_real = self.discriminator(
                    real_samples).view(-1)
                netD_loss_real = criterion(netD_predictions_real, labels)
                netD_loss_real.backward()

                # Testing discriminator on fake samples
                noise = torch.randn(real_samples.size(0), self.nz, 1, 1,
                                    device=self.device)
                # fake bact of samples created with generator
                fake_samples = self.generator(noise)
                labels.fill_(Label.FAKE)

                # calculate the predicted value and loss from fake samples
                netD_predictions_fake = self.discriminator(
                    fake_samples.detach()).view(-1)
                netD_loss_fake = criterion(netD_predictions_fake, labels)
                netD_loss_fake.backward()

                # calculate the total loss of discriminator
                netD_loss = 0.5*(netD_loss_real + netD_loss_fake)
                self.optim_disc.step()

                # train netG
                self.generator.zero_grad()
                labels.fill_(Label.REAL)

                # loss of generator
                netG_output = self.discriminator(fake_samples).view(-1)
                netG_loss = criterion(netG_output, labels)
                netG_loss.backward()

                # optimizes generator using BCELoss
                self.optim_gen.step()

                # Print and save losses and generated images
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                          % (epoch, self.num_epochs, i, len(self.dataloader),
                             netD_loss.item(), netG_loss.item()))

                # if statement used for printing images
                nth_iteration = len(self.dataloader) // 3
                if ((i + 1) % nth_iteration == 0) or (
                        (epoch == self.num_epochs) and (i == len(self.dataloader) - 1)):
                    self.save_iteration_images(
                        fixed_noise, epoch, i, nth_iteration)

                # save loss and accuracy of both D(x) and G(x) for further visualization
                self.D_losses.append(netD_loss.item())
                self.G_losses.append(netG_loss.item())
                self.D_accuracies.append((
                                                 (self.discriminator.accuracy(
                                                     netD_predictions_real,
                                                     labels.fill_(Label.REAL))
                                                  + self.discriminator.accuracy(
                                                             netD_predictions_fake,
                                                             labels.fill_(Label.FAKE))
                                                  ) / 2).item())
                self.G_accuracies.append(
                    self.generator.accuracy(
                        netG_output, labels.fill_(Label.REAL)).item())

    def save_iteration_images(self, fixed_noise, epoch, iteration, nth_iteration):
        with torch.no_grad():
            self.generator.eval()
            fake = self.generator(fixed_noise).detach().cpu()
            self.img_list.append(vutils.make_grid(
                fake, padding=2, normalize=True))
            save_img_generated(self.img_list,
                               f"fig-epoch{epoch}-{self.num_epochs}"
                               f"-itr{(iteration + 1) // nth_iteration}"
                               f"-{len(self.dataloader) // nth_iteration}.png")

    def save_model(self):
        PATH = os.path.join("datasets", "model", self.model_name + ".pt")

        torch.save({
            "discriminator": self.discriminator.state_dict(),
            "generator": self.generator.state_dict(),
            "netD_optimize": self.optim_disc.state_dict(),
            "netG_optimize": self.optim_gen.state_dict(),
        }, PATH)

    def load_model(self):
        PATH = os.path.join("datasets", "model", self.model_name + ".pt")
        if not os.path.isfile(PATH):
            print("No model found")
            return
        checkpoint = torch.load(PATH)

        self.discriminator.load_state_dict(checkpoint["discriminator"])
        self.generator.load_state_dict(checkpoint["generator"])
        self.optim_disc.load_state_dict(checkpoint["netD_optimize"])
        self.optim_gen.load_state_dict(checkpoint["netG_optimize"])
