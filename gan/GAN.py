import torch
from pytorch_gan_metrics import get_inception_score, get_fid
import os
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

from Visualization import save_img_generated, plot_iteration_values, SubFigure, IterationValues


class Label:
    REAL = 1.
    FAKE = 0.


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
                 batch_size=128, lr=0.0002, beta1=0.5, nz=100, load=False, model_save=True):
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
        self.f1_scores = []
        self.recall = []
        self.precision = []
        self.img_list = []
        self.inception_scores_mean = []
        self.inception_scores_std = []

        self.model_save = model_save

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
            print("Starting Training using MSE...")
            criterion = nn.MSELoss().to(device=self.device)
        else:
            print("Starting Training using BCE...")
            criterion = nn.BCELoss().to(device=self.device)

        return fixed_noise, numb_episodes, criterion

    def train(self):
        (fixed_noise, numb_episodes,
         criterion) = self.pre_training()
        nth_iteration = len(self.dataloader) // 3

        for epoch in range(1, self.num_epochs + 1):
            for i, data_batch in enumerate(self.dataloader, 0):
                # training netG in real_samples
                self.discriminator.zero_grad()
                # real samples created form batches
                real_samples = data_batch[0].to(self.device)

                # fills tabel with real label(1) and fake label(0)
                labels_real = torch.full((real_samples.size(0),), Label.REAL,
                                         dtype=torch.float, device=self.device)
                labels_fake = labels_real.clone().fill_(Label.FAKE)

                # calculate the loss and predicted value from real samples
                netD_predictions_real = self.discriminator(
                    real_samples).view(-1)
                netD_loss_real = criterion(netD_predictions_real, labels_real)
                netD_loss_real.backward()

                # Testing discriminator on fake samples
                noise = torch.randn(real_samples.size(0), self.nz, 1, 1,
                                    device=self.device)
                # fake bact of samples created with generator
                fake_samples = self.generator(noise)

                # calculate the predicted value and loss from fake samples
                netD_predictions_fake = self.discriminator(
                    fake_samples.detach()).view(-1)
                netD_loss_fake = criterion(netD_predictions_fake, labels_fake)
                netD_loss_fake.backward()

                # calculate the total loss of discriminator
                netD_loss = netD_loss_real + netD_loss_fake
                self.optim_disc.step()

                # train netG
                self.generator.zero_grad()

                # loss of generator
                netG_output = self.discriminator(fake_samples).view(-1)
                netG_loss = criterion(netG_output, labels_real)
                netG_loss.backward()

                # optimizes generator using BCELoss
                self.optim_gen.step()

                # save loss of both D(x) and G(x) and f1 score
                # for further visualization
                self.D_losses.append(netD_loss.item())
                self.G_losses.append(netG_loss.item())

                f1, precision, recall = self.discriminator.calc_f1_score(
                    netD_predictions_real, netD_predictions_fake,
                    labels_real, labels_fake
                )
                self.f1_scores.append(f1)
                self.precision.append(precision)
                self.recall.append(recall)

                # Print and save losses and generated images
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                          % (epoch, self.num_epochs, i, len(self.dataloader),
                             netD_loss.item(), netG_loss.item()))

                # if statement used for printing images
                if ((i + 1) % nth_iteration == 0) or (
                        (epoch == self.num_epochs) and (i == len(self.dataloader) - 1)):
                    self.save_iteration_images(
                        fixed_noise, epoch, i, nth_iteration)
                    fake_samples = (fake_samples - torch.min(fake_samples)) / \
                        (torch.max(fake_samples) - torch.min(fake_samples))
                    IS, IS_std = get_inception_score(
                        fake_samples, use_torch=True)
                    self.inception_scores_mean.append(IS)
                    self.inception_scores_std.append(IS_std)

                    self.plot_charts_for_gan(epoch)

                    if self.model_save:
                        self.save_model()

    def save_iteration_images(self, fixed_noise, epoch, iteration, nth_iteration):
        with torch.no_grad():
            self.generator.eval()
            fake = self.generator(fixed_noise).detach().cpu()
            image = vutils.make_grid(
                fake, padding=2, normalize=True
            )

            self.img_list.append(image)
            save_img_generated(image,
                               f"fig-epoch{epoch}-{self.num_epochs}"
                               f"-itr{(iteration + 1) // nth_iteration}"
                               f"-{len(self.dataloader) // nth_iteration}.png")

    def plot_charts_for_gan(self, epoch):
        """plots data for analyzing runs. Added to store data
        regardless if idun terminates the code
        tries low_value to see the low scores with more details"""

        loss = SubFigure("Loss", [IterationValues("G(x)", self.G_losses),
                                  IterationValues("D(x)", self.D_losses)])
        loss_clipped = loss
        loss_clipped.ylim = 10
        f1_prec_recall_figure = SubFigure(
            "F1, precision and recall for discriminator",
            [IterationValues("F1", self.f1_scores),
             IterationValues("Precision", self.precision),
             IterationValues("Recall", self.recall)])
        incept_score = SubFigure("Inception Score", [IterationValues(
            "Inception Score", self.inception_scores_mean),
            IterationValues("Inception Score std", self.inception_scores_std)
        ])

        plot_iteration_values(
            loss_clipped,
            f1_prec_recall_figure,
            incept_score,
            title=f"Clipped loss, F1 score, and inception for epoch {epoch}-{self.num_epochs}",
            file_name="fig-loss-f1-inception-clipped.png")

        plot_iteration_values(
            loss,
            f1_prec_recall_figure,
            incept_score,
            title=f"Loss and F1 score for epoch{epoch}-{self.num_epochs}",
            file_name="fig-loss-f1-inception.png")

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
