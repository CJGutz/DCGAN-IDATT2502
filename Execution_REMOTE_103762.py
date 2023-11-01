import os
import zipfile
import tqdm
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse

# Define Visualization class (in Visualization.py)
from Visualization import print_start_img
# Define DCGAN class (in DCGAN.py)
from DCGAN import DCGAN as dcgan
# Define Discriminator class (in Discriminator.py)
from Discriminator import Discriminator as netD
# Define Generator class (in Generator.py)
from Generator import Generator as netG
from DatasetLoader import data_loader


def run():

    parser = argparse.ArgumentParser(
        prog="DCGAN implementation",
    )
    parser.add_argument("dataset", type=str)
    parser.add_argument("-c", "--channels", required=True,
                        type=int, choices=[1, 3])
    parser.add_argument("-i", "--img-size", default=64, type=int)
    parser.add_argument("-l", "--layers", default=4, type=int)
    parser.add_argument('-b', '--batch-size', default=128, type=int)
    parser.add_argument('-e', '--epochs', default=5, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.0002, type=float)
    parser.add_argument("-b1", "--beta1", default=0.5, type=float)
    parser.add_argument("--ndf", default=64, type=int)
    parser.add_argument("--ngf", default=64, type=int)
    parser.add_argument("--nz", default=100, type=int)

    args = parser.parse_args()

    # Check if CUDA (GPU support) is available
    if torch.cuda.is_available():
        # Get the number of available GPUs
        gpu_count = torch.cuda.device_count()
        print(f"Number of available GPUs: {gpu_count}")
    else:
        gpu_count = 0
        print("CUDA is not available on this system.")

    dataloader = data_loader(
        args.dataset, args.img_size, args.batch_size, args.channels)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (
        torch.cuda.is_available() and gpu_count > 0) else "cpu")

    # Create an instance of discriminator and generator
    generator = netG(args.nz, args.ngf, args.channels, args.layers)
    discriminator = netD(args.channels, args.ndf, args.layers)

    # Create an instance of the dcgan
    gan = dcgan(args.epochs, dataloader, args.channels, device, generator,
                discriminator, args.batch_size, args.learning_rate, args.beta1, args.nz)

    gan.train()


if __name__ == "__main__":
    run()
