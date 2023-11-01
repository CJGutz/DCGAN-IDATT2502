import os
import zipfile
import tqdm
import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, argument_validation
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


def download_and_extract_zip(zip_file_path, extract_path):
    if not os.path.isdir(extract_path):
        with zipfile.ZipFile(zip_file_path) as zip_ref:
            for file in tqdm(zip_ref.namelist()):
                zip_ref.extract(file, extract_path)


def data_loader(dataset_numb, image_size, batch_size, ds_root="./datasets"):
    if dataset_numb == 1:
        # number of color channels, since its grey scaling, 1 is need, and not 3 (RGB)
        nc = 1
        # needed since the dataset is differently put than the others
        transform = transforms.Compose([transforms.Resize(
            image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])
        dataset = torchvision.datasets.MNIST(
            root=ds_root, train=True, download=True, transform=transform)
    else:
        # rest of datasets have 3 color channels
        nc = 3
        transform = transforms.Compose(
            [transforms.Resize(image_size),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ])

        if dataset_numb == 2:
            zip_file_path = f'{ds_root}/img_align_celeba.zip'
            extract_path = f'{ds_root}/celeba-dataset'
            download_and_extract_zip(zip_file_path, extract_path)

            dataset = dset.ImageFolder(root=extract_path, transform=transform)

        elif dataset_numb == 3:
            dataloader = 3

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True)
    print_start_img(dataloader)

    return dataloader, nc


def run(dataset_numb):

    parser = argparse.ArgumentParser(
        prog="DCGAN implementation",
    )
    parser.add_argument("dataset")
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
        gpu_count = torch.cuda.device_count()
        print("GPU number available: ", gpu_count)
    else:
        gpu_count = 0
        print("CUDA not available")

    dataloader, channel_number = data_loader(
        dataset_numb, args.img_size, args.batch_size)

    # Device is based on CUDA available gpu
    device = torch.device("cuda:0" if (
        torch.cuda.is_available() and gpu_count > 0) else "cpu")

    # Create an instance of discriminator and generator
    generator = netG(args.nz, args.ngf, channel_number, args.layers)
    discriminator = netD(channel_number, args.ndf, args.layers)

    # Create an instance of the dcgan
    gan = dcgan(args.epochs, dataloader, channel_number, device, generator, discriminator,
                args.batch_size, args.lr, args.beta1, args.nz)

    gan.train()


if __name__ == "__main__":
    run(2)
