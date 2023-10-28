# metode run() som kjører programmet, metode for å laste in dataset
import os
import zipfile
import tqdm
import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Define Visualization class (in Visualization.py)
from Visualization import print_start_img


def data_loader(dataset_numb, image_size, batch_size):
    ds_root = "./datasets"
    extract_path = f'{ds_root}/celeba-dataset'

    if dataset_numb == 1:
        # number of color channels, since its back and white, 1 is need, and not 3 (RGB)
        nc = 1
        # needed since the dataset is differently put than the others
        transform = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size),
                                        transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset = torchvision.datasets.MNIST(root=ds_root, train=True, download=True, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print_start_img(dataloader)
        return dataloader, nc

    # rest of datasets have 3 channels
    nc = 3
    transform = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

    if dataset_numb == 2:
        # If dateset isn't in the directory its downloaded from the zipfile
        if not os.path.isdir(extract_path):
            # Specify the path to the manually uploaded datasets zip file
            zip_file_path = f'{ds_root}/img_align_celeba.zip'

            # Get the total number of files in the zip archive
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                total_files = len(zip_ref.namelist())

            # Create a progress bar using tqdm
            pbar = tqdm(total=total_files, desc='Extracting files')
            # Extract each file and update the progress bar
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                for file in zip_ref.namelist():
                    zip_ref.extract(file, extract_path)
                    pbar.update(1)
            pbar.close()

        dataset = dset.ImageFolder(root=extract_path,
                                   transform=transform)

    elif dataset_numb == 3:
        dataset = 3

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Plot some training images
    print_start_img(dataloader)

    return dataloader, nc


def run(dataset_numb, img_size, batch_size):
    dataset_numb = dataset_numb
    img_size = img_size

    # flyttes til GAN-klassen etter hvert
    beta1 = 0.5
    lr = 0.0002

    batch_size = batch_size
    dataloader, nc = data_loader(dataset_numb, img_size, batch_size)

    # further implementation needed


if __name__ == "__main__":
    run(1, 64, 32)
