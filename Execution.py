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
# Define DCGAN class (in DCGAN.py)
from DCGAN import DCGAN as gan


def download_and_extract_zip(zip_file_path, extract_path, desc='Extracting files'):
    # If dateset isn't in the directory its downloaded from the zipfile
    if not os.path.isdir(extract_path):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:

            # Get the total number of files in the zip archive
            total_files = len(zip_ref.namelist())
            pbar = tqdm(total=total_files, desc=desc)

            # Extract each file and update the progress bar
            for file in zip_ref.namelist():
                zip_ref.extract(file, extract_path)
                pbar.update(1)
            pbar.close()


def data_loader(dataset_numb, image_size, batch_size, ds_root="./datasets"):
    if dataset_numb == 1:
        # number of color channels, since its grey scaling, 1 is need, and not 3 (RGB)
        nc = 1
        # needed since the dataset is differently put than the others
        transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset = torchvision.datasets.MNIST(root=ds_root, train=True, download=True, transform=transform)
    else:
        # rest of datasets have 3 color channels
        nc = 3
        transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        if dataset_numb == 2:
            zip_file_path = f'{ds_root}/img_align_celeba.zip'
            extract_path = f'{ds_root}/celeba-dataset'
            download_and_extract_zip(zip_file_path, extract_path)

            dataset = dset.ImageFolder(root=extract_path, transform=transform)

        elif dataset_numb == 3:
            dataloader = 3

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print_start_img(dataloader)

    return dataloader, nc


def run(dataset_numb, img_size, batch_size):
    dataset_numb = dataset_numb
    img_size = img_size

    # flyttes til GAN-klassen etter hvert
    beta1 = 0.5
    lr = 0.0002

    batch_size = batch_size
    dataloader, channel_number = data_loader(dataset_numb, img_size, batch_size)

    # further implementation needed


if __name__ == "__main__":
    run(2, 64, 32)
