# metode run() som kjører programmet, metode for å laste in dataset
import os
import zipfile
import tqdm
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from tqdm import tqdm

# Define Visualization class (in Visualization.py)
from Visualization import print_start_img


def data_loader(dataset_numb, image_size, batch_size):
    ds_root = "./datasets"
    extract_path = f'{ds_root}/celeba-dataset'

    if dataset_numb == 1:
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
                                   transform=transforms.Compose([
                                       transforms.Resize(image_size),
                                       transforms.CenterCrop(image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))

    elif dataset_numb == 2:
        dataset = 2

    elif dataset_numb == 3:
        dataset = 3

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Plot some training images
    print_start_img(dataloader)

    return dataloader


def run(dataset_numb, img_size, batch_size):
    dataset_numb = dataset_numb
    img_size = img_size

    # flyttes til GAN-klassen etter hvert
    beta1 = 0.5
    lr = 0.0002

    # flyttes til dataloader-metoden etter hvert
    numb_channels = 3

    batch_size = batch_size
    dataloader = data_loader(dataset_numb, img_size, batch_size)

    # further implementation needed


if __name__ == "__main__":
    run(1, 64, 32)
