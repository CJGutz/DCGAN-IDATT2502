import torch
import zipfile
from torchvision.transforms import transforms
import torchvision.datasets as dset
import os
from tqdm import tqdm


VISION_DATASETS = {
    "MNIST": dset.MNIST,
    "FashionMNIST": dset.FashionMNIST,
    "CIFAR10": dset.CIFAR10,
}


def download_and_extract_zip(zip_file_path, extract_path):
    if not os.path.isdir(extract_path):
        with zipfile.ZipFile(zip_file_path) as zip_ref:
            for file in tqdm(zip_ref.namelist()):
                zip_ref.extract(file, extract_path)


def data_loader(dataset_path, image_size, batch_size, channels, dataset_dir="./datasets"):
    normalization_args = list((0.5 for _ in range(channels)))

    transform = transforms.Compose(
        [transforms.Resize(image_size),
         transforms.CenterCrop(image_size),
         transforms.ToTensor(),
         transforms.Normalize(normalization_args, normalization_args)
         ])

    if dataset_path in VISION_DATASETS.keys():
        dataset = VISION_DATASETS[dataset_path]
        dataset = dataset(root=dataset_dir, train=True,
                          transform=transform, download=True,)
        model_name = dataset_path
    else:
        if dataset_path.endswith('.zip'):
            zip_path = dataset_path
            dataset_path = dataset_path.replace('.zip', '')
            download_and_extract_zip(
                zip_path, dataset_path)

        dataset = dset.ImageFolder(root=dataset_path, transform=transform)
        model_name = os.path.basename(dataset_path)
        if not model_name:
            model_name = os.path.basename(os.path.dirname(dataset_path))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=1, drop_last=True, pin_memory=False)

    return dataloader, model_name
