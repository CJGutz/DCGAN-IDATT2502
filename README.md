# Deep Convolutional GAN (DCGAN)

<a name="readme-top"></a>

This project is part of the course *IDATT2502 - Applied Machine Learning* at the Norwegian University of Science and Technology (NTNU) by:

- Carl Johan Gützkow - [GitHub](https://github.com/CJGutz)
- Eirik Elvestad - [GitHub](https://github.com/eirikelv)
- Tomas Beranek - [GitHub](https://github.com/tomasbera)

## Overview

In this project, we explore various applications and metrics for **Generative Adversary Networks**.

## Installation and Execution

1. Clone the repository:
    ```console
    git clone https://github.com/CJGutz/DCGAN-IDATT2502.git
    ```

    OR

    ```console
    git clone git@github.com:CJGutz/DCGAN-IDATT2502.git
    ```

2. Download the required packages:
    ```console
    pip install -r requirements.txt
    ```

3. To train a DCGAN model with default values:
    ```console
    python3 Entrypoint.py dcgan MNIST --channels 1
    ```

    To save and load models:
    ```console
    python3 Entrypoint.py dcgan MNIST --channels 1 --load-model --save-model
    ```

## Datasets

Specify your preferred dataset directory or zip path as an argument to use it. For torchvision datasets, refer to [DatasetLoader.py](./DatasetLoader.py) for supported datasets.

### Example Script:

```console
python3 Entrypoint.py dcgan datasets/celeba-dataset -c 3 -i 64 -l 3 -b 128 -e 5 -lr 0.0002 -b1 0.5 --ndf 64 --ngf 64 --nz 100
```

### Tested Datasets
These datasets are already a part of the program and can be directly accessed from the command line using the specific 
titles listed in the table below.

| Datasets       | Description                                             |
|----------------|---------------------------------------------------------|
| MNIST          | Handwritten digits (0-9) in 28x28 grayscale images      |
| celeba-dataset | 200,000+ celebrity images with 40 attribute labels each |
| FashionMNIST   | 28x28 grayscale images of 10 fashion categories         |
| CIFAR10        | 60,000 32x32 color images across 10 classes             |

<br>
If the celeba dataset is to be used follow the steps beneath

1. Download the zipfile from "https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg?resourcekey=0-rJlzl934LzC-Xp28GeIBzQ"

    <br>

2. Place the zipfile into your dataset folder or directory

<p align="right">(<a href="#readme-top">back to top</a>)</p>
