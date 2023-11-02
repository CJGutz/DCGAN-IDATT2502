# Deep Convolutional GAN
This is a project in the course *IDATT2502 - Applied Machine Learning* at the Norwegian University of Science and Technology (NTNU) by:
- Carl Johan Gützkow
- Eirik Elvestad
- Tomas Beranek
  
In this project we have developed a **Deep Convolutional Generative Adversarial Network**.

# How to

 ```console
    pip install -r requirements.txt
 ```

# Datasets

Set your preferred dataset directory or zip path as the argument to use it.
If you want to use a dataset from torchvision, you can see the [DatasetLoader.py](./DatasetLoader.py) file
for a list of supported datasets. Just use the name of the dataset as the argument.

Example Script: 
```console
    python3 Execution.py datasets/celeba-dataset -c 3 -i 64 -l 3 -b 128 -e 5 -lr 0.0002 -b1 0.5 --ndf 64 --ngf 64 --nz 100
 ```