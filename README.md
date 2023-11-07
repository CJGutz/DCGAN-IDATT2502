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

To train a DCGAN model with the given default values you run.
```console
   python3 Entrypoint.py dcgan MNIST --channels 1
```
It is also possible to train other models.
Find available models in [Entrypoint.py](./Entrypoint.py)

Remember to save and load models if you want to persist your training.
The models will be saved even if an error occurrs in the training or if you shut the training down yourself.
```console
   python3 Entrypoint.py dcgan MNIST --channels 1 --load-model --save-model
```

## Datasets

Set your preferred dataset directory or zip path as the argument to use it.
If you want to use a dataset from torchvision, you can see the [DatasetLoader.py](./DatasetLoader.py) file
for a list of supported datasets. Just use the name of the dataset as the argument.

Example Script: 
```console
python3 Entrypoint.py gan datasets/celeba-dataset -g lsgan -c 3 -i 64 -l 3 -b 128 -e 5 -lr 0.0002 -b1 0.5 --ndf 64 --ngf 64 --nz 100
```
