# graf som ser på D(x) loss og G(x) loss
# printe ut bilder etter hver epoch og på starten (2 forskjllige metoder)
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np


def print_start_img(dataloader, grid_size=(8, 8), title=""):
    # Get a batch of real images
    real_batch = next(iter(dataloader))

    # Ensure grid size is within the bounds of the batch size
    grid_size = (min(grid_size[0], real_batch[0].size(0)), grid_size[1])

    # Plot the images
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.title(title)

    # Display a grid of images
    image_grid = vutils.make_grid(real_batch[0][:grid_size[0] * grid_size[1]], padding=2, normalize=True)
    plt.imshow(np.transpose(image_grid.numpy(), (1, 2, 0)))
    plt.show()


def print_epoch_img():
    return 0


# method that shows Generator and Discriminator loss during training
def print_loss(G_loss, D_loss):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss")
    plt.plot(G_loss, label="G")
    plt.plot(D_loss, label="D")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
