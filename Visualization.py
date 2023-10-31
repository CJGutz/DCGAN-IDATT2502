import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np


def print_start_img(dataloader, grid_size=(8, 4), title="Starting Images"):
    real_batch = next(iter(dataloader))
    max_cols = min(grid_size[1], int(real_batch[0].size(0) / grid_size[0]))
    grid_size = (min(grid_size[0], real_batch[0].size(0)), max_cols)

    # Plot images
    plt.figure(figsize=(5, 5))
    plt.axis("off")
    plt.title(title)

    # Display images
    image_grid = vutils.make_grid(real_batch[0][:grid_size[0] * grid_size[1]], padding=2, normalize=True,
                                  nrow=grid_size[0])
    plt.imshow(np.transpose(image_grid.numpy(), (1, 2, 0)))
    plt.show()


def print_epoch_images(dataloader, img_list, device):
    real_batch = next(iter(dataloader))

    plt.figure(figsize=(12, 12))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0))
    )

    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.show()


# method that shows Generator and Discriminator loss during training
def print_loss(G_loss, D_loss):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss")
    plt.plot(G_loss, label="G(x)")
    plt.plot(D_loss, label="D(x)")
    plt.xlabel("Iters")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
