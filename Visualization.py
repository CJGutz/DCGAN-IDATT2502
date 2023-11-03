import os
import math
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils


def print_start_img(dataloader, grid_imgs=(8, 5), title="Starting Images"):
    image_batch = next(iter(dataloader))
    plt.figure(figsize=(5, 5))
    plt.axis("off")
    plt.title(title)

    # Display images
    plt.imshow(np.transpose(vutils.make_grid(
        image_batch[0][:grid_imgs[0] * grid_imgs[1]],
        padding=2, normalize=True), (1, 2, 0)))

    path = os.path.join("datasets", "figures", "start_fig.png")
    plt.savefig(path)
    plt.close()


def print_epoch_images(dataloader, img_list, epoch, num_epochs, itr, data_len, grid_imgs=(8, 8)):
    real_batch = next(iter(dataloader))

    plt.figure(figsize=(12, 12))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(
        real_batch[0][:grid_imgs[0] * grid_imgs[1]],
        padding=2, normalize=True), (1, 2, 0)))

    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))

    plt.savefig(os.path.join("datasets", "figures",
                f"epoch_fig-epoch{epoch}-{num_epochs}-itr{math.floor(itr)}-{math.floor(data_len)}.png"))
    plt.close()


# method that shows Generator and Discriminator loss during training
def print_loss(G_loss, D_loss):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss")
    plt.plot(G_loss, label="G(x)")
    plt.plot(D_loss, label="D(x)")
    plt.xlabel("Iters")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join("datasets", "figures", "loss_fig.png"))
    plt.close()
