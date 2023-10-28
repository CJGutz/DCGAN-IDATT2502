# graf som ser på D(x) loss og G(x) loss
# printe ut bilder etter hver epoch og på starten (2 forskjllige metoder)
import matplotlib.pyplot as plt


def print_start_img():
    return 0


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
