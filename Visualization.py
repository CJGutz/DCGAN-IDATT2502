from dataclasses import dataclass
import os
from typing import List
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

    plt.savefig(os.path.join("datasets", "figures", "start_fig.png"))
    plt.close()


def save_img_generated(img_list, image_name):

    plt.figure(figsize=(12, 12))
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.axis("off")
    plt.savefig(os.path.join("datasets", "figures", image_name))
    plt.close()


@dataclass
class IterationValues:
    label: str
    values: list


@dataclass
class SubFigure:
    y_label: str
    values_labels: List[IterationValues]


def plot_iteration_values(
    *graphs: SubFigure,
    title="Iteration values",
    file_name="iteration_values.png"
):
    """Creates subplots for to visualize simple graphs

    Args:
        graphs (SubFigure): SubFigure objects that contain the values
        and labels for the graphs
    """
    _, axes = plt.subplots(len(graphs), figsize=(10, 10))

    plt.title(title)
    for count, sub_figure in enumerate(graphs):
        ax = axes[count]
        for plots in sub_figure.values_labels:
            ax.plot(plots.values, label=plots.label)
        ax.set_xlabel("Iterations")
        ax.set_ylabel(sub_figure.y_label)
        ax.grid()
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("datasets", "figures", file_name))
