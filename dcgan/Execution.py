import torch
import sys
import signal
import argparse
from Visualization import (
    IterationValues,
    SubFigure,
    print_start_img,
    plot_iteration_values
)
from dcgan.DCGAN import DCGAN as dcgan
from dcgan.Discriminator import Discriminator
from dcgan.Generator import Generator
from DatasetLoader import data_loader


def run(cli_args):
    parser = argparse.ArgumentParser(
        prog="DCGAN implementation",
    )
    parser.add_argument("dataset", type=str)
    parser.add_argument("-c", "--channels", required=True,
                        type=int, choices=[1, 3])
    parser.add_argument("-i", "--img-size", default=64, type=int)
    parser.add_argument("-l", "--layers", default=4, type=int)
    parser.add_argument('-b', '--batch-size', default=128, type=int)
    parser.add_argument('-e', '--epochs', default=5, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.0002, type=float)
    parser.add_argument("-b1", "--beta1", default=0.5, type=float)
    parser.add_argument("--ndf", default=64, type=int,
                        help="discriminator features")
    parser.add_argument("--ngf", default=64, type=int,
                        help="generator features")
    parser.add_argument("--nz", default=100, type=int,
                        help="generator noise size")
    parser.add_argument("--load-model", action="store_true", default=False)
    parser.add_argument("--no-model-save", action="store_true", default=False)

    args = parser.parse_args(cli_args)

    # Check if CUDA (GPU support) is available
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print("GPU number available: ", gpu_count)
    else:
        gpu_count = 0
        print("CUDA not available")

    dataloader, model_name = data_loader(
        args.dataset, args.img_size, args.batch_size, args.channels)
    print_start_img(dataloader)

    # Device is based on CUDA available gpu
    device = torch.device("cuda:0" if (
        torch.cuda.is_available() and gpu_count > 0) else "cpu")
    # Create an instance of discriminator and generator
    generator = Generator(args.nz, args.ngf, args.channels,
                          args.layers).to(device)
    discriminator = Discriminator(
        args.channels, args.ndf, args.layers).to(device)
    # Create an instance of the dcgan
    gan = dcgan(generator, discriminator, args.epochs, dataloader, model_name,
                args.channels, device,
                args.batch_size, args.learning_rate, args.beta1,
                args.nz, args.load_model)

    def tear_down(signal, frame):
        if not args.no_model_save:
            print(f"Saving model {model_name}")
            gan.save_model()
            print("Model saved")
        plot_iteration_values(
            SubFigure("Loss", [IterationValues("G(x)", gan.G_losses),
                               IterationValues("D(x)", gan.D_losses)]),
            SubFigure("Accuracy", [IterationValues("G(x)", gan.G_accuracies),
                                   IterationValues("D(x)", gan.D_accuracies)]),
            title="Loss and Accuracy",
            file_name=f"{model_name}-accuracy-loss.png")
        sys.exit(0)

    signal.signal(signal.SIGINT, tear_down)
    try:
        gan.train()
    except Exception as e:
        print(e)

    tear_down(None, None)
