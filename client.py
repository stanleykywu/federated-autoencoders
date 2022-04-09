from collections import OrderedDict
import argparse

from models.Image_VAE import ImageVAE

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST, GTSRB, CIFAR10
from dataset.dataset_manager import *

import flwr as fl

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["fmnist", "gtrsb", "cifar10"])
    parser.add_argument("--classes", "--names-list", nargs="+", default=[])
    parser.add_argument("--epochs", nargs="?", const=10, type=int, default=10)
    parser.add_argument("--latent_size", nargs="?", const=10, type=int, default=10)
    return parser.parse_args()


def load_data(dataset: str):
    """Load dataset (training and test set)."""
    if dataset == "fmnist":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ],
        )
        trainset = FashionMNISTSubloader(
            ".",
            to_include=args.classes,
            train=True,
            download=True,
            transform=transform,
        )
        testset = FashionMNISTSubloader(
            ".",
            to_include=args.classes,
            train=False,
            download=True,
            transform=transform,
        )
    elif dataset == "cifar10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.Resize((32, 32)),
            ]
        )
        trainset = CIFAR10SubLoader(
            ".",
            to_include=args.classes,
            train=True,
            download=True,
            transform=transform,
        )
        testset = CIFAR10SubLoader(
            ".",
            to_include=args.classes,
            train=False,
            download=True,
            transform=transform,
        )
    elif dataset == "gtrsb":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.Resize((32, 32)),
            ]
        )
        trainset = GTSRBSubloader(
            ".",
            to_include=args.classes,
            split="train",
            download=True,
            transform=transform,
        )
        testset = GTSRBSubloader(
            ".",
            to_include=args.classes,
            split="test",
            download=True,
            transform=transform,
        )
    else:
        raise NotImplementedError

    trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
    testloader = DataLoader(testset, batch_size=128)
    return trainloader, testloader


def train(net, trainloader, epochs):
    """Train the network on the training set."""
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    for _ in range(epochs):
        for images, _ in trainloader:
            optimizer.zero_grad()
            recon_images, mu, logvar = net(images)
            recon_loss = F.mse_loss(recon_images, images)
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 0.05 * kld_loss
            loss.backward()
            optimizer.step()


def test(net, testloader):
    """Validate the network on the entire test set."""
    total, loss = 0, 0.0
    with torch.no_grad():
        for data in testloader:
            images = data[0].to(DEVICE)
            recon_images, mu, logvar = net(images)
            recon_loss = F.mse_loss(recon_images, images)
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss += recon_loss + kld_loss
            total += len(images)
    return loss / total


def sample(net):
    """Generates samples using the decoder of the trained VAE."""
    with torch.no_grad():
        z = torch.randn(10)
        z = z.to(DEVICE)
        gen_image = net.decode(z)
    return gen_image


def generate(net, image):
    """Reproduce the input with trained VAE."""
    with torch.no_grad():
        return net.forward(image)


def main(args):
    # Load model and data
    net = ImageVAE(latent_size=args.latent_size)
    trainloader, testloader = load_data(args.dataset)

    class Client(fl.client.NumPyClient):
        def get_parameters(self):
            return [val.cpu().numpy() for _, val in net.state_dict().items()]

        def set_parameters(self, parameters):
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            train(net, trainloader, epochs=args.epochs)
            return self.get_parameters(), len(trainloader), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss = test(net, testloader)
            return float(loss), len(testloader), {}

    fl.client.start_numpy_client("[::]:8080", client=Client())


if __name__ == "__main__":
    args = run_argparse()
    main(args)
