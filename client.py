from collections import OrderedDict
import argparse

from flwr.server.strategy import FedAvg

from models.Image_VAE import ImageVAE
from utils.metrics import eval_backprop_loss, eval_reconstruction

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset.dataset_manager import *

import flwr as fl

from tqdm import tqdm

import matplotlib.pyplot as plt


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["fmnist", "gtsrb", "cifar10"])
    parser.add_argument("--classes", "--names-list", nargs="+", default=[])
    parser.add_argument("--epochs", nargs="?", const=10, type=int, default=10)
    parser.add_argument("--latent_size", nargs="?", const=10, type=int, default=10)
    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.add_argument("--type", dest="type", choices=["client", "server"])
    return parser.parse_args()


def load_data(dataset: str):
    """Load dataset (training and test set)."""
    if dataset == "fmnist":
        transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.Resize((32, 32)),
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
    elif dataset == "gtsrb":
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


def train(net, trainloader, epochs, testloader=None, verbose=False):
    """Train the network on the training set."""
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    metrics = []
    for i in tqdm(range(epochs), desc=f"Training VAE on {epochs} epochs"):
        for images, _ in trainloader:
            optimizer.zero_grad()
            recon_images, mu, logvar = net(images)
            recon_loss = F.mse_loss(recon_images, images)
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 0.05 * kld_loss
            loss.backward()
            optimizer.step()

        trn_loss = eval_backprop_loss(net, trainloader)
        trn_reconstruction_loss = eval_reconstruction(net, trainloader)

        epoch_metrics = {
            "Training backprop loss": float(trn_loss),
            "Training recon loss": float(trn_reconstruction_loss),
        }

        if testloader:
            tst_loss = eval_backprop_loss(net, testloader)
            tst_reconstruction_loss = eval_reconstruction(net, testloader)
            epoch_metrics["Testing backprop loss"] = float(tst_loss)
            epoch_metrics["Testing recon loss"] = float(tst_reconstruction_loss)

        metrics.append(epoch_metrics)

        if verbose:
            print("Metrics at epoch {}: {}".format(i, metrics))

    return metrics


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


def save_images(net, loader, classes, call_num, num_images=0):
    """Save the generated images for the trained VAE"""
    seen_images = 0
    for batch in loader:
        images = batch[0].to(DEVICE)
        recon_images, _, _ = generate(net, images)
        for i, (recon_image, image) in enumerate(zip(recon_images, images)):
            plt.imshow(np.squeeze(image).T)
            plt.savefig(f"images/original_{i}_{call_num}_{classes}.png", dpi=100)
            plt.clf()
            plt.imshow(np.squeeze(recon_image).T)
            plt.savefig(f"images/generated_{i}_{call_num}_{classes}.png", dpi=100)
            plt.clf()
            seen_images += 1
            if num_images and num_images == seen_images:
                break
        break


def main(args):
    # Load model and data
    net = ImageVAE(latent_size=args.latent_size)
    trainloader, testloader = load_data(args.dataset)

    class Client(fl.client.NumPyClient):
        def __init__(self):
            super(Client, self).__init__()
            self.calls = 0

        def get_parameters(self):
            print("sending client parameters...")
            return [val.cpu().numpy() for _, val in net.state_dict().items()]

        def set_parameters(self, parameters):
            print("updating client parameters...")
            # params_dict = zip(net.state_dict().keys(), parameters)
            # state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            # net.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            train_metrics = train(
                net,
                trainloader,
                epochs=args.epochs,
                testloader=testloader if args.verbose else None,
                verbose=args.verbose,
            )
            np.save(f"models/{args.dataset}_{args.classes}_{args.epochs}_train_metrics", train_metrics)
            print(train_metrics)
            save_images(net, trainloader, args.classes, self.calls, 2)
            self.calls += 1
            return self.get_parameters(), len(trainloader), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            tst_loss = eval_backprop_loss(net, testloader)
            return float(tst_loss), len(testloader), {}

    if args.type == "client":
        fl.client.start_numpy_client("localhost:8080", client=Client())
    else:
        strategy = FedAvg(min_available_clients=10, min_fit_clients=10, min_eval_clients=10,
                          eval_fn=eval_fn_wrapper(net, testloader))
        fl.server.start_server(
            "localhost:8080", config={"num_rounds": 3}, strategy=strategy
        )


def eval_fn_wrapper(net, testloader):
    def eval_fn(parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        tst_loss = eval_backprop_loss(net, testloader)
        return float(tst_loss), {}

    return eval_fn

if __name__ == "__main__":
    args = run_argparse()
    main(args)
