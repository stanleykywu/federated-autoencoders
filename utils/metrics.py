import torch
import torch.nn.functional as F

from tqdm import tqdm


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def eval_reconstruction(net, testloader):
    """Validate the network on just reconstruction loss."""
    total, recon_loss = 0, 0.0
    with torch.no_grad():
        for data in testloader:
            images = data[0].to(DEVICE)
            recon_images, _, _ = net(images)
            recon_loss += F.mse_loss(recon_images, images)
            total += len(images)
    return recon_loss / total


def eval_backprop_loss(net, testloader):
    """Validate the network on backprop loss (recon + kl divergence)"""
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
