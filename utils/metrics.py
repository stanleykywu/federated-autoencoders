import torch
import torch.nn.functional as F

from tqdm import tqdm


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def eval_reconstruction(net, testloader):
    total, recon_loss = 0, 0.0
    with torch.no_grad():
        for data in tqdm(testloader, desc=f"Evaluating recon loss on VAE"):
            images = data[0].to(DEVICE)
            recon_images, _, _ = net(images)
            recon_loss += F.mse_loss(recon_images, images)
            total += len(images)
    return recon_loss / total
