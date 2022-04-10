import torch
import torch.nn.functional as F


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def eval_reconstruction(net, testloader):
    total, loss = 0, 0.0
    with torch.no_grad():
        for data in testloader:
            images = data[0].to(DEVICE)
            recon_images, _, _ = net(images)
            loss += F.binary_cross_entropy(images, recon_images)
            total += len(images)
    return loss / total
