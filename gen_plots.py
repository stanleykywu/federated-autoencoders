import numpy as np
import pathlib

from matplotlib import pyplot as plt

for p in pathlib.Path("models").iterdir():
    if p.is_dir() or p.__str__() == "models/Image_VAE.py":
        continue
    metrics = np.load(p, allow_pickle=True)
    backprop_training_loss = []
    recon_training_loss = []
    if "central" in p.__str__():
        continue
    if "eval" in p.__str__():
        plt.plot([i for i in range(len(metrics))], metrics)
        plt.xlabel("Training Iteration")
        plt.ylabel("Average Test KL + MSE Loss")
        plt.title(f"{p.parts[1].split('_')[0]}: {p.parts[1].split('_')[1]}")
        plt.savefig("plots/" + p.parts[1].__str__() + ".png", bbox_inches='tight')
        plt.clf()
        print(p.parts[1].split('_')[1], metrics[-1])
        continue
    for metric in metrics:
        backprop_training_loss += [metric["Training backprop loss"]]
        recon_training_loss += [metric["Training recon loss"]]

    plt.plot([i for i in range(len(recon_training_loss))], recon_training_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Joint MSE + KL Loss")
    plt.title(f"{p.parts[1].split('_')[0]}: {p.parts[1].split('_')[1]}")
    plt.savefig("plots/" + p.parts[1].__str__() + ".png", bbox_inches='tight')
    plt.clf()

cifar = []
gtsrb = []
fmnist = []
for p in pathlib.Path("models").iterdir():
    if "central" in p.__str__():
        if "cifar" in p.__str__():
            cifar += [np.load(p, allow_pickle=True)]
        if "gtsrb" in p.__str__():
            gtsrb += [np.load(p, allow_pickle=True)]
        if "fmnist" in p.__str__():
            fmnist += [np.load(p, allow_pickle=True)]

print(gtsrb)

plt.plot([i for i in range(len(cifar))], cifar)
plt.xlabel("Training Iteration")
plt.ylabel("Joint MSE + KL Loss")
plt.title("CIFAR Centralized Performance")
plt.savefig("plots/cifar_central.png", bbox_inches='tight')
plt.clf()
plt.plot([i for i in range(len(gtsrb))], gtsrb)
plt.xlabel("Training Iteration")
plt.ylabel("Joint MSE + KL Loss")
plt.title("GTSRB Centralized Performance")
plt.savefig("plots/gtsrb_central.png", bbox_inches='tight')
plt.clf()
plt.plot([i for i in range(len(fmnist))], fmnist)
plt.xlabel("Training Iteration")
plt.ylabel("Joint MSE + KL Loss")
plt.title("FMNIST Centralized Performance")
plt.savefig("plots/fmnist_central.png", bbox_inches='tight')
plt.clf()