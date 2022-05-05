import numpy as np
import pathlib
import pandas as pd

from matplotlib import pyplot as plt


cifar = pd.DataFrame()
fmnist = pd.DataFrame()
gtsrb = pd.DataFrame()
for p in pathlib.Path("metrics").iterdir():
    if p.is_dir() or p.__str__() == "metrics/Image_VAE.py" or "central" in p.__str__():
        continue
    metrics = np.load(p, allow_pickle=True)
    backprop_training_loss = []
    recon_training_loss = []
    print(metrics)
    if "central" in p.__str__():
        continue

    if "cifar" in p.__str__():
        cifar.append(np.load(p, allow_pickle=True))
    if "gtsrb" in p.__str__():
        gtsrb.append(np.load(p, allow_pickle=True))
    if "fmnist" in p.__str__():
        fmnist.append(np.load(p, allow_pickle=True))

print(cifar)
print(gtsrb)
print(fmnist)
