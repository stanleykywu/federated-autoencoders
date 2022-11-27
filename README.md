Read our paper [here](https://github.com/stanleykywu/federated-autoencoders/blob/main/Federated%20Autoencoders.pdf)

In order to run our experiments please see client.py for CLI implementation details.

The help from our CLI tool shows the following:
```bash
git:(main) -> python client.py --help
usage: client.py [-h] [--dataset {fmnist,gtsrb,cifar10}] [--classes CLASSES [CLASSES ...]] [--epochs [EPOCHS]] [--latent_size [LATENT_SIZE]] [--verbose] [--type {client,server}]

optional arguments:
  -h, --help            show this help message and exit
  --dataset {fmnist,gtsrb,cifar10}
  --classes CLASSES [CLASSES ...], --names-list CLASSES [CLASSES ...]
  --epochs [EPOCHS]
  --latent_size [LATENT_SIZE]
  --verbose
  --type {client,server}
```

Please note that you will need to update the number of clients in the client.py at the Line 237 based on the number of clients.

Please see our files in `log/` for examples of usage with multiple clients, one for each class and the expected outputs.
