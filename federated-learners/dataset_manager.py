import numpy as np
import torchvision

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class SubLoader(torchvision.datasets.CIFAR10):
    def __init__(self, *args, include_list=[], **kwargs):
        super(SubLoader, self).__init__(*args, **kwargs)

        exclude_items = list(classes)
        for item in include_list:
            exclude_items.remove(item)
        exclude_list = [classes.index(item) for item in exclude_items]
        print("Generating DataLoader with classes:", include_list)

        if self.train:
            labels = np.array(self.targets)
            exclude = np.array(exclude_list).reshape(1, -1)
            mask = ~(labels.reshape(-1, 1) == exclude).any(axis=1)

            self.data = self.data[mask]
            self.targets = labels[mask].tolist()
        else:
            labels = np.array(self.targets)
            exclude = np.array(exclude_list).reshape(1, -1)
            mask = ~(labels.reshape(-1, 1) == exclude).any(axis=1)

            self.data = self.data[mask]
            self.targets = labels[mask].tolist()
