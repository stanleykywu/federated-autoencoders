import numpy as np
import torchvision


class CIFAR10SubLoader(torchvision.datasets.CIFAR10):
    def __init__(self, *args, to_include=[], **kwargs):
        super(CIFAR10SubLoader, self).__init__(*args, **kwargs)

        self._classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )

        exclude_items = list(self._classes)
        for item in to_include:
            exclude_items.remove(item)
        exclude_list = [self._classes.index(item) for item in exclude_items]
        print("Generating CIFAR-10 DataLoader with classes:", to_include)

        labels = np.array(self.targets)
        exclude = np.array(exclude_list).reshape(1, -1)
        mask = ~(labels.reshape(-1, 1) == exclude).any(axis=1)

        self.data = self.data[mask]
        self.targets = labels[mask].tolist()


class FashionMNISTSubloader(torchvision.datasets.FashionMNIST):
    def __init__(self, *args, to_include=[], **kwargs):
        super(FashionMNISTSubloader, self).__init__(*args, **kwargs)

        self._classes = (
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        )

        exclude_items = list(self._classes)
        for item in to_include:
            exclude_items.remove(item)
        exclude_list = [self._classes.index(item) for item in exclude_items]
        print("Generating FashionMNIST DataLoader with classes:", to_include)

        labels = np.array(self.targets)
        exclude = np.array(exclude_list).reshape(1, -1)
        mask = ~(labels.reshape(-1, 1) == exclude).any(axis=1)

        self.data = self.data[mask]
        self.targets = labels[mask].tolist()


class GTSRBSubloader(torchvision.datasets.GTSRB):
    def __init__(self, *args, to_include=[], **kwargs):
        super(GTSRBSubloader, self).__init__(*args, **kwargs)

        self._classes = (
            "Speedlimit(20km/h)",
            "Speedlimit(30km/h)",
            "Speedlimit(50km/h)",
            "Speedlimit(60km/h)",
            "Speedlimit(70km/h)",
            "Speedlimit(80km/h)",
            "Endofspeedlimit(80km/h)",
            "Speedlimit(100km/h)",
            "Speedlimit(120km/h)",
            "Nopassing",
            "Nopassingvehover3.5tons",
            "Right-of-wayatintersection",
            "Priorityroad",
            "Yield",
            "Stop",
            "Novehicles",
            "Veh>3.5tonsprohibited",
            "Noentry",
            "Generalcaution",
            "Dangerouscurveleft",
            "Dangerouscurveright",
            "Doublecurve",
            "Bumpyroad",
            "Slipperyroad",
            "Roadnarrowsontheright",
            "Roadwork",
            "Trafficsignals",
            "Pedestrians",
            "Childrencrossing",
            "Bicyclescrossing",
            "Bewareofice/snow",
            "Wildanimalscrossing",
            "Endspeed+passinglimits",
            "Turnrightahead",
            "Turnleftahead",
            "Aheadonly",
            "Gostraightorright",
            "Gostraightorleft",
            "Keepright",
            "Keepleft",
            "Roundaboutmandatory",
            "Endofnopassing",
            "Endnopassingveh>3.5tons",
        )

        exclude_items = list(self._classes)
        for item in to_include:
            exclude_items.remove(item)
        exclude_list = [self._classes.index(item) for item in exclude_items]
        print("Generating GTSRB DataLoader with classes:", to_include)

        labels = np.array([sample[1] for sample in self._samples])
        exclude = np.array(exclude_list).reshape(1, -1)
        mask = ~(labels.reshape(-1, 1) == exclude).any(axis=1)

        self._samples = [s for idx, s in enumerate(self._samples) if mask[idx] == 1]
