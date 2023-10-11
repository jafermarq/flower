
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

def load_data():
    """Load CIFAR-10 (training and test set)."""
    trf = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    trainset = MNIST("./data", train=True, download=True, transform=trf)
    testset = MNIST("./data", train=False, download=True, transform=trf)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=32, num_workers=4)

    return trainloader, testloader
