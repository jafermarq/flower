import warnings
from time import sleep
from collections import OrderedDict
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import flwr as fl
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor


warnings.filterwarnings("ignore", category=UserWarning)


def train(net, trainloader, epochs, device, train_settings):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=train_settings["lr"], momentum=train_settings["momentum"])
    for _ in range(epochs):
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            criterion(net(images.to(device)), labels.to(device)).backward()
            optimizer.step()


def test(net, testloader, device):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            outputs = net(images.to(device))
            labels = labels.to(device)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


def load_data():
    """Load CIFAR-10 (training and test set)."""
    trf = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    trainset = MNIST("./data", train=True, download=True, transform=trf)
    testset = MNIST("./data", train=False, download=True, transform=trf)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=32, num_workers=4)

    return trainloader, testloader


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net_config, trainloader, testloader, local_epochs, device):
        self.net = instantiate(net_config)
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = local_epochs
        self.device = device

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.net, self.trainloader, epochs=self.local_epochs,
              device=self.device, train_settings=config)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader, device=self.device)
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:

    # Print config structured as YAML
    print(OmegaConf.to_yaml(cfg))

    # Load the dataset
    trainloader, testloader = load_data()

    # Let's delay the start of the client for a few seconds
    # to ensure the server is up and ready to accept connections
    # from clients
    sleep(cfg.client.wait_for_server)


    # Instantiate client object, note that here I only need to pass the arguemtns
    # that were not specified in the config. Those that were, can still be overriden.
    client = instantiate(cfg.client.client,
                         trainloader=trainloader,
                         testloader=testloader)

    # Start Flower client
    fl.client.start_numpy_client(
        server_address=f"{cfg.server.address}:{cfg.server.port}",
        client=client,
    )

if __name__ == "__main__":
    main()