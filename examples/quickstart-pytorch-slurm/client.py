import logging
from time import sleep
from collections import OrderedDict
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import flwr as fl
import torch

from dataset import load_data
from models import train, test

log = logging.getLogger(__name__)

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net_config, trainloader, testloader, local_epochs, device):
        self.net = instantiate(net_config)
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = local_epochs
        self.device = device
        log.info("Client creation completled")

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        log.info("Fit beings")
        log.info(f"Config received: {config}")
        self.set_parameters(parameters)
        train(self.net, self.trainloader, epochs=self.local_epochs,
              device=self.device, train_settings=config)
        log.info("Fit ends")
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader, device=self.device)
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}


@hydra.main(config_path="conf", config_name="base_client", version_base=None)
def main(cfg: DictConfig) -> None:

    # Print config structured as YAML
    print(OmegaConf.to_yaml(cfg))

    # Load the dataset
    trainloader, testloader = load_data()

    # Let's delay the start of the client for a few seconds
    # to ensure the server is up and ready to accept connections
    # from clients
    sleep(cfg.wait_for_server)


    # Instantiate client object, note that here I only need to pass the arguemtns
    # that were not specified in the config. Those that were, can still be overriden.
    client = instantiate(cfg.client,
                         trainloader=trainloader,
                         testloader=testloader)

    # Start Flower client
    fl.client.start_numpy_client(
        server_address=f"{cfg.server.address}:{cfg.server.port}",
        client=client,
    )

if __name__ == "__main__":
    main()