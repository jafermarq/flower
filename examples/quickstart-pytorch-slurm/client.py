import logging
from time import sleep, time
from collections import OrderedDict
from typing import Dict
from flwr.common import Config, Scalar
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import flwr as fl
import torch

from dataset import load_data
from models import train, test

log = logging.getLogger(__name__)


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self, net_config, trainloader, testloader, local_epochs, device, location
    ):
        self.net = instantiate(net_config)
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = local_epochs
        self.device = device
        self.location_info = location
        log.info("Client creation completled.")
        log.info(self.location_info)

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        """Info that can be fetched by the server.

        Contains metadata from this client.
        """

        # only share what's needed (as plain dictionary)
        info = self.location_info
        return {
            "name": info.name,
            "fit": info.enrollment.for_fit,
            "eval": info.enrollment.for_eval,
        }

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        log.info("Fit beings")
        t_start = time()
        log.info(f"Config received: {config}")
        self.set_parameters(parameters)
        train(
            self.net,
            self.trainloader,
            epochs=self.local_epochs,
            device=self.device,
            train_settings=config,
        )
        t_end = time() - t_start
        log.info(f"Fit ends (too: {t_end:.2f} s)")
        return (
            self.get_parameters(config={}),
            len(self.trainloader.dataset),
            {"time_train": t_end},
        )

    def evaluate(self, parameters, config):
        log.info("Evaluate beings")
        t_start = time()
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader, device=self.device)
        t_end = time() - t_start
        log.info(f"Evaluate ends (too: {t_end:.2f} s)")
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
    client = instantiate(cfg.client, trainloader=trainloader, testloader=testloader)

    # Start Flower client
    fl.client.start_numpy_client(
        server_address=f"{cfg.server.address}:{cfg.server.port}",
        client=client,
    )


if __name__ == "__main__":
    main()
