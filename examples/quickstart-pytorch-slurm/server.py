from typing import List, Tuple
from collections import OrderedDict

import flwr as fl
from flwr.common import Metrics

import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from models import test
from dataset import load_data

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

def get_on_fit_config(server_config: DictConfig):
    # The function below will be called by the strategy before commencing
    # a new fit round. The dictionary it returns is the `config` the
    # client's 'fit()' method receives
    def fit_config_fn(server_round: int):
        # convert config to standard Python dict.
        fit_config = OmegaConf.to_container(
            server_config.fit_config, resolve=True
        )
        fit_config["curr_round"] = server_round  # add round info
        return fit_config

    return fit_config_fn


def get_evaluate_fn(testloader, device, model: DictConfig):
    """Return a function that will be executed by the strategy
    after aggregating models sent by the clients"""
    def evaluate(
        server_round: int, parameters_ndarrays, config):
        """Use the entire MNIST test set for evaluation."""
        net = instantiate(model)
        params_dict = zip(net.state_dict().keys(), parameters_ndarrays)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        net.to(device)

        loss, accuracy = test(net, testloader, device=device)

        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate


@hydra.main(config_path="conf", config_name="base_server", version_base=None)
def main(cfg: DictConfig) -> None:

    # Prepare testset for centralised evaluation
    _, testloader = load_data()

    # Instantiate the strategy
    strategy = instantiate(cfg.strategy,
                           evaluate_metrics_aggregation_fn=weighted_average,
                           on_fit_config_fn=get_on_fit_config(cfg),
                           evaluate_fn=get_evaluate_fn(testloader,
                                                       cfg.device,
                                                       cfg.model))

    # Start Flower server
    fl.server.start_server(
        server_address=f"{cfg.address}:{cfg.port}",
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()