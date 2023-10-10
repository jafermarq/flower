from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

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

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:

    server_config = cfg.server

    # Instantiate the strategy
    strategy = instantiate(server_config.strategy,
                           evaluate_metrics_aggregation_fn=weighted_average,
                           on_fit_config_fn=get_on_fit_config(server_config))

    # Start Flower server
    fl.server.start_server(
        server_address=f"{server_config.address}:{server_config.port}",
        config=fl.server.ServerConfig(num_rounds=server_config.num_rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()