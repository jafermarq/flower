import torch
from datasets import Dataset
from torch.utils.data import DataLoader
import flwr as fl

from dataset import apply_eval_transforms
from model import get_model, set_parameters, test


def fit_config(server_round: int):
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "lr": 0.001,  # Learning rate used by clients
        "batch_size": 64,  # Batch size to use by clients during fit()
    }
    return config


def get_evaluate_fn(
    centralized_testset: Dataset,
):
    """Return an evaluation function for centralized evaluation."""

    def evaluate(server_round, parameters, config):
        """Use the entire CIFAR-100 test set for evaluation."""

        # Determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = get_model()
        set_parameters(model, parameters)
        model.to(device)

        # Apply transform to dataset
        testset = centralized_testset.with_transform(apply_eval_transforms)

        testloader = DataLoader(testset, batch_size=128)
        # Run evaluation
        loss, accuracy = test(model, testloader, device=device)

        return loss, {"accuracy": accuracy}

    return evaluate


def get_strategy(dataset):
    # Configure the strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.5,  # Sample 50% of available clients for training each round
        fraction_evaluate=0.0,  # No federated evaluation
        on_fit_config_fn=fit_config,
        evaluate_fn=get_evaluate_fn(dataset),  # Global evaluation function
    )

    return strategy
