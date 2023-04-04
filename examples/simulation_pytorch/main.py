import argparse
import pickle
import flwr as fl
from flwr.common.typing import Scalar
import ray
import torch
import torchvision
import numpy as np
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Callable, Optional, Tuple, List


from flwr.server.client_manager import SimpleClientManager

from dataset_utils import get_cifar_10, do_fl_partitioning, get_dataloader
from utils import Net, train, test
from with_resume import FlowerServerResumable, CustomFedAvg


parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")

parser.add_argument("--num_client_cpus", type=int, default=1)
parser.add_argument("--num_rounds", type=int, default=10)
parser.add_argument("--resume", type=str, default="", help='path to pickle containing checkpoint')


# Flower client, adapted from Pytorch quickstart example
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid: str, fed_dir_data: str):
        self.cid = cid
        self.fed_dir = Path(fed_dir_data)
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}

        # Instantiate model
        self.net = Net()

        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_parameters(self, config):
        return get_params(self.net)

    def fit(self, parameters, config):
        set_params(self.net, parameters)

        # Load data for this client and get trainloader
        num_workers = 2 #int(ray.get_runtime_context().get_assigned_resources()["CPU"])
        trainloader = get_dataloader(
            self.fed_dir,
            self.cid,
            is_train=True,
            batch_size=config["batch_size"],
            workers=num_workers,
        )

        # Send model to device
        self.net.to(self.device)

        # Train
        train(self.net, trainloader, epochs=config["epochs"], device=self.device)

        # Return local model and statistics
        return get_params(self.net), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_params(self.net, parameters)

        # Load data for this client and get trainloader
        num_workers = 2 #int(ray.get_runtime_context().get_assigned_resources()["CPU"])
        valloader = get_dataloader(
            self.fed_dir, self.cid, is_train=False, batch_size=50, workers=num_workers
        )

        # Send model to device
        self.net.to(self.device)

        # Evaluate
        loss, accuracy = test(self.net, valloader, device=self.device)

        # Return statistics
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}


def fit_config(server_round: int) -> Dict[str, Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epochs": 5,  # number of local epochs
        "batch_size": 64,
    }
    return config


def get_params(model: torch.nn.ModuleList) -> List[np.ndarray]:
    """Get model weights as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_params(model: torch.nn.ModuleList, params: List[np.ndarray]):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def get_evaluate_fn(
    testset: torchvision.datasets.CIFAR10,
) -> Callable[[fl.common.NDArrays], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(
        server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, float]]:
        """Use the entire CIFAR-10 test set for evaluation."""

        # determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = Net()
        set_params(model, parameters)
        model.to(device)

        testloader = torch.utils.data.DataLoader(testset, batch_size=50)
        loss, accuracy = test(model, testloader, device=device)

        # here i'm simulating saving a checkpoint cointaining some useful data alongside the model
        checkpoint = {'state_dict': model.cpu().state_dict(), 'sumo_state': torch.randn(8,1),
                      'last_loss': loss, 'round_number': server_round}

        with open(f'state_round_{server_round}.pkl', 'wb') as f:  # open a text file
            pickle.dump(checkpoint, f) # serialize the list

        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate


def load_checkpoint(path_to_pickle):
    from flwr.common import ndarrays_to_parameters
    with open(path_to_pickle, 'rb') as f:
        data = pickle.load(f)
    print(data.keys())
    # load model state
    state_dict = data.pop('state_dict')
    init_params = ndarrays_to_parameters([value for value in state_dict.values()])

    return init_params, data

# Start simulation (a _default server_ will be created)
# This example does:
# 1. Downloads CIFAR-10
# 2. Partitions the dataset into N splits, where N is the total number of
#    clients. We refere to this as `pool_size`. The partition can be IID or non-IID
# 3. Starts a simulation where a % of clients are sample each round.
# 4. After the M rounds end, the global model is evaluated on the entire testset.
#    Also, the global model is evaluated on the valset partition residing in each
#    client. This is useful to get a sense on how well the global model can generalise
#    to each client's data.
if __name__ == "__main__":
    # parse input arguments
    args = parser.parse_args()

    pool_size = 100  # number of dataset partions (= number of total clients)
    client_resources = {
        "num_cpus": args.num_client_cpus
    }  # each client will get allocated 1 CPUs

    # Download CIFAR-10 dataset
    train_path, testset = get_cifar_10()

    # partition dataset (use a large `alpha` to make it IID;
    # a small value (e.g. 1) will make it non-IID)
    # This will create a new directory called "federated": in the directory where
    # CIFAR-10 lives. Inside it, there will be N=pool_size sub-directories each with
    # its own train/set split.
    fed_dir = do_fl_partitioning(
        train_path, pool_size=pool_size, alpha=1000, num_classes=10, val_ratio=0.1
    )


    initial_parameters = None
    resume_checkpoint = None
    if args.resume:
        initial_parameters, resume_checkpoint = load_checkpoint(args.resume)

    # configure the strategy
    strategy = CustomFedAvg(
        fraction_fit=0.1,
        fraction_evaluate=0.1,
        min_fit_clients=10,
        min_evaluate_clients=10,
        min_available_clients=pool_size,  # All clients should be available
        on_fit_config_fn=fit_config,
        evaluate_fn=get_evaluate_fn(testset),  # centralised evaluation of global model
        initial_parameters=initial_parameters, # pass inital parameters for strategy
        resume_checkpoint=resume_checkpoint # pass checkpoint with state
    )

    def client_fn(cid: str):
        # create a single client instance
        return FlowerClient(cid, fed_dir)

    # (optional) specify Ray config
    ray_init_args = {"include_dashboard": False}


    # start round (if no checkpoint is passed, first round is #1)
    start_round = 1 if resume_checkpoint is None else resume_checkpoint.get('round_number',1)
    # start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=pool_size,
        server=FlowerServerResumable(start_round=start_round, client_manager=SimpleClientManager(), strategy=strategy), # start from specific round
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds + start_round-1),
        strategy=strategy,
        ray_init_args=ray_init_args,
    )
