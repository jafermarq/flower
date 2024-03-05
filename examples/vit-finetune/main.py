import argparse

import flwr as fl
import matplotlib.pyplot as plt

from server import get_strategy
from client import get_client_fn
from dataset import get_dataset_with_partitions

parser = argparse.ArgumentParser(
    description="Finetuning of a ViT with Flower Simulation."
)

parser.add_argument(
    "--num-clients",
    type=int,
    default=20,
    help="Number total clients in the experiment.",
)
parser.add_argument(
    "--num-rounds",
    type=int,
    default=10,
    help="Number of rounds.",
)


def main():

    args = parser.parse_args()

    # Download and partition dataset
    federated_c100, centralised_testset = get_dataset_with_partitions(args.num_clients)

    # Construct `get_client` function
    client_fn = get_client_fn(federated_c100)

    # Construct strategy
    strategy = get_strategy(centralised_testset)

    # To control the degree of parallelism
    # With default settings in this example,
    # each client should take just ~2GB of VRAM.
    client_resources = {
        "num_cpus": 4,
        "num_gpus": 0.25,
    }

    # Launch simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.num_clients,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )

    print(history)

    # Basic plotting
    global_accuracy_centralised = history.metrics_centralized["accuracy"]
    round = [data[0] for data in global_accuracy_centralised]
    acc = [100.0 * data[1] for data in global_accuracy_centralised]
    plt.plot(round, acc)
    plt.grid()
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Round")
    plt.title("Federated finetuning of ViT for CIFAR-100")
    plt.savefig("central_evaluation.png")


if __name__ == "__main__":
    main()
