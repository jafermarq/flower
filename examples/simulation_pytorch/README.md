# example_simulation_ray

This code splits CIFAR-10 dataset into `pool_size` partitions (user defined) and does a few rounds of CIFAR-10 training. In this example, we leverage [`Ray`](https://docs.ray.io/en/latest/index.html) to simulate Flower Clients participating in FL rounds in an resource-aware fashion. This is possible via the [`RayClientProxy`](https://github.com/adap/flower/blob/main/src/py/flwr/simulation/ray_transport/ray_client_proxy.py) which bridges a standard Flower server with standard Flower clients while excluding the gRPC communication protocol and the Client Manager in favour of Ray's scheduling and communication layers.

## Requirements

*    Flower 0.18.0
*    A recent version of PyTorch. This example has been tested with Pytorch 1.7.1, 1.8.2 (LTS) and 1.10.2.
*    A recent version of Ray. This example has been tested with Ray 1.4.1, 1.6 and 1.9.2.

From a clean virtualenv or Conda environment with Python 3.7+, the following command will isntall all the dependencies needed:
```bash
$ pip install -r requirements.txt
```

# How to run

This example:

1. Downloads CIFAR-10
2. Partitions the dataset into N splits, where N is the total number of
   clients. We refere to this as `pool_size`. The partition can be IID or non-IID
4. Starts a Ray-based simulation where a % of clients are sample each round.
   This example uses N=10, so 10 clients will be sampled each round.
5. After the M rounds end, the global model is evaluated on the entire testset.
   Also, the global model is evaluated on the valset partition residing in each
   client. This is useful to get a sense on how well the global model can generalise
   to each client's data.

The command below will assign each client 2 CPU threads. If your system does not have 2xN(=10) = 20 threads to run all 10 clients in parallel, they will be queued but eventually run. The server will wait until all N clients have completed their local training stage before aggregating the results. After that, a new round will begin.

```bash
$ python main.py --num_client_cpus 2 # note that `num_client_cpus` should be <= the number of threads in your system.
```

# Test resuming functionality

When you run the code as above (e.g.`python main.py`) on each call to the server's evaluate method, a pickle is saved containing the state of the global model, as well as other metadata. If you run the code passing flag `--resume` followed by the path to the pickle you want to resume from, the code will: (1) initialise the global model with that in the pickle; (2) set the server's starting round to that found in the pickle; and, (3) pass one of the variables in the checkpoint as part of the instructions to each client (i.e. via `configure_fit()`) using a custmo strategy.

There are other ways to achieve this same functionality without needing a custom strategy. For example, the whole metadata (except the model state) in the checkpoint could be passed to `fit_config()` in `main.py` (in this way a custom strategy won't be needed -- although it will certainly give more flexibility for more complicated settings). Likewise, a custom server is not strictly needed. A (custom) strategy could be implemented to keep track of the (real) round count and ignore the round index shown (and logged) by the server.
