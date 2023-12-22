# Flower (w/ DriverAPI) Example using PyTorch

> This example contains experimental features of Flower.

Running FL with Flower typically involves designing and running two components: the server side, where the strategy lives; and the clients, where the training takes place. Normally, you'd first launch the server, then the clients. After a pre-defined number of rounds, the FL job is completed and both processes exit.

The Driver API (i.e. the new way of using Flower) introduces several improvements:

* **Long-running server**, i.e. a single flower server entity can be reused across multiple FL jobs/experiments
* **Decoupling the strategy from the server**. Now the strategy will run in a separate process. We'll refer to this as the "driver".
* **Long-running client**, enabling clients to be decoupled from the _application_ (i.e. the driver) and therefore able to participate in multiple FL jobs/experiments.


### Launching long-running components

> This example does not enable SSL. Should you want to do so, please follow this guide: https://flower.dev/docs/framework/how-to-enable-ssl-connections.html

First, let's launch the server and clients. Note this won't start any training since this is defined in a separate script (i.e. `driver.py`) that will be launched aftwerwards. 

Let's start by launching the server. The `flower-server` needs to bind to two addresses:
*   `Fleet API` address to communicate with the clients (clients will need to connect to this address)
*   `Driver API` address to communicate with the driver (driver will need to connect to this address)

By default these addresses are `0.0.0.0:9091` and `0.0.0.0:9092`, but you can override them by passing command line arguments `--grpc-rere-fleet-api-address` and `--driver-api-address`

```bash
flower-server --insecure --grpc-rere-fleet-api-address=<FLEET_ADDRESS> --driver-api-address=<DRIVER_ADDRESS>
# Something like the below will be printed
INFO flwr 2023-12-22 11:27:56,915 | app.py:362 | Starting Flower server
WARNING flwr 2023-12-22 11:27:56,916 | app.py:465 | Option `--insecure` was set. Starting insecure HTTP server.
INFO flwr 2023-12-22 11:27:56,923 | app.py:547 | Flower ECE: Starting Driver API (gRPC-rere) on <DRIVER_ADDRESS>
DEBUG flwr 2023-12-22 11:27:56,925 | state_factory.py:41 | Using InMemoryState
INFO flwr 2023-12-22 11:27:56,926 | app.py:600 | Flower ECE: Starting Fleet API (gRPC-rere) on <FLEET_ADDRESS>
```

Next, let's launch the clients. To launch a client we need to pass a `--callable` that should point to a `<file>:<flower-callable>`. For example, below it points to the `hospital_a` object in `client.py`. Then, we need to specify the address to which clients connect to the server (this is the `FLEET_ADDRESS` we specified when launching the server earlier)

```bash
flower-client --callable client:hospital_a --insecure --server=<FLEET_ADDRESS>
```

After a client connects to the FLower server, it will wait idle until it is told to do some work (this will only take place once we launch the driver script)

We can launch other clients, not necessarily on the same machine. To showcase this, `client.py` includes another callable, `hospital_b`.

```bash
flower-client --callable client:hospital_b --insecure --server=<FLEET_ADDRESS>
```

Each of these clients, although they share some components, they are different since each loads a different `Hydra` config. Note it is perfectly possible to define each client in completely separate Python scripts too without making use of config files.


### Starting FL

With server and clients connected, and waiting for instructions, let's launch the driver script to start an FL job. The `driver.py` should look quite familiar if you have worked with Flower before but never used the Driver API.

Launch the driver:
```bash
# recall the driver.py is using Hydra configs
python driver.py address=<DRIVER_ADDRESS>

# will print the usual log
INFO flwr 2023-12-22 12:01:30,445 | grpc.py:52 | Opened insecure gRPC connection (no certificates were passed)
[2023-12-22 12:01:30,445][flwr][INFO] - Opened insecure gRPC connection (no certificates were passed)
INFO flwr 2023-12-22 12:01:30,446 | grpc_driver.py:73 | [Driver] Connected to <DRIVER_ADDRESS>
INFO flwr 2023-12-22 12:01:30,446 | app.py:125 | Starting Flower server, config: ServerConfig(num_rounds=3, round_timeout=None)
INFO flwr 2023-12-22 12:01:30,446 | server.py:89 | Initializing global parameters
INFO flwr 2023-12-22 12:01:30,449 | server.py:276 | Requesting initial parameters from one random client
INFO flwr 2023-12-22 12:01:33,472 | server.py:280 | Received initial parameters from one random client
INFO flwr 2023-12-22 12:01:33,473 | server.py:91 | Evaluating initial parameters
100%|██████████████████████████████████████████████████| 313/313 [00:01<00:00, 223.54it/s]
[2023-12-22 12:01:34,899][strategy][INFO] - Saved checkpoint in: <....>
INFO flwr 2023-12-22 12:01:34,899 | server.py:94 | initial parameters (loss, other metrics): 722.522, {'accuracy': 0.0931}
INFO flwr 2023-12-22 12:01:34,899 | server.py:104 | FL starting
[2023-12-22 12:01:36,914][strategy][INFO] - Client LOCATION-B excluded from fit round.
[2023-12-22 12:01:38,930][strategy][INFO] - Client LOCATION-A enrolled in fit round.
DEBUG flwr 2023-12-22 12:01:38,931 | server.py:222 | fit_round 1: strategy sampled 1 clients (out of 2)
DEBUG flwr 2023-12-22 12:01:55,015 | server.py:236 | fit_round 1 received 1 results and 0 failures
...
```

You'll notice that: (1) the server starts logging some messages showing that instructions are being received from the driver and are sent to the clients; (2) the clients will start doing some work. Once the driver script finishes (because it runs for N rounds), the server and clients will remain connected. You could launch the same driver script or a different one. You can launch two in parallel as well.

The logs and checkpoints that the driver generates are stored in `outputs/driver/<date>/<time>`.