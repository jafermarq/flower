import argparse
from typing import Dict, List, Tuple
from collections import OrderedDict
import random

from minio import Minio
import torch
import flwr as fl
from flwr.common import FitIns, FitRes, Parameters, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
from transformers import AutoModelForCausalLM
from torchvision.models import resnet18


from minio_utils import push_to_minio, torch_state_dict_to_bytes, pull_from_minio_chunked, bytes_to_torch_state_dict


parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--resnet",
    action='store_true',
)

class FedAvgWithParamRefs(FedAvg):

    def __init__(self, minio_access_key, minio_secret_key, use_resnet: bool, *args, **kwargs):
        self.minio_access_key = minio_access_key
        self.minio_secret_key = minio_secret_key
        if not use_resnet:
            self.model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        else:
            self.model = resnet18()
        self.client = Minio('localhost:9000',
                access_key=self.minio_access_key,
                secret_key=self.minio_secret_key,
                secure=False,
            )
        self.bucket_name = 'flower-test'
        found = self.client.bucket_exists(self.bucket_name)
        if not found:
            self.client.make_bucket(self.bucket_name)

        initial_parameters = ndarrays_to_parameters([val.cpu().numpy() for _, val in self.model.state_dict().items()])
        super().__init__(initial_parameters=initial_parameters, *args, **kwargs)
    
    def _generate_file_name(self, server_round: int) -> str:
        return f"global-model-round-{server_round}.pt"

    def push_state_dict_to_minio(self, parameters: Parameters, server_round: int) -> int:

        ndarrays = parameters_to_ndarrays(parameters)
        params_dict = zip(self.model.state_dict().keys(), ndarrays)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

        buffer, buffer_length = torch_state_dict_to_bytes(state_dict)

        print(f"Server pushing model to MinIO...")
        push_to_minio(self.client, self.bucket_name, self._generate_file_name(server_round), buffer, buffer_length)
        print(f"Done!")
        return buffer_length


    def configure_fit(self, server_round: int,
                      parameters: Parameters,
                      client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        """Just like FedAvg's `configure_fit()` but pushes model's state_dict to attached MinIO as opposed
        to send it as a list of NumPy arrays in FitIns."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
    
        # Don't add parameters to FitIns
        fit_ins = FitIns(Parameters(tensors=[], tensor_type="hmmm"), config)

        # Push model to MinIO
        # length of buffer is important to know so we can download and decode it later
        buffer_length = self.push_state_dict_to_minio(parameters, server_round)

        # Pass MinIO stuff through config (obviously this is not the best practice)
        config['bucket_name'] = self.bucket_name
        config['server_round'] = server_round
        config['file'] = self._generate_file_name(server_round)
        config['buffer_length'] = buffer_length

        # Rest is a usual in FedAvg

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(self, server_round: int,
                      results: List[Tuple[ClientProxy, FitRes]],
                      failures: List[Tuple[ClientProxy, FitRes] | BaseException]) -> Tuple[Parameters | None, Dict[str, Scalar]]:
        
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        
        # Here do aggregation. This example just takes one model pushed to MinIO by a client and 
        # treats it as the global model for the next round. We used the metrics to communicate
        # the filepath where models are

        # pull all models from MinIO
        metrics = [res.metrics for _, res in results]
        print(f"Received: {metrics}")
        
        idx = random.choice(range(len(metrics)))

        model_to_download = metrics[idx]['model_file']
        buffer_length = metrics[idx]['buffer_length']

        state_dict_as_bytes = pull_from_minio_chunked(self.client, self.bucket_name,
                                                      file_name=model_to_download,
                                                      buffer_length=buffer_length)
        
        # Now we assume this is the new "aggregated" model
        state_dict = bytes_to_torch_state_dict(state_dict_as_bytes)

        aggrageted_paramters = ndarrays_to_parameters([val.cpu().numpy() for _, val in state_dict.items()])

        return aggrageted_paramters, {}


if __name__ == "__main__":
    args = parser.parse_args()

    # Define strategy
    strategy = FedAvgWithParamRefs(minio_access_key="ROOTUSER", minio_secret_key="CHANGEME123", fraction_evaluate=0.0, use_resnet=args.resnet)

    # Start Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )
