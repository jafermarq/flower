import argparse
from typing import Tuple

import numpy as np
import flwr as fl
from transformers import AutoModelForCausalLM
from torchvision.models import resnet18
from minio import Minio
from minio_utils import pull_from_minio_chunked, bytes_to_torch_state_dict, torch_state_dict_to_bytes, push_to_minio
from secrets import token_hex

# Get node id
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--resnet",
    action='store_true',
)

# Define Flower client
class FlowerClient(fl.client.NumPyClient):

    def __init__(self,  minio_access_key, minio_secret_key, use_resnet: bool):
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

    def get_parameters(self, config):
        # Not used
        return [np.random.randn(1)]
    
    def push_updated_model_to_minio(self, bucket_name: str, server_round:int) -> Tuple[str, int]:

        buffer, buffer_length = torch_state_dict_to_bytes(self.model.state_dict())
        file_name = f"round_{server_round}/{token_hex(8)}.pt"
        print(f"Client pushing: {file_name}")
        push_to_minio(self.client, bucket_name, file_name , buffer, buffer_length)
        print('Done!')
        return file_name, buffer_length

    def fit(self, parameters, config):

        bucket_name = config['bucket_name']
        file = config['file']
        buffer_length = config['buffer_length']
        server_round = config['server_round']
        bytes = pull_from_minio_chunked(self.client, bucket_name, file_name=file, buffer_length=buffer_length)

        state_dict = bytes_to_torch_state_dict(bytes)

        self.model.load_state_dict(state_dict)

        # Now do some training


        # Now push updated model to MinIO
        pushed_file, buffer_length = self.push_updated_model_to_minio(bucket_name, server_round)

        return self.get_parameters(config={}), 1, {'model_file': pushed_file, 'buffer_length': buffer_length}




if __name__ == "__main__":
    args = parser.parse_args()

    # Start Flower client
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(minio_access_key="ROOTUSER", minio_secret_key="CHANGEME123", use_resnet=args.resnet),
    )
    