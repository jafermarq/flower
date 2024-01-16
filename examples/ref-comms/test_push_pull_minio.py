

# mostly borrowed from MinIO Python example: https://min.io/docs/minio/linux/developers/python/minio-py.html

import argparse
from collections import OrderedDict
from typing import Tuple
from io import BytesIO
import torch
from minio import Minio
from transformers import AutoModelForCausalLM
from tqdm import tqdm


parser = argparse.ArgumentParser(description="Yes")

parser.add_argument(
    "--toy",
    action="store_true",
    help="Pushes/Pull the state_dict of a single nn.Conv2D layer (else a 1B params model)",
)

StateDict = OrderedDict[str, torch.Tensor]

def torch_state_dict_to_bytes(state_dict: StateDict) -> Tuple[BytesIO, int]:
    buffer = BytesIO()
    torch.save(state_dict, buffer)
    return buffer, len(buffer.getbuffer())

def bytes_to_torch_state_dict(some_bytes: bytes) -> StateDict:
    return torch.load(BytesIO(some_bytes))


def main():

    args = parser.parse_args()

    # Create client
    client = Minio('localhost:9000',
        access_key="ROOTUSER",
        secret_key="CHANGEME123",
        secure=False,
    )


    # The destination bucket and filename on the MinIO server
    bucket_name = "global-models"
    server_round = 4
    destination_file = f"global-model-round-{server_round}-state-dict.pt"

    # Let's send the state_dict of this layer
    if args.toy:
        model = torch.nn.Conv2d(3, 64, 5)
    else:
        model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    buffer, buffer_length = torch_state_dict_to_bytes(model.state_dict())
    print(f"{buffer_length = } bytes")

    # Make the bucket if it doesn't exist.
    found = client.bucket_exists(bucket_name)
    if not found:
        client.make_bucket(bucket_name)

    # Push object
    result = client.put_object(
        bucket_name, destination_file, BytesIO(buffer.getvalue()), length=buffer_length,
    )
    print(result.object_name, result.etag)
    print('Pushed!!')

    # Let's now pull the state_dict and apply it to this new layer
    chunk_size = 1024 * 1024 # 1M chunk

    result = bytearray()
    for offset in tqdm(range(0, buffer_length, chunk_size), desc="Downloading from MinIO"):
        try:
            response = client.get_object(bucket_name, destination_file, length=chunk_size, offset=offset)
            result.extend(response.read())
        finally:
            response.close()
            response.release_conn()
    print('downloaded!')

    downloaded_state_dict = bytes_to_torch_state_dict(result)
    print("Converted back to state_dict !")

    if args.toy:
        model_ = torch.nn.Conv2d(3, 64, 5)
    else:
        model_ = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # simple setting to zeros    
    for name, p in model_.named_parameters():
        print(f"resetting: {name}")
        p.data *= 0

    print("Loading state_dict ...")
    model_.load_state_dict(downloaded_state_dict)

    print("Checking consistency...")
    # Ensure they are identical
    for p1, p2 in  zip(model.parameters(), model_.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            assert False, "state dicts don't match :("

    print('All seems good :)')


if __name__ == "__main__":
    main()
