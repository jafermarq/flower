

from collections import OrderedDict
from io import BytesIO
from typing import Tuple
import torch
from minio import Minio
from tqdm import tqdm

StateDict = OrderedDict[str, torch.Tensor]


def torch_state_dict_to_bytes(state_dict: StateDict) -> Tuple[BytesIO, int]:
    buffer = BytesIO()
    torch.save(state_dict, buffer)
    return buffer, len(buffer.getbuffer())

def bytes_to_torch_state_dict(some_bytes: bytes) -> StateDict:
    return torch.load(BytesIO(some_bytes))


def push_to_minio(client: Minio, bucket_name: str, destination_file: str, buffer: bytes, buffer_length: int):

    client.put_object(
        bucket_name, destination_file, BytesIO(buffer.getvalue()), length=buffer_length,
    )

def pull_from_minio_chunked(client: Minio, bucket_name: str, file_name: str, buffer_length: int, chunk_size: int = 1024**2):

    result = bytearray()
    for offset in tqdm(range(0, buffer_length, chunk_size), desc="Downloading from MinIO"):
        try:
            response = client.get_object(bucket_name, file_name, length=chunk_size, offset=offset)
            result.extend(response.read())
        finally:
            response.close()
            response.release_conn()
        
    return result