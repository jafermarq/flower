# Federated finetuning of a ViT

This example shows how to use Flower's Simulation Engine to federate the finetuning of a [ViT-Base](https://pytorch.org/vision/main/models/generated/torchvision.models.vit_b_16.html#torchvision.models.vit_b_16) that has been pretrained on ImageNet. To keep things simple we'll be finetuning it to CIFAR-100, creating 20 partitions using [Flower-datasets](https://flower.ai/docs/datasets/). We'll be finetuning just the exit `head` of the ViT, this means that the training is not that costly and each client requires just ~1.5GB of VRAM (for a batchs ize of 64 images).

### Environment setup

Create a new Python environment. The steps below show how to do it with Conda.

```bash

# Create env
conda create -n flower-vit-finetune python=3.10 -y

# Activate it
conda activate flower-vit-finetune

# Install PyTorch + TorchVision
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Then, install Flower and FlowerDatasets
pip install -r requirements.txt
```

### Run with `start_simulation()`

Running the example is quite straightforward. You can control the number of rounds `--num-rounds` (which defaults to 10).

```bash
python main.py
```

![](_static/central_evaluation.png)


Take a look at the [Documentation](https://flower.ai/docs/framework/how-to-run-simulations.html) for more details on how you can customise your simulation.

Running the example as-is on an RTX 3090Ti should take ~1min/round running 4 clients in parallel (plus the _global model_ during centralized evaluation stages) in a single GPU. Note that more clients could fit in VRAM, but since the GPU utilization is high (99%-100%) we are probably better off not doing that (at least in this case).

```bash

+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.161.07             Driver Version: 535.161.07   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 3090 Ti     Off | 00000000:0B:00.0 Off |                  Off |
| 59%   82C    P2             444W / 450W |   8127MiB / 24564MiB |    100%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A     30741      C   python                                     1966MiB |
|    0   N/A  N/A     31419      C   ray::DefaultActor.run                      1536MiB |
|    0   N/A  N/A     31420      C   ray::DefaultActor.run                      1536MiB |
|    0   N/A  N/A     31421      C   ray::DefaultActor.run                      1536MiB |
|    0   N/A  N/A     31422      C   ray::DefaultActor.run                      1536MiB |
+---------------------------------------------------------------------------------------+
```


### Run with Flower Next (preview)

```bash
flower-simulation --client-app=client:app --server-app=server:app --num-supernodes=20 \
    --backend-config='{"client_resources": {"num_cpus":4, "num_gpus":0.25}}'
```