

# Federating Large Models without gRPC limitations

TL;DR; Don't communicate models over `gRPC`, instead communicate a reference to such model that lives in a DB. Then clients pull from the DB, do training, push their updated models to the DB and the server pulls those models for aggreagtion. This example implements such DB as a [MinIO](https://min.io/) service running in a Docker container. 

> This example has hardoced credentials, so run this for testing purposes only

### Build your Python environment

```bash
conda install -n minio-tests python=3.10 -y

pip install -r requirements.txt
```

You'll need to have Docker installed.


### Launch MinIO DB

Launch a container with a very vanilla MinIO setup. Here we'll just use the command provided in the [MinIO](https://min.io/docs/minio/container/index.html) docker docs.
```bash
mkdir -p minio/data # this is only needed the first time you run the example

./run_minio.sh
```

### Basics

The `test_push_pull_minio.py` shows in a single function how to push/pull to/from `MinIO` without FL. What gets pushed too the DB is a serialised version of the `state_dict`. You can test it like this:

```bash
python test_push_pull_minio.py --toy # will communicate the state_dict of a small nn.Conv2d layer

python test_push_pull_minio.py # same test but with a 1B LLLM
```

### Now with Flower

Run a 3-round federation with a 1B LLM model (or a smaller ResNet18 if you pass `--resnet` when executing the commands below).

```bash
python server.py
```

Now start at least two clients:
```bash
python client.py
```

### Limitations

Models are not deleted from the DB, so if you run it for long or with very large models you'll run out of space. If you want to erase the contents of the DB the best is to: (1) terminalte the MinIO container, (2) erase the contents of `./minio/data` (or the path you mounted on when executing the `docker run`)