

This directory contains two docker images suitable for `x86` and `arm64` platforms respectively. The images setup `pyenv`, `poetry` and clone the `flwr` repo. Build the image, run the container, and do your FL magic.

## How to use
Build the image suitable for your platform by executing `./build.sh`. Then run the container:

```bash
    # will run the container in interactive mode and remove it once you exit it
    docker run --rm -it flwr

    # once in, do the following to activate your environment
    poetry shell
```
