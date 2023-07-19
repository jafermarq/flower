FROM amd64/ubuntu:22.04

ARG USERNAME=flwr-docker
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ENV DEBIAN_FRONTEND=noninteractive

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

# Install system dependencies
RUN apt update 
RUN apt install -y curl wget gnupg python3 python-is-python3 python3-pip git \
    build-essential tmux vim nano htop llvm \
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

RUN python -m pip install \
    pip==23.1.2 \
    setuptools==68.0.0

USER $USERNAME
ENV PATH="/home/$USERNAME/.local/bin:${PATH}"
ENV HOME="/home/$USERNAME"
WORKDIR $HOME

# Set Pyenv
RUN git clone --depth 1 --branch v2.3.22 https://github.com/pyenv/pyenv.git ~/.pyenv
ENV PYENV_ROOT="$HOME/.pyenv"
ENV PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"

# Install a bunch of Python versions
RUN pyenv install 3.7.17
RUN pyenv install 3.8.17
RUN pyenv install 3.10.12
ARG PYTHON_GLOBAL=3.9.17
RUN pyenv install $PYTHON_GLOBAL

# now add to .bahsrc
RUN echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
RUN echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
RUN echo 'eval "$(pyenv init -)"' >> ~/.bashrc

# let's set global to a reasonalbe version
RUN pyenv global $PYTHON_GLOBAL

# Setup poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
RUN echo 'export PATH="/home/$USERNAME/.local/bin:$PATH"' >> ~/.bashrc

# clone flower repo
ARG BRANCH=main
RUN git clone -b $BRANCH https://github.com/adap/flower.git
WORKDIR $HOME/flower

# setup poetry for flower
RUN pyenv local $PYTHON_GLOBAL
RUN poetry env use $PYTHON_GLOBAL
RUN poetry install --all-extras

RUN poetry run pip install -e .

#TODO: make this configurable at build time ? 
RUN poetry run pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
