# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Flower type definitions."""


from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

Weights = List[np.ndarray]


@dataclass
class Parameters:
    """Model parameters."""

    tensors: List[bytes]
    tensor_type: str


@dataclass
class ParametersRes:
    """Response when asked to return parameters."""

    parameters: Parameters


@dataclass
class FitIns:
    """Fit instructions for a client."""

    parameters: Parameters
    config: Dict[str, str]


@dataclass
class FitRes:
    """Fit response from a client."""

    parameters: Parameters
    # config: Dict[str, str]
    num_examples: int
    num_examples_ceil: int
    fit_duration: float


@dataclass
class EvaluateIns:
    """Evaluate instructions for a client."""

    parameters: Parameters
    config: Dict[str, str]


@dataclass
class EvaluateRes:
    """Evaluate response from a client."""

    num_examples: int
    loss: float
    accuracy: float


@dataclass
class Reconnect:
    """Reconnect message from server to client."""

    seconds: Optional[int]


@dataclass
class Disconnect:
    """Disconnect message from client to server."""

    reason: str


@dataclass
class GetPoolSizeRes:
    """GetPoolSize response from virtual client manager."""

    pool_size: int


@dataclass
class WakeUpClientsIns:
    """WakeUpClients message from RemoteClientManager to VirtualClientManager"""

    cids: str


@dataclass
class WakeUpClientsRes:
    """VirtualClientManager replying back to RemoteClientManager"""

    reason: str


@dataclass
class IsAvailableRes:
    """VirtualClientManager replying back to RemoteClientManager indicting
    if it is still running jobs from previous `wakeup_clients` request."""

    status: bool


@dataclass
class ReadyForSamplingRes:
    """Contains: boolean telling if the RemoteClientManager can sample from the
    connected clients and a integer that indicates the number of clients ready."""

    wait: bool
    num_clients: int
