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
"""Flower virtual client manager (abstract base class)."""


from abc import ABC, abstractmethod

from flwr.common import (
    GetPoolSizeRes,
    WakeUpClientsIns,
    ReadyForSamplingRes
)


class VirtualClientManager(ABC):
    """Abstract base class for Flower virtual client managers."""

    def __init__(self):
        self.pending_jobs = False

    @abstractmethod
    def get_pool_ids(self) -> GetPoolSizeRes:
        """Returns the list of client ids to be used for training/val/test."""

    @abstractmethod
    def wakeup_clients(self, ins: WakeUpClientsIns) -> None:
        """Tells which clients in the virtual pool to instantiate."""

    @abstractmethod
    def is_ready_for_sampling(self) -> ReadyForSamplingRes:
        """Tells RemoteClientManager that some clients are ready to be sampled."""

    @abstractmethod
    def set_config(self, serialised_config: str) -> None:
        """Used by the RemoteClientManager to send the config containing client/vcm side parameters.
        This config is a serialised dict obtained when launching the server and reading a toml file."""
