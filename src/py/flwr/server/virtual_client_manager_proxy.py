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
"""Flower client (abstract base class)."""


from abc import ABC, abstractmethod

from flwr.common import (
    GetPoolSizeRes,
    WakeUpClientsIns,
    IsAvailableRes,
    Disconnect
)


class VirtualClientManagerProxy(ABC):
    """Abstract base class for Flower virtual client manager proxies."""

    def __init__(self, pool_size: int):
        self.pool_size = pool_size
        self.pending_jobs = False

    @abstractmethod
    def get_pool_size(self) -> GetPoolSizeRes:
        """Return the size of virtual pool of clients."""

    @abstractmethod
    def wakeup_clients(self, ins: WakeUpClientsIns) -> None:
        """Tells which clients in the virtual pool to instantiate."""

    @abstractmethod
    def is_available(self) -> IsAvailableRes:
        """Tells whether the VCM still has submitted jobs that haven't finished running"""

    @abstractmethod
    def disconnect(self) -> Disconnect:
        """Shutdown signal from RemoteClientManager."""
