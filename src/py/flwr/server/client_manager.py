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
"""Flower ClientManager."""


import random
import threading
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from .client_proxy import ClientProxy
from .virtual_client_manager_proxy import VirtualClientManagerProxy
from .criterion import Criterion
from flwr.common import WakeUpClientsIns


class ClientManager(ABC):
    """Abstract base class for managing Flower clients."""

    @abstractmethod
    def num_available(self) -> int:
        """Return the number of available clients."""

    @abstractmethod
    def register(self, client: ClientProxy) -> bool:
        """Register Flower ClientProxy instance.

        Returns:
            bool: Indicating if registration was successful
        """

    @abstractmethod
    def unregister(self, client: ClientProxy) -> None:
        """Unregister Flower ClientProxy instance."""

    @abstractmethod
    def all(self) -> Dict[str, ClientProxy]:
        """Return all available clients."""

    @abstractmethod
    def wait_for(self, num_clients: int, timeout: int) -> bool:
        """Wait until at least `num_clients` are available."""

    @abstractmethod
    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""


class SimpleClientManager(ClientManager):
    """Provides a pool of available clients."""

    def __init__(self) -> None:
        self.clients: Dict[str, ClientProxy] = {}
        self._cv = threading.Condition()

    def __len__(self) -> int:
        return len(self.clients)

    def wait_for(self, num_clients: int, timeout: int = 86400) -> bool:
        """Block until at least `num_clients` are available or until a timeout
        is reached.

        Current timeout default: 1 day.
        """
        with self._cv:
            return self._cv.wait_for(
                lambda: len(self.clients) >= num_clients, timeout=timeout
            )

    def num_available(self) -> int:
        """Return the number of available clients."""
        return len(self)

    def register(self, client: ClientProxy) -> bool:
        """Register Flower ClientProxy instance.

        Returns:
            bool: Indicating if registration was successful. False if ClientProxy is
                already registered or can not be registered for any reason
        """
        if client.cid in self.clients:
            return False

        self.clients[client.cid] = client
        with self._cv:
            self._cv.notify_all()

        return True

    def unregister(self, client: ClientProxy) -> None:
        """Unregister Flower ClientProxy instance.

        This method is idempotent.
        """
        if client.cid in self.clients:
            del self.clients[client.cid]

            with self._cv:
                self._cv.notify_all()

    def all(self) -> Dict[str, ClientProxy]:
        """Return all available clients."""
        return self.clients

    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
        virtual_pool: Optional[bool] = None,
    ) -> List[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""
        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = num_clients
        
        # print(f"min_num_clients: {min_num_clients}")
        self.wait_for(min_num_clients)
        # Sample clients which meet the criterion
        available_cids = list(self.clients)
        if criterion is not None:
            available_cids = [
                cid for cid in available_cids if criterion.select(self.clients[cid])
            ]
        if virtual_pool:
            num_clients = min(len(available_cids), num_clients)
        sampled_cids = random.sample(available_cids, num_clients)
        return [self.clients[cid] for cid in sampled_cids]


class RemoteClientManager(SimpleClientManager):

    def __init__(self) -> None:
        super(RemoteClientManager, self).__init__()
        self.vcm: VirtualClientManagerProxy = None
        self.pool_size: int
        # need to initialize with true for first client
        # woken up when server._get_initial_weights()
        self.wait_until_vcm_is_available = True

    def wait_for_vcm(self, timeout: int = 86400) -> bool:
        """Block until a VirtualClientManager has connected.

        Current timeout default: 1 day.
        """

        print("Waiting for VirtualClientManager to connect with server...")
        with self._cv:
            return self._cv.wait_for(
                lambda: self.vcm is not None, timeout=timeout
            )

    def register_vcm(self, vcm: VirtualClientManagerProxy) -> bool:
        """Register Flower VirtualClientManagerProxy instance.

        Returns:
            bool: Indicating if registration was successful.
        """

        self.vcm = vcm

        with self._cv:
            self._cv.notify_all()

        print("RemoteClientManager.register_vcm: Registered a Virtual Client Manager")
        return True

    def unregister_vcm(self) -> None:
        """Unregister Flower VirtualClientManager instance.

        This method is idempotent.
        """
        del self.vcm

        with self._cv:
            self._cv.notify_all()

    def get_virtual_pool_size(self) -> None:
        """ Gets the size of the virtual pool on the VCM's side."""
        self.pool_size = self.vcm.get_pool_size().pool_size
        print(f"Received --> pool_size = {self.pool_size} clients.")

    def wakeup_clients(self, cids: List[int]) -> None:
        """ Tells VCM which clients to use in this round/sub-round,
        len(cids) == number clients to use in round. It can perfectly happen
        that the machine where VCM is, can't allocate that many clients at a
        given time, if this happens it will use the first N (where N is the number
        of clients that the machine can allocate), then the next N, until
        all clients have done the specified amount of local epochs."""

        mssg = ""
        for n in cids[:-1]:
            mssg += str(n)
            mssg += ","
        mssg += str(cids[-1])
        mssg_ins = WakeUpClientsIns(cids=mssg)
        self.vcm.wakeup_clients(mssg_ins)

    def check_if_vcm_is_available(self) -> bool:
        """Check if there are Ray jobs running on the VirtualClientManager
        side."""
        available = False
        while not(available) and self.wait_until_vcm_is_available:
            available = self.vcm.is_available().status
            time.sleep(5)

        self.wait_until_vcm_is_available = False
        # print(f"OBTAINED: {status_res}")
        return available

    def shutdown_vcm(self) -> None:
        """Tells VCM to shutdown."""
        print("Telling VCM to shutdown...")
        _ = self.vcm.disconnect()

    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        """ Randomly choose client id from pool, tell VirtualClientManager
        to wake up those clients, wait until they connect, then continue... """

        # print(f"There are {len(self.clients)} clients connected")
        if min_num_clients is None:
            min_num_clients = num_clients

        if self.vcm is None:
            self.wait_for_vcm()
            self.get_virtual_pool_size()

        # Take `min_num_clients` from total number of clients in pool
        cids = random.sample(list(range(self.pool_size)), min_num_clients)

        # Wakeup clients in cid in cids
        if self.check_if_vcm_is_available():
            self.wakeup_clients(cids)

        # Block until clients managed by VCM are instantiated,
        # allocated, warmedup and connected to the server.

        wait = True
        clients_wait_for = 0
        while wait:
            # ask VCM whether clients are ready and how many are online
            res = self.vcm.is_ready_for_sampling()
            # print(f"client_manager got: {res}")
            wait = res.wait
            clients_wait_for = res.num_clients
            time.sleep(2)

        # ensure the number of clients that VCM said are online
        # are indeed connected
        self.wait_for(clients_wait_for)

        # Sample clients which meet the criterion
        available_cids = list(self.clients)
        # print(f'available_cids: {available_cids}')
        if criterion is not None:
            available_cids = [
                cid for cid in available_cids if criterion.select(self.clients[cid])
            ]

        num_clients = min(len(available_cids), num_clients)
        sampled_cids = random.sample(available_cids, num_clients)
        return [self.clients[cid] for cid in sampled_cids]
