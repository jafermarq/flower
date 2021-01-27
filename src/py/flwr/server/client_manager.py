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
from math import ceil

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

    def __init__(self, num_vcm: int = 1) -> None:
        super(RemoteClientManager, self).__init__()
        self.vcm: List[VirtualClientManagerProxy] = []
        self.pool_size: int
        self.num_vcm = num_vcm
        self.wait_until_vcm_is_available: bool = True
        self.vcm_failure: bool = False

    def wait_for_vcm(self, num_vcm: int, timeout: int = 86400) -> bool:
        """Block until a VirtualClientManager has connected.

        Current timeout default: 1 day.
        """

        print(f"Waiting for VirtualClientManager(s) to connect with server. Expecting {self.num_vcm}...")
        with self._cv:
            return self._cv.wait_for(
                lambda: len(self.vcm) >= num_vcm, timeout=timeout
            )

    def register_vcm(self, vcm: VirtualClientManagerProxy) -> bool:
        """Register Flower VirtualClientManagerProxy instance.

        Returns:
            bool: Indicating if registration was successful.
        """

        if self.num_vcm == len(self.vcm):
            print(f"RCM not accepting more VCMs, {len(self.vcm)}/{self.num_vcm} already connected")
            return False

        # add VCM to list
        self.vcm.append(vcm)
        print(f"RMC has {len(self.vcm)}/{self.num_vcm} VCM(s) connected")

        with self._cv:
            self._cv.notify_all()

        # print("RemoteClientManager.register_vcm: Registered a Virtual Client Manager")
        return True

    def unregister_vcm(self, vcm: VirtualClientManagerProxy) -> None:
        """Unregister Flower VirtualClientManager instance.

        This method is idempotent.
        """
        # TODO: when a VCM goes down we should: (1) tell remaining VCMs to reset, (2) end round
        idx = self.vcm.index(vcm)
        print(f"Unregistering the {idx}-th VCM")
        self.vcm.remove(vcm)
        self.vcm_failure = True

        print(f"RMC has {len(self.vcm)}/{self.num_vcm} VCM(s) connected")
        with self._cv:
            self._cv.notify_all()

    def get_virtual_pool_size(self) -> None:
        """ Gets the size of the virtual pool on the VCM's side."""

        # use first VCM to get pool size
        self.pool_size = self.vcm[0].get_pool_size().pool_size
        print(f"Received --> pool_size = {self.pool_size} clients.")

    def _cids_list_to_string(self, cids: List[int]):
        """Converst a list of integers into a csv string."""
        mssg = ""
        for n in cids[:-1]:
            mssg += str(n)
            mssg += ","
        mssg += str(cids[-1])
        return mssg

    def wakeup_clients(self, cids: List[int]) -> None:
        """ Tells VCM which clients to use in this round/sub-round,
        len(cids) == number clients to use in round. It can perfectly happen
        that the machine where VCM is, can't allocate that many clients at a
        given time, if this happens it will use the first N (where N is the number
        of clients that the machine can allocate), then the next N, until
        all clients have done the specified amount of local epochs.
        If there are"""

        num_vcm = len(self.vcm)
        if num_vcm > 1:
            chunk_size = ceil(len(cids)/num_vcm)
            # print(f"Splitting cids into {len(self.vcm)} partition of max_size: {chunk_size}...")
            sub_cids = [cids[x:x+chunk_size] for x in range(0, len(cids), chunk_size)]
            for i, cids in enumerate(sub_cids):
                mssg = self._cids_list_to_string(cids)
                mssg_ins = WakeUpClientsIns(cids=mssg)
                self.vcm[i].wakeup_clients(mssg_ins)
        else:
            mssg = self._cids_list_to_string(cids)
            mssg_ins = WakeUpClientsIns(cids=mssg)
            self.vcm[0].wakeup_clients(mssg_ins)

    def check_if_vcm_is_available(self) -> bool:
        """Check if there are Ray jobs running on the VirtualClientManager
        side."""
        # TODO: Can we simplify this?
        available = [False] * len(self.vcm)
        # We skip this if the VCM is in the middle of a round
        while not(all(available)) and self.wait_until_vcm_is_available:
            for i, vcm in enumerate(self.vcm):
                available[i] = vcm.is_available().status
            # print(f"available: {available} --> {not(all(available))}")
            time.sleep(5)

        self.wait_until_vcm_is_available = False
        # print(f"OBTAINED: {status_res}")
        return all(available)

    def start_new_round(self) -> None:
        """ Indicate that we want to wait in check_if_vcm_is_available()
        for the VCM to be ready. """
        self.wait_until_vcm_is_available = True
        self.vcm_failure = False

    def shutdown_vcm(self) -> None:
        """Tells VCM to shutdown."""
        print("Telling VCMs to shutdown...")
        for vcm in self.vcm:
            _ = vcm.disconnect()

    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        """ Randomly choose client id from pool, tell VirtualClientManager
        to wake up those clients, wait until they connect, then continue...
        Always num_clients==min_num_clients, this number is specified
        when passing the strategy object upon server construction."""

        # print(f"sample(self, {num_clients}, {min_num_clients})")
        if min_num_clients is None:
            # we'll reach this point when sampling client0 to init global weights
            min_num_clients = num_clients

        if not(self.vcm):
            # wait until VCMs are connected for the first time
            self.wait_for_vcm(self.num_vcm)
            self.get_virtual_pool_size()

        # Wakeup clients in cid in cids
        if self.check_if_vcm_is_available():
            # Take `min_num_clients` from total number of clients in pool
            cids = random.sample(list(range(self.pool_size)), min_num_clients)
            self.wakeup_clients(cids)

        # Block until clients managed by VCM are instantiated,
        # allocated, warmedup and connected to the server.

        wait = [True] * len(self.vcm)
        clients_wait_for = [0] * len(self.vcm)
        while all(wait):
            for i, vcm in enumerate(self.vcm):
                # ask VCM whether clients are ready and how many are online
                res = vcm.is_ready_for_sampling()
                wait[i] = res.wait
                clients_wait_for[i] = res.num_clients
            time.sleep(2)

        # ensure the number of clients that VCM said are online
        # are indeed connected
        self.wait_for(sum(clients_wait_for))

        # Sample clients which meet the criterion
        available_cids = list(self.clients)
        if criterion is not None:
            # TODO: this needs to be revisited (is it even ever used?)
            print("> Revisit the use of `criterion` for RemoteClientManager")
            available_cids = [
                cid for cid in available_cids if criterion.select(self.clients[cid])
            ]

        return [self.clients[cid] for cid in available_cids]
