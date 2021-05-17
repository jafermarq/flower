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
from logging import DEBUG, INFO

from .client_proxy import ClientProxy
from .virtual_client_manager_proxy import VirtualClientManagerProxy
from .criterion import Criterion
from flwr.common import WakeUpClientsIns, GetPoolSizeRes
from flwr.common.logger import log

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
        self.pool_ids: GetPoolSizeRes
        self.ids_to_use = None
        self.vcm_failure: bool = False

    def wait_for(self, num_clients: int, timeout: int = 86400) -> bool:
        """Block until at least `num_clients` are available or until a timeout
        is reached. Current timeout default: 1 day. If `vcm_failure`=True,
        then we'll stop waiting (and treat it as a failure inmediately. """
        with self._cv:
            return self._cv.wait_for(
                lambda: (len(self.clients) >= num_clients) or self.vcm_failure, timeout=timeout
            )

    def wait_for_vcm(self, timeout: int = 86400) -> bool:
        """Block until VirtualClientManager has connected. Current timeout
        default: 1 day. """

        print(f"Waiting for VirtualClientManager to connect with server.")
        with self._cv:
            return self._cv.wait_for(
                lambda: self.vcm is not None, timeout=timeout
            )

    def register_vcm(self, vcm: VirtualClientManagerProxy) -> bool:
        """Register Flower VirtualClientManagerProxy instance.

        Returns:
            bool: Indicating if registration was successful.
        """

        if self.vcm is not None:
            print(f"VCM already connected with RCM")
            return False

        # add VCM to list
        self.vcm = vcm

        with self._cv:
            self._cv.notify_all()

        # print("RemoteClientManager.register_vcm: Registered a Virtual Client Manager")
        return True

    def unregister_vcm(self, vcm: VirtualClientManagerProxy) -> None:
        """Unregister Flower VirtualClientManager instance.

        This method is idempotent.
        """
        log(DEBUG, f"Unregistering VCM")
        self.vcm = None

        # ! We'll flag as a "failure" when a VCM disconnects since they should only disconnect
        # ! at the end of the FL experiment (when instructed to do so by the server) - see end of server.fit()
        self.vcm_failure = True

        log(DEBUG, f"RMC has removed connected VCM")
        with self._cv:
            self._cv.notify_all()

    def get_virtual_pool_ids(self) -> None:
        """ Gets the ids of clients in the virtual pool on the VCM's side."""

        # use first VCM to get pool size
        self.pool_ids = self.vcm.get_pool_ids()
        train_ids_len = len(self.pool_ids.train_ids)
        val_ids_len = None if self.pool_ids.val_ids is None else len(self.pool_ids.val_ids)
        test_ids_len = None if self.pool_ids.test_ids is None else len(self.pool_ids.test_ids)
        log(INFO, f"Received --> pool_ids = List[train={train_ids_len}," +
              f"val={val_ids_len},test={test_ids_len}] IDs.")

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

        mssg = self._cids_list_to_string(cids)
        mssg_ins = WakeUpClientsIns(cids=mssg)
        self.vcm.wakeup_clients(mssg_ins)

    def start_new_round(self, clients_per_round: int) -> None:
        """ A new round begins -> wake up clients for this round."""
        self.vcm_failure = False

        # sample and wakeup clients for this round
        if len(self.ids_to_use) > 0:
            cids = random.sample(self.ids_to_use, clients_per_round)
            self.wakeup_clients(cids)
        else:
            log(DEBUG, "> List doesn't contain cids..")

    def update_id_list_to_use(self, ids: List[int]) -> None:
        """ Updates the list of client ids to wake up ahead of a call to
        sample(). For example, in some scenarios we might have some clients
        to be exclusivelly used for train or validation. """
        self.ids_to_use = ids

    def shutdown_vcm(self) -> None:
        """Tells VCM to shutdown."""
        log(DEBUG, "Telling VCMs to shutdown...")
        self.vcm.disconnect()

    def init_upon_vcm_connects(self, config: dict) -> None:
        """Waits for VCMs to connect with Server/RCM. Then sends the config to
        VCM and waits until received pool information."""
        if not(self.vcm):
            # wait until VCMs are connected for the first time
            self.wait_for_vcm()
            # send config to VCM
            res = self.vcm.set_config(config)
            if res != 4:  # 4 == ACK from proto
                log(DEBUG, f"Error sending config to vcm. Disconecting VCM")
                self.unregister_vcm(self.vcm)
            else:
                log(DEBUG, f"Seding config to vcm: ACK")
            self.get_virtual_pool_ids()

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

        # Block until clients managed by VCM are instantiated,
        # allocated, warmedup and connected to the server.

        wait = True
        clients_wait_for = 0
        while wait:
            # ask VCM whether clients are ready and how many are online
            res = self.vcm.is_ready_for_sampling()
            wait = res.wait
            clients_wait_for = res.num_clients
            time.sleep(2)

        # ensure the number of clients that VCM said are online
        # are indeed connected. Stop waiting if a vcm goes down.
        self.wait_for(clients_wait_for)

        # Sample clients which meet the criterion
        available_cids = list(self.clients)[:clients_wait_for]  # this gives extra safety (although it shouldn't be necessary)
        if criterion is not None:
            # TODO: this needs to be revisited (is it even ever used?)
            print("> Revisit the use of `criterion` for RemoteClientManager")
            available_cids = [
                cid for cid in available_cids if criterion.select(self.clients[cid])
            ]

        return [self.clients[cid] for cid in available_cids]
