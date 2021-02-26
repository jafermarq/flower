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
"""Networked Flower client implementation."""


from flwr import common
from flwr.common import serde
from flwr.proto.transport_pb2 import VirtualClientManagerMessage, RemoteClientManagerMessage
from flwr.server.virtual_client_manager_proxy import VirtualClientManagerProxy
from flwr.server.grpc_server.grpc_bridge import GRPCBridge


class GrpcVirtualClientManagerProxy(VirtualClientManagerProxy):
    """Flower virtual client manager proxy which delegates over the network using gRPC."""

    def __init__(
        self,
        bridge: GRPCBridge,
        pool_size: int = -1,
    ):
        super().__init__(pool_size)
        self.bridge = bridge

    def get_pool_ids(self) -> common.GetPoolSizeRes:

        get_pool_size_msg = serde.get_pool_size_to_proto()
        vcm_msg: VirtualClientManagerMessage = self.bridge.request(
            RemoteClientManagerMessage(get_pool_size=get_pool_size_msg)
        )
        pool_size_res = serde.get_pool_size_res_from_proto(vcm_msg.get_pool_size_res)
        return pool_size_res

    def wakeup_clients(self, ins: common.WakeUpClientsIns) -> common.WakeUpClientsRes:
        """Tells which clients in the virtual pool to instantiate."""
        # print("GrpcVirtualClientManagerProxy.wakeup_clients()")
        wakup_clients_msg = serde.wakeup_clients_to_proto(ins)
        vcm_msg: VirtualClientManagerMessage = self.bridge.request(
            RemoteClientManagerMessage(wakeup_clients=wakup_clients_msg)
        )

        wakeup_clients_res = serde.wakeup_clients_res_from_proto(vcm_msg.wakeup_clients_res)
        return wakeup_clients_res

    def disconnect(self) -> common.Disconnect:
        disconnect_msg = serde.disconnect_vcm_to_proto()

        vcm_msg: VirtualClientManagerMessage = self.bridge.request(
            RemoteClientManagerMessage(disconnect=disconnect_msg)
        )

        disconnect_res = serde.disconnect_vcm_res_from_proto(vcm_msg)

        return disconnect_res

    def is_ready_for_sampling(self) -> common.ReadyForSamplingRes:

        is_ready_for_sampling_msg = serde.is_ready_for_sampling_to_proto()

        vcm_msg: VirtualClientManagerMessage = self.bridge.request(
            RemoteClientManagerMessage(is_ready_for_sampling=is_ready_for_sampling_msg)
        )
        # print("is_ready_for_sampling")
        # print(f"vcm_msg: {vcm_msg}")
        return serde.is_ready_for_sampling_res_from_proto(vcm_msg.is_ready_for_sampling_res)

    def set_config(self, config: str) -> str:

        set_config_msg = serde.setconfig_to_proto(config)

        vcm_msg: VirtualClientManagerMessage = self.bridge.request(
            RemoteClientManagerMessage(set_config=set_config_msg)
        )

        # reply is a string (e.g. ACK), no need for fancy serde stages..
        res = vcm_msg.set_config_res.reason
        return res
