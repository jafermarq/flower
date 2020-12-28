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
"""Handle server messages by calling appropriate client methods."""


from typing import Tuple

from flwr.virtualclientmanager.virtualclientmanager import VirtualClientManager
from flwr.common import serde
from flwr.proto.transport_pb2 import VirtualClientManagerMessage, Reason, RemoteClientManagerMessage

# pylint: disable=missing-function-docstring


class UnkownRemoteClientManagerMessage(Exception):
    """Signifies that the received message is unknown."""


def handle(
    vcm: VirtualClientManager, rcm_msg: RemoteClientManagerMessage
) -> Tuple[VirtualClientManagerMessage, int, bool]:
    # print(f"RECEIVED: {rcm_msg}")
    if rcm_msg.HasField("get_pool_size"):
        return _get_pool_size(vcm), 0, True
    if rcm_msg.HasField("wakeup_clients"):
        return _wakeup_clients(vcm, rcm_msg.wakeup_clients), 0, True
    if rcm_msg.HasField("is_available"):
        return _is_available(vcm), 0, True
    raise UnkownRemoteClientManagerMessage()


def _get_pool_size(vcm: VirtualClientManager) -> VirtualClientManagerMessage:
    # print("> vcm.grpc_vcm.message_handler._get_pool_size()")
    # No need to deserialize _get_pool_size (it's empty)
    num_clients = vcm.get_pool_size()
    pool_size_res_proto = serde.get_pool_size_res_to_proto(num_clients)
    return VirtualClientManagerMessage(get_pool_size_res=pool_size_res_proto)


def _wakeup_clients(vcm: VirtualClientManager, wakeup_msg: RemoteClientManagerMessage.WakeUpClients) -> VirtualClientManagerMessage:
    # Deserialize wakeup instruction
    # print("> vcm.grpc_vcm.message_handler._wakeup()")
    wakeup_clients_ins = serde.wakeup_clients_from_proto(wakeup_msg)
    # print(f"Telling VCM to wakup clinets: {wakeup_clients_ins}")
    # Wake up clients w/ Ray
    vcm.wakeup_clients(wakeup_clients_ins.cids)
    res = serde.wakeup_clients_res_to_proto(Reason.ACK)
    return VirtualClientManagerMessage(wakeup_clients_res=res)


def _is_available(vcm: VirtualClientManager) -> VirtualClientManagerMessage:
    # print("> vcm.grpc_vcm.message_handler._is_available()")
    # No need to deserialize _is_available (it's empty)
    status = vcm.is_available()
    pool_size_res_proto = serde.is_available_res_to_proto(status)
    return VirtualClientManagerMessage(is_available_res=pool_size_res_proto)