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
    if rcm_msg.HasField("disconnect"):
        disconnect_msg, sleep_duration = _disconnect(vcm)
        return disconnect_msg, sleep_duration, False
    if rcm_msg.HasField("get_pool_size"):
        return _get_pool_ids(vcm), 0, True
    if rcm_msg.HasField("wakeup_clients"):
        return _wakeup_clients(vcm, rcm_msg.wakeup_clients), 0, True
    if rcm_msg.HasField("is_ready_for_sampling"):
        return _is_ready_for_sampling(vcm), 0, True
    if rcm_msg.HasField("set_config"):
        return _set_config(vcm, rcm_msg.set_config), 0, True
    raise UnkownRemoteClientManagerMessage()


def _get_pool_ids(vcm: VirtualClientManager) -> VirtualClientManagerMessage:
    # print("> vcm.grpc_vcm.message_handler._get_pool_size()")
    # No need to deserialize _get_pool_size (it's empty)
    pool_size_res = vcm.get_pool_ids()
    pool_size_res_proto = serde.get_pool_size_res_to_proto(pool_size_res)
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


# pylint: disable=unused-argument
def _disconnect(vcm: VirtualClientManager) -> Tuple[VirtualClientManagerMessage, int]:
    # Determine the reason for sending Disconnect message
    reason = Reason.ACK
    sleep_duration = 0
    # Build Disconnect message
    disconnect_res_proto = serde.disconnect_vcm_res_to_proto(reason)
    return VirtualClientManagerMessage(disconnect_res=disconnect_res_proto), sleep_duration


def _is_ready_for_sampling(vcm: VirtualClientManager) -> VirtualClientManagerMessage:
    # No need to deserialize _wait_for_sampling (it's empty)
    res = vcm.is_ready_for_sampling()
    # print("serde._is_ready_for_sampling()")
    # print(f"res: {res}")
    res_proto = serde.is_ready_for_sampling_res_to_proto(res)
    # print(f"res_proto: {res_proto}")
    return VirtualClientManagerMessage(is_ready_for_sampling_res=res_proto)


def _set_config(vcm: VirtualClientManager, mssg: RemoteClientManagerMessage.SetConfig) -> VirtualClientManagerMessage:

    config_str = serde.setconfig_from_proto(mssg)
    vcm.set_config(config_str)

    # return ACK back to RCM
    res = serde.set_config_res_to_proto(Reason.ACK)
    return VirtualClientManagerMessage(set_config_res=res)
