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
"""This module contains functions for protobuf serialization and
deserialization."""


from typing import Any, List, cast

from flwr.proto.transport_pb2 import (
    ClientMessage,
    Parameters,
    Reason,
    Scalar,
    ServerMessage,
    VirtualClientManagerMessage,
    RemoteClientManagerMessage
)

from . import typing

# pylint: disable=missing-function-docstring


def parameters_to_proto(parameters: typing.Parameters) -> Parameters:
    """."""
    return Parameters(tensors=parameters.tensors, tensor_type=parameters.tensor_type)


def parameters_from_proto(msg: Parameters) -> typing.Parameters:
    """."""
    tensors: List[bytes] = list(msg.tensors)
    return typing.Parameters(tensors=tensors, tensor_type=msg.tensor_type)


#  === Reconnect message ===


def reconnect_to_proto(reconnect: typing.Reconnect) -> ServerMessage.Reconnect:
    """Serialize flower.Reconnect to ProtoBuf message."""
    return ServerMessage.Reconnect(seconds=reconnect.seconds)


def reconnect_from_proto(msg: ServerMessage.Reconnect) -> typing.Reconnect:
    """Deserialize flower.Reconnect from ProtoBuf message."""
    return typing.Reconnect(seconds=msg.seconds)


# === Disconnect message ===


def disconnect_to_proto(disconnect: typing.Disconnect) -> ClientMessage.Disconnect:
    """Serialize flower.Disconnect to ProtoBuf message."""
    reason_proto = Reason.UNKNOWN
    if disconnect.reason == "RECONNECT":
        reason_proto = Reason.RECONNECT
    elif disconnect.reason == "POWER_DISCONNECTED":
        reason_proto = Reason.POWER_DISCONNECTED
    elif disconnect.reason == "WIFI_UNAVAILABLE":
        reason_proto = Reason.WIFI_UNAVAILABLE
    return ClientMessage.Disconnect(reason=reason_proto)


def disconnect_from_proto(msg: ClientMessage.Disconnect) -> typing.Disconnect:
    """Deserialize flower.Disconnect from ProtoBuf message."""
    print(f"serde.disconnect_from_proto: {msg}")
    if msg.reason == Reason.RECONNECT:
        return typing.Disconnect(reason="RECONNECT")
    if msg.reason == Reason.POWER_DISCONNECTED:
        return typing.Disconnect(reason="POWER_DISCONNECTED")
    if msg.reason == Reason.WIFI_UNAVAILABLE:
        return typing.Disconnect(reason="WIFI_UNAVAILABLE")
    return typing.Disconnect(reason="UNKNOWN")


# === GetWeights messages ===


def get_parameters_to_proto() -> ServerMessage.GetParameters:
    """."""
    return ServerMessage.GetParameters()


# Not required:
# def get_weights_from_proto(msg: ServerMessage.GetWeights) -> None:


def parameters_res_to_proto(res: typing.ParametersRes) -> ClientMessage.ParametersRes:
    """."""
    parameters_proto = parameters_to_proto(res.parameters)
    return ClientMessage.ParametersRes(parameters=parameters_proto)


def parameters_res_from_proto(msg: ClientMessage.ParametersRes) -> typing.ParametersRes:
    """."""
    parameters = parameters_from_proto(msg.parameters)
    return typing.ParametersRes(parameters=parameters)


# === Fit messages ===


def fit_ins_to_proto(ins: typing.FitIns) -> ServerMessage.FitIns:
    """Serialize flower.FitIns to ProtoBuf message."""
    parameters_proto = parameters_to_proto(ins.parameters)
    config_msg = metrics_to_proto(ins.config)
    return ServerMessage.FitIns(parameters=parameters_proto, config=config_msg)


def fit_ins_from_proto(msg: ServerMessage.FitIns) -> typing.FitIns:
    """Deserialize flower.FitIns from ProtoBuf message."""
    parameters = parameters_from_proto(msg.parameters)
    config = metrics_from_proto(msg.config)
    return typing.FitIns(parameters=parameters, config=config)


def fit_res_to_proto(res: typing.FitRes) -> ClientMessage.FitRes:
    """Serialize flower.FitIns to ProtoBuf message."""
    parameters_proto = parameters_to_proto(res.parameters)
    metrics_msg = None if res.metrics is None else metrics_to_proto(res.metrics)
    return ClientMessage.FitRes(
        parameters=parameters_proto,
        num_examples=res.num_examples,
        num_examples_ceil=res.num_examples_ceil,  # Deprecated
        fit_duration=res.fit_duration,  # Deprecated
        metrics=metrics_msg,
    )


def fit_res_from_proto(msg: ClientMessage.FitRes) -> typing.FitRes:
    """Deserialize flower.FitRes from ProtoBuf message."""
    parameters = parameters_from_proto(msg.parameters)
    metrics = None if msg.metrics is None else metrics_from_proto(msg.metrics)
    return typing.FitRes(
        parameters=parameters,
        num_examples=msg.num_examples,
        num_examples_ceil=msg.num_examples_ceil,  # Deprecated
        fit_duration=msg.fit_duration,  # Deprecated
        metrics=metrics,
    )


# === Evaluate messages ===


def evaluate_ins_to_proto(ins: typing.EvaluateIns) -> ServerMessage.EvaluateIns:
    """Serialize flower.EvaluateIns to ProtoBuf message."""
    parameters_proto = parameters_to_proto(ins.parameters)
    config_msg = metrics_to_proto(ins.config)
    return ServerMessage.EvaluateIns(parameters=parameters_proto, config=config_msg)


def evaluate_ins_from_proto(msg: ServerMessage.EvaluateIns) -> typing.EvaluateIns:
    """Deserialize flower.EvaluateIns from ProtoBuf message."""
    parameters = parameters_from_proto(msg.parameters)
    config = metrics_from_proto(msg.config)
    return typing.EvaluateIns(parameters=parameters, config=config)


def evaluate_res_to_proto(res: typing.EvaluateRes) -> ClientMessage.EvaluateRes:
    """Serialize flower.EvaluateIns to ProtoBuf message."""
    metrics_msg = None if res.metrics is None else metrics_to_proto(res.metrics)
    return ClientMessage.EvaluateRes(
        loss=res.loss,
        num_examples=res.num_examples,
        accuracy=res.accuracy,  # Deprecated
        metrics=metrics_msg,
    )


def evaluate_res_from_proto(msg: ClientMessage.EvaluateRes) -> typing.EvaluateRes:
    """Deserialize flower.EvaluateRes from ProtoBuf message."""
    metrics = None if msg.metrics is None else metrics_from_proto(msg.metrics)
    return typing.EvaluateRes(
        loss=msg.loss,
        num_examples=msg.num_examples,
        accuracy=msg.accuracy,  # Deprecated
        metrics=metrics,
    )


# === GetPoolSize messages ===

def get_pool_size_to_proto() -> RemoteClientManagerMessage.GetPoolSize:
    """."""
    return RemoteClientManagerMessage.GetPoolSize()


def get_pool_size_res_to_proto(res: typing.GetPoolSizeRes) -> VirtualClientManagerMessage.GetPoolSizeRes:

    train_ids_proto = scalar_list_to_proto(res.train_ids)
    val_ids_proto = None if res.val_ids is None else scalar_list_to_proto(res.val_ids)
    test_ids_proto = None if res.test_ids is None else scalar_list_to_proto(res.test_ids)
    # print(f"serde.get_pool_size_res_to_proto -> res: {res}")
    return VirtualClientManagerMessage.GetPoolSizeRes(train_ids=train_ids_proto,
                                                      val_ids=val_ids_proto,
                                                      test_ids=test_ids_proto)


def get_pool_size_res_from_proto(msg: VirtualClientManagerMessage.GetPoolSizeRes) -> typing.GetPoolSizeRes:
    # print(f"get_pool_size_res_from_proto: {msg}")
    train_ids = scalar_list_from_proto(msg.train_ids)
    val_ids = None if msg.val_ids is None else scalar_list_from_proto(msg.val_ids)
    test_ids = None if msg.test_ids is None else scalar_list_from_proto(msg.test_ids)
    return typing.GetPoolSizeRes(train_ids=train_ids,
                                 val_ids=val_ids,
                                 test_ids=test_ids)


# === WakeUpClients messages ===

def wakeup_clients_to_proto(msg: typing.WakeUpClientsIns) -> RemoteClientManagerMessage.WakeUpClients:
    return RemoteClientManagerMessage.WakeUpClients(cids=msg.cids)


def wakeup_clients_from_proto(msg: RemoteClientManagerMessage.WakeUpClients) -> typing.WakeUpClientsIns:
    # print(f"serde.wakeup_clients_from_proto() -->  obtained: {msg}")
    return typing.WakeUpClientsIns(cids=msg.cids)


def wakeup_clients_res_to_proto(res: typing.WakeUpClientsRes) -> VirtualClientManagerMessage.WakeUpClientsRes:
    # print(f"serde.wakeup_clients_res_to_proto() -->  res: {res}")
    return VirtualClientManagerMessage.WakeUpClientsRes(reason=res)


def wakeup_clients_res_from_proto(msg: VirtualClientManagerMessage.WakeUpClientsRes) -> typing.WakeUpClientsRes:
    # print(f"serde.wakeup_clients_res_from_proto() -> msg: {msg}")
    return typing.WakeUpClientsRes(reason=msg.reason)


# === IsAvailable (VCM) messages ===

def is_available_to_proto() -> RemoteClientManagerMessage.IsAvailable:
    return RemoteClientManagerMessage.IsAvailable()


def is_available_res_to_proto(res: typing.IsAvailableRes) -> VirtualClientManagerMessage.IsAvailableRes:
    return VirtualClientManagerMessage.IsAvailableRes(available=res.status)


def is_available_res_from_proto(msg: VirtualClientManagerMessage.IsAvailableRes) -> typing.IsAvailableRes:
    status = msg.available
    return typing.IsAvailableRes(status=status)


# === Disconnect VCM ===

def disconnect_vcm_to_proto() -> RemoteClientManagerMessage.Disconnect:
    return RemoteClientManagerMessage.Disconnect()


def disconnect_vcm_res_to_proto(res: typing.Disconnect) -> VirtualClientManagerMessage.DisconnectRes:
    return VirtualClientManagerMessage.DisconnectRes(reason=res)


def disconnect_vcm_res_from_proto(msg: VirtualClientManagerMessage.DisconnectRes) -> typing.Disconnect:
    return typing.Disconnect(msg)


# === WaitForSampling messages ===


def is_ready_for_sampling_to_proto() -> RemoteClientManagerMessage.IsReadyForSampling:
    return RemoteClientManagerMessage.IsReadyForSampling()


def is_ready_for_sampling_res_to_proto(res: typing.ReadyForSamplingRes) -> VirtualClientManagerMessage.IsReadyForSamplingRes:
    # print("serde.is_ready_for_sampling_res_to_proto")
    return VirtualClientManagerMessage.IsReadyForSamplingRes(wait=res.wait, num_clients=res.num_clients)


def is_ready_for_sampling_res_from_proto(msg: VirtualClientManagerMessage.IsReadyForSamplingRes) -> typing.ReadyForSamplingRes:
    # print("serde.is_ready_for_sampling_res_from_proto")
    return typing.ReadyForSamplingRes(wait=msg.wait, num_clients=msg.num_clients)


# === SendConfig messages ===


def setconfig_to_proto(config: str) -> RemoteClientManagerMessage.SetConfig:
    return RemoteClientManagerMessage.SetConfig(config=config)


def setconfig_from_proto(msg: RemoteClientManagerMessage.SetConfig) -> str:
    return msg.config


def set_config_res_to_proto(res: typing.SetConfigRes) -> VirtualClientManagerMessage.SetConfigRes:
    return VirtualClientManagerMessage.SetConfigRes(reason=res)


# === Metrics messages ===


def metrics_to_proto(metrics: typing.Metrics) -> Any:
    """Serialize... ."""
    proto = {}
    for key in metrics:
        proto[key] = scalar_to_proto(metrics[key])
    return proto


def metrics_from_proto(proto: Any) -> typing.Metrics:
    """Deserialize... ."""
    metrics = {}
    for k in proto:
        metrics[k] = scalar_from_proto(proto[k])
    return metrics


def scalar_to_proto(scalar: typing.Scalar) -> Scalar:
    """Serialize... ."""

    if isinstance(scalar, bool):
        return Scalar(bool=scalar)

    if isinstance(scalar, bytes):
        return Scalar(bytes=scalar)

    if isinstance(scalar, float):
        return Scalar(double=scalar)

    if isinstance(scalar, int):
        return Scalar(sint64=scalar)

    if isinstance(scalar, str):
        return Scalar(string=scalar)

    raise Exception(
        f"Accepted types: {bool, bytes, float, int, str} (but not {type(scalar)})"
    )


def scalar_from_proto(scalar_msg: Scalar) -> typing.Scalar:
    """Deserialize... ."""
    scalar = getattr(scalar_msg, scalar_msg.WhichOneof("scalar"))
    return cast(typing.Scalar, scalar)


def scalar_list_to_proto(scalar_list: List[typing.Scalar]) -> List[Scalar]:
    return [scalar_to_proto(s) for s in scalar_list]


def scalar_list_from_proto(proto_scalar_list: List[Scalar]) -> List[typing.Scalar]:
    return [scalar_from_proto(ps) for ps in proto_scalar_list]
