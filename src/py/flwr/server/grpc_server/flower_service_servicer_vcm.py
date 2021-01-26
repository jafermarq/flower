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
"""Servicer for FlowerService.

Relevant knowledge for reading this modules code:
    - https://github.com/grpc/grpc/blob/master/doc/statuscodes.md
"""
from typing import Callable, Iterator

import grpc

from flwr.proto import transport_pb2_grpc
from flwr.proto.transport_pb2 import VirtualClientManagerMessage, RemoteClientManagerMessage
from flwr.server.client_manager import RemoteClientManager
from flwr.server.grpc_server.grpc_bridge import GRPCBridge_VCM
from flwr.server.grpc_server.grpc_virtual_client_manager_proxy import GrpcVirtualClientManagerProxy


def default_bridge_vcm_factory() -> GRPCBridge_VCM:
    """Return GRPCBridge instance."""
    return GRPCBridge_VCM()


def default_grpc_virtual_client_manager_factory(bridge: GRPCBridge_VCM) -> GrpcVirtualClientManagerProxy:
    """Return GrpcVirtualClientManagerProxy instance."""
    return GrpcVirtualClientManagerProxy(bridge=bridge)


def register_virtual_client_manager(
    client_manager: RemoteClientManager,
    vcm: GrpcVirtualClientManagerProxy,
    context: grpc.ServicerContext,
) -> bool:
    """Try registering GrpcVirtualClientManagerProxy with RemoteClientManager."""
    is_success = client_manager.register_vcm(vcm)
    if is_success:

        def rpc_termination_callback() -> None:
            vcm.bridge.close()
            client_manager.unregister_vcm()

        context.add_callback(rpc_termination_callback)

    return is_success


class FlowerServiceServicerVCM(transport_pb2_grpc.FlowerServiceVCMServicer):
    """FlowerServiceServicerVCM for bi-directional gRPC message stream between
    a RemoteClientManager on the server side and a VirtualClientManager on the
    client side."""

    def __init__(
        self,
        client_manager: RemoteClientManager,
        grpc_bridge_factory: Callable[[], GRPCBridge_VCM] = default_bridge_vcm_factory,
        grpc_vcm_factory: Callable[
            [str, GRPCBridge_VCM], GrpcVirtualClientManagerProxy
        ] = default_grpc_virtual_client_manager_factory,
    ) -> None:
        self.client_manager: RemoteClientManager = client_manager
        self.grpc_bridge_factory = grpc_bridge_factory
        self.vcm_factory = grpc_vcm_factory

    def Join(  # pylint: disable=invalid-name
        self,
        request_iterator: Iterator[VirtualClientManagerMessage],
        context: grpc.ServicerContext,
    ) -> Iterator[RemoteClientManagerMessage]:
        """Method will be invoked by each GrpcClientProxy which participates in
        the network.
        # TODO: Update this description
        Protocol:
            - The first message is sent from the server to the client
            - Both ServerMessage and ClientMessage are message "wrappers"
                wrapping the actual message
            - The Join method is (pretty much) protocol unaware
        """

        bridge = self.grpc_bridge_factory()
        client = self.vcm_factory(bridge)
        is_success = register_virtual_client_manager(self.client_manager, client, context)

        if is_success:
            # Get iterators
            client_message_iterator = request_iterator
            server_message_iterator = bridge.server_message_iterator()
            # All messages will be pushed to client bridge directly
            while True:
                try:
                    # Get server message from bridge and yield it
                    server_message = next(server_message_iterator)
                    yield server_message
                    # Wait for client message
                    client_message = next(client_message_iterator)
                    bridge.set_client_message(client_message)
                except StopIteration:
                    break
