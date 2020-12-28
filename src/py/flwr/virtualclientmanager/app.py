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
"""Flower virtual client manager app."""


import time
from logging import INFO

from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from flwr.common.logger import log

from .virtualclientmanager import VirtualClientManager
from .grpc_virtualclientmanager.connection import insecure_grpc_connection
from .grpc_virtualclientmanager.message_handler import handle


def start_virtual_client_manager(
    server_address: str,
    virtual_client_manager: VirtualClientManager,
    grpc_max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
) -> None:
    """Start a Flower VirtualClientManager which connects to a gRPC server.

    Arguments:
        server_address: str. The IPv6 address of the server. If the Flower
            server runs on the same machine on port 8080, then `server_address`
            would be `"[::]:8080"`.
        virtual_client_manager: # TODO
        grpc_max_message_length: int (default: 536_870_912, this equals 512MB).
            The maximum length of gRPC messages that can be exchanged with the
            Flower server. The default should be sufficient for most models.
            Users who train very large models might need to increase this
            value. Note that the Flower server needs to be started with the
            same value (see `flwr.server.start_server`), otherwise it will not
            know about the increased limit and block larger messages.

    Returns:
        None.
    """
    while True:
        sleep_duration: int = 0
        with insecure_grpc_connection(
            server_address, max_message_length=grpc_max_message_length
        ) as conn:
            receive, send = conn
            log(INFO, "Opened (insecure) gRPC connection - VCM")

            while True:
                server_message = receive()
                # print("VCM received a message.. passing to handler...")
                # print(f"server_message: {server_message}")
                vcm_message, sleep_duration, keep_going = handle(
                    virtual_client_manager, server_message
                )
                send(vcm_message)
                if not keep_going:
                    # if break inmediatelly, grpc will trigger
                    # GRPCBridgeClosed exception making the server crash
                    # adding a 1s sleep seems enough
                    # unsure atm why this isn't needed for clients app.py
                    time.sleep(1)
                    break
        if sleep_duration == 0:
            log(INFO, "Disconnect and shut down")
            break
        # Sleep and reconnect afterwards
        log(
            INFO,
            "Disconnect, then re-establish connection after %s second(s)",
            sleep_duration,
        )
        time.sleep(sleep_duration)
