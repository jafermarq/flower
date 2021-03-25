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
"""Adaptive Federated Optimization (FedOpt) [Reddi et al., 2020] abstract
strategy.
Paper: https://arxiv.org/abs/2003.00295
"""


from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from flwr.common import FitRes, FitIns, Scalar, Weights, parameters_to_weights
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from .fedavg import FedAvg


class FedOpt(FedAvg):
    """Configurable FedAdagrad strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        *,
        fraction_fit: float = 0.1,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 2,
        min_eval_clients: int = 2,
        min_available_clients: int = 2,
        eval_fn: Optional[Callable[[Weights], Optional[Tuple[float, float]]]] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Weights,
        eta: float = 1e-1,
        eta_l: float = 1e-1,
        tau: float = 1e-9,
    ) -> None:
        """Federated Optim strategy interface.
        Implementation based on https://arxiv.org/abs/2003.00295
        Args:
            fraction_fit (float, optional): Fraction of clients used during
                training. Defaults to 0.1.
            fraction_eval (float, optional): Fraction of clients used during
                validation. Defaults to 0.1.
            min_fit_clients (int, optional): Minimum number of clients used
                during training. Defaults to 2.
            min_eval_clients (int, optional): Minimum number of clients used
                during validation. Defaults to 2.
            min_available_clients (int, optional): Minimum number of total
                clients in the system. Defaults to 2.
            eval_fn (Callable[[Weights], Optional[Tuple[float, float]]], optional):
                Function used for validation. Defaults to None.
            on_fit_config_fn (Callable[[int], Dict[str, str]], optional):
                Function used to configure training. Defaults to None.
            on_evaluate_config_fn (Callable[[int], Dict[str, str]], optional):
                Function used to configure validation. Defaults to None.
            accept_failures (bool, optional): Whether or not accept rounds
                containing failures. Defaults to True.
            initial_parameters (Weights): Initial set of parameters from the server.
            eta (float, optional): Server-side learning rate. Defaults to 1e-1.
            eta_l (float, optional): Client-side learning rate. Defaults to 1e-1.
            tau (float, optional): Controls the algorithm's degree of adaptability.
                Defaults to 1e-9.
        """
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_eval=fraction_eval,
            min_fit_clients=min_fit_clients,
            min_eval_clients=min_eval_clients,
            min_available_clients=min_available_clients,
            eval_fn=eval_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
        )
        self.current_weights = initial_parameters
        self.delta_t: Optional[Weights] = None
        self.v_t: Optional[Weights] = None
        self.eta = eta
        self.eta_l = eta_l
        self.tau = tau
        self.beta_1 = None

    def __repr__(self) -> str:
        rep = f"FedOpt(accept_failures={self.accept_failures})"
        return rep

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Weights]:

        # convert message
        results = [
                    (parameters_to_weights(fit_res.parameters), fit_res.num_examples)
                    for client, fit_res in results
                ]

        ## compute all delta_i_t (line 9 Alg2)
        delta_i_t = [[layer_w - self.current_weights[i] for i, layer_w in enumerate(client_res)] for client_res, _ in results]

        ## compute delta_t: (1/S)*sum(delta_i_t)
        S = len(results) # total number of clients that participated
        delta_t = delta_i_t[0]
        for d_i_t in delta_i_t[1:]:
            delta_t = [sum(layer) for layer in zip(delta_t, d_i_t)]
        delta_t = [d/S for d in delta_t]

        # Delta_t in Line 10 Eq2
        if self.delta_t is None:
            self.delta_t = [np.zeros_like(subset_weights) for subset_weights in delta_t]

        # V_t in Line 11 Eq2
        if self.v_t is None:
            self.v_t = [np.zeros_like(subset_weights) for subset_weights in delta_t]

        # Delta_t in Line 10 Eq2
        self.delta_t = [self.beta_1*self.delta_t[i] + (1.0-self.beta_1)*res_layer for i, res_layer in enumerate(delta_t)] 

    def configure_fit(
        self, rnd: int, weights: Weights, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Save weights locally and configure the next round of training."""
        self.current_weights = weights

        # Return client/config pairs
        return super().configure_fit(rnd, weights, client_manager)