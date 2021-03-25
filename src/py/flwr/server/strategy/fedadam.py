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
"""Adaptive Federated Optimization using Adam (FedAdam) [Reddi et al., 2020]
strategy.
Paper: https://arxiv.org/abs/2003.00295
"""


from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from flwr.common import FitRes, Scalar, Weights
from flwr.server.client_proxy import ClientProxy

from .fedopt import FedOpt


class FedAdam(FedOpt):
    """Configurable FedAdam strategy implementation."""

    # pylint: disable-msg=too-many-arguments,too-many-instance-attributes
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
        initial_parameters: Weights = None,
        eta: float = 1e-1,
        eta_l: float = 3e-2,
        tau: float = 1e-2,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
    ) -> None:
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
            eta=eta,
            eta_l=eta_l,
            tau=tau,
        )
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.delta_t: Optional[Weights] = None
        self.v_t: Optional[Weights] = None

    def __repr__(self) -> str:
        rep = f"FedYogi(accept_failures={self.accept_failures})"
        return rep

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Weights]:
        """Aggregate fit results."""

        # Compute delta_t in Line10 Eq2
        super().aggregate_fit(
            rnd=rnd, results=results, failures=failures
        )

        # V_t in Line 12 Eq2
        delta_t_pow2 = [d_t**2 for i, d_t in enumerate(self.delta_t)]

        self.v_t = [self.v_t[i] - (1.0 - self.beta_2)*d2 * np.sign(self.v_t[i] - d2) for i, d2 in enumerate(delta_t_pow2)]

        # x_t_plus1 as in Line 14 Eq2
        x_t_plus1 = [self.current_weights[i] + self.eta * dd/(np.sqrt(self.v_t[i]) + self.tau) for i, dd in enumerate(self.delta_t)]

        self.current_weights = x_t_plus1

        return self.current_weights
