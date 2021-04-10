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

import random
import copy

from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import FitRes, FitIns, EvaluateIns, Scalar, Weights, parameters_to_weights, weights_to_parameters
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from .aggregate import aggregate

from .fedavg import FedAvg


class FedDelay(FedAvg):
    """Configurable FedDelay strategy implementation."""

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
        aggregate_every_n: int = 5,
        num_rounds_after_final_aggregation: int = 1,
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
            initial_parameters=None,
        )

        self.aggregate_every_n = aggregate_every_n
        self.num_rounds_after_final_aggregation = num_rounds_after_final_aggregation

    def __repr__(self) -> str:
        rep = f"FedDelay(accept_failures={self.accept_failures})"
        return rep

    def configure_fit(
        self, rnd: int, weights: Union[Weights, List[Weights]],
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:

        ''' Prepares instructions for each client. Here each client, will
        make use of a different model. '''

        # if at the beginning of the FL experiment, we might only have a single model,
        # if that's the case, then deep copy it N times (N=num_clients_per_round).
        # Then proceed as normal
        if rnd == 1:
            print(f"Replicating single model into {self.min_fit_clients}")
            # ! This is only going to work for the VCM setup, where the numbe of clients
            # ! per round is fixed (so there's no really a notion of `min_fit_clients`)
            weights = [weights for _ in range(self.min_fit_clients)]

        parameters = [weights_to_parameters(w) for w in weights]

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        client_ins_pairs = []
        # Return client/config pairs (additionally, we keep track of which model is being
        # trained in which client)
        for i in range(len(clients)):
            config = {}
            if self.on_fit_config_fn is not None:
                # Custom fit config function provided
                config = self.on_fit_config_fn(rnd)
            config["model_id"] = i
            ins = FitIns(parameters[i], config)
            client_ins_pairs.append((clients[i], ins))

        return client_ins_pairs

    def configure_evaluate(
        self, rnd: int, weights: Weights, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if a centralized evaluation
        # function is provided
        # if self.eval_fn is not None:
        #     return []

        # Parameters and config
        # if isinstance(weights, Weights):
        #     weights = [weights for _ in range(self.min_fit_clients)]

        parameters = [weights_to_parameters(w) for w in weights]

        # Sample clients
        if rnd >= 0:
            sample_size, min_num_clients = self.num_evaluation_clients(
                client_manager.num_available()
            )
            clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=min_num_clients
            )
        else:
            clients = list(client_manager.all().values())

        # Return client/config pairs
        # It can happen that we want to evaluate on more/less number of clients than
        # the number of clients used for fit (which is the number of different models
        # being learnt in parallel). We need to deal with this situation. 
        # one simple approach is to let each model being used multiple times

        model_idx = [random.randrange(start=0, stop=len(parameters)-1) for _ in range(len(clients))]
        client_ins_pairs = []
        for i in range(len(clients)):

            config = {}
            # so client knows which of its partitions to use val/test
            config["is_test"] = self.eval_test_set
            if self.on_evaluate_config_fn is not None:
                # Custom evaluation config function provided
                config = self.on_evaluate_config_fn(rnd)
            config["model_id"] = model_idx[i]
            client_ins_pairs.append((clients[i], EvaluateIns(parameters[model_idx[i]], config)))

        return client_ins_pairs

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Weights]:

        ''' Only if rnd%self.aggregate_every_n==0 we aggregate the incoming models.
        This will return a list of Weights where all entries are the same aggregated
        model. If not aggregating, then we return a list of Weights, where the i-th
        entry is the model returned by the i-th sampled client. Returning `None` will
        result in round r+1 using the same models as in round r.'''

        if not results:
            return None
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None

        # conver results into weights and aggregate if required

        # TODO: If at the end of all rounds we want to merge all models and evaluate
        # a single one on the test set (after a few rounds of normal train+aggregate rounds)
        # we could do it by adding `or rdn > total_rounds - self.num_rounds_after_final_aggregation`
        if rnd % self.aggregate_every_n == 0:
            # aggregate
            print("Aggregating incoming models... FedAvg style")
            weights_results = [
                (parameters_to_weights(fit_res.parameters), fit_res.num_examples)
                for client, fit_res in results
            ]
            agg_weights = aggregate(weights_results)

            # now we generate N copies of the aggregated model (meaning than in the next round
            # all clients will start for from the sample gobal model)
            return [agg_weights for _ in range(self.min_fit_clients)]
        else:
            # don't aggregate, next FL round will train each model with a different client
            weights = [parameters_to_weights(fit_res.parameters) for _, fit_res in results]
            return weights

    # def evaluate(self, weights: Weights) -> Optional[Tuple[float, float]]:
    #     """ Evaluate model weights using an evaluation function (if
    #     provided). This is only used when doing centralised evaluation.
    #     """
    #     if self.eval_fn is None:
    #         # No evaluation function provided
    #         return None
    #     return self.eval_fn(weights)
