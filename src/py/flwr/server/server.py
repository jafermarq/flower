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
"""Flower server."""


import concurrent.futures
import timeit
from logging import DEBUG, INFO
from typing import List, Optional, Tuple, cast, Callable, Dict
from tqdm import tqdm

from flwr.common import (
    Disconnect,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Reconnect,
    Weights,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager, RemoteClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.strategy import FedAvg, Strategy

FitResultsAndFailures = Tuple[List[Tuple[ClientProxy, FitRes]], List[BaseException]]
EvaluateResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, EvaluateRes]], List[BaseException]
]
ReconnectResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, Disconnect]], List[BaseException]
]


def set_strategy(strategy: Optional[Strategy]) -> Strategy:
    """Return Strategy."""
    return strategy if strategy is not None else FedAvg()


def func_to_method(f, obj):
    """ Makes function f be an obj's method. """
    return f.__get__(obj, type(obj))


class Server:
    """Flower server."""

    def __init__(
        self, client_manager: ClientManager,
        on_init_fn: Optional[Callable[[None], None]],
        on_round_end_fn: Optional[Callable[[Dict], None]],
        init_global_model_fn: Optional[Callable[[None], Weights]],
        strategy: Optional[Strategy] = None,
    ) -> None:
        self._client_manager: ClientManager = client_manager
        self.weights: Weights = []
        self.strategy: Strategy = set_strategy(strategy)
        self.starting_round = 1

        # to be passed during on_init()
        # this will be sent to the VCMs
        self.config = None

        self.eval_every_n = 1
        self.skip_validation_eval = False

        # init global model
        self.weights = init_global_model_fn()

        # make these actual class methods (if defined)
        self.on_init = func_to_method(on_init_fn, self)
        self.on_round_end = func_to_method(on_round_end_fn, self)

        self.on_init()

    def client_manager(self) -> ClientManager:
        """Return ClientManager."""
        return self._client_manager

    # pylint: disable=too-many-locals
    def fit(self, num_rounds: int) -> History:
        """Run federated averaging for a number of rounds."""
        history = History()

        # Run federated learning for num_rounds
        log(INFO, "[TIME] FL starting")
        start_time = timeit.default_timer()

        # wait until VCM(s) have connected, send config and
        # get pool client info
        assert(self.config is not None), ("Make sure you passed a serialised config during on_init()." +
                                         " This is needed to initialize the VCM(s).")
        self._client_manager.init_upon_vcm_connects(self.config)

        for current_round in range(self.starting_round, num_rounds + 1):
            # Train model and replace previous global model
            weights_prime, client_metrics = self.fit_round(rnd=current_round)
            if weights_prime is not None:
                self.weights = weights_prime

            metrics = {'acc_cen': None, 'loss_cen': None,
                       'acc_fed': None, 'loss_fed': None}

            if current_round % self.eval_every_n == 0:
                # Evaluate on validation set (always federated)
                if self.skip_validation_eval:
                    print("> Undefined Validation split (skipping round eval...)")
                    t_round = timeit.default_timer() - start_time
                else:
                    res_fed = self.evaluate(rnd=current_round, is_testset=False)
                    if res_fed is not None and res_fed[0] is not None:
                        loss_fed, acc_fed, _ = res_fed
                        metrics['loss_fed'] = loss_fed
                        metrics['acc_fed'] = acc_fed
                        t_round = timeit.default_timer() - start_time
                        log(
                            INFO, "eval(fed): (loss:%s, acc:%s, time:%s)",
                            loss_fed, acc_fed, t_round,
                        )
                        history.add_loss_distributed(
                            rnd=current_round, loss=cast(float, loss_fed)
                        )
                        history.add_accuracy_distributed(
                            rnd=current_round, acc=cast(float, acc_fed)
                        )

            # Round ended, run post round stages
            args = {'current_round': current_round, 'server_metrics': metrics,
                    'client_metrics': client_metrics,
                    'weights': self.weights, 't_round': t_round}

            self.on_round_end(args)

        # evaluate on test set
        print("> Training ended. Evaluating on test set")
        if self.strategy.eval_fn is not None:
            # centralized evaluation
            res_cen = self.strategy.evaluate(weights=self.weights)
            if res_cen is not None:
                loss_cen, acc_cen = res_cen
                metrics['acc_cen'], metrics['loss_cen'] = acc_cen, loss_cen
                t_round = timeit.default_timer() - start_time
                log(
                    INFO, "fit progress: (%s, %s, %s, %s)",
                    current_round, loss_cen,
                    acc_cen, t_round,
                )
                history.add_loss_centralized(rnd=current_round, loss=loss_cen)
                history.add_accuracy_centralized(rnd=current_round, acc=acc_cen)
        else:
            # evaluation on clients
            res_fed = self.evaluate(rnd=current_round, is_testset=True)
            if res_fed is not None and res_fed[0] is not None:
                loss_fed, acc_fed, _ = res_fed
                metrics['loss_fed'] = loss_fed
                metrics['acc_fed'] = acc_fed
                t_round = timeit.default_timer() - start_time
                log(
                    INFO, "eval(fed): (%s, %s, %s, %s)",
                    current_round, loss_fed,
                    acc_fed, t_round,
                )
                history.add_loss_distributed(
                    rnd=current_round, loss=cast(float, loss_fed)
                )
                history.add_accuracy_distributed(
                    rnd=current_round, acc=cast(float, acc_fed)
                )

        # Round ended, run post round stages
        args = {'current_round': current_round+1, 'server_metrics': metrics,
                'weights': self.weights, 't_round': t_round}

        self.on_round_end(args)

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "[TIME] FL finished in %s", elapsed)

        # Telling VirtualClientManager to shutdown
        if isinstance(self._client_manager, RemoteClientManager):
            self._client_manager.shutdown_vcm()

        return history

    def evaluate(
        self, rnd: int, is_testset: bool
    ) -> Optional[Tuple[Optional[float], EvaluateResultsAndFailures]]:
        """Validate current global model on a number of clients."""

        if isinstance(self._client_manager, RemoteClientManager):
            results_and_failures = self._round_with_rcm(self.strategy.configure_evaluate,
                                                        evaluate_clients,
                                                        rnd, is_testset)
            results, failures = results_and_failures
        else:
            raise NotImplementedError()

        # Aggregate the evaluation results
        loss_aggregated, acc_aggregated = self.strategy.aggregate_evaluate(rnd,
                                                                           results,
                                                                           failures)
        return loss_aggregated, acc_aggregated, results_and_failures

    def fit_round(self, rnd: int) -> Optional[Weights]:
        """Perform a single round of federated averaging."""

        if isinstance(self._client_manager, RemoteClientManager):

            results, failures = self._round_with_rcm(self.strategy.configure_fit,
                                                     fit_clients, rnd)
        else:
            raise NotImplementedError()

        log(
            DEBUG,
            "fit_round received %s results and %s failures",
            len(results),
            len(failures),
        )

        # Return metrics and aggregated training results
        metrics = [res[1].metrics for res in results]
        return self.strategy.aggregate_fit(rnd, results, failures), metrics

    def _round_with_rcm(self, get_instructions_fn, task_fn, rnd, is_testset: bool = False):

        results = []
        failures = []

        # determine what task are we donig (fit/evaluate) and notivy the RCM. This is
        # needed so the RCM samples from the correct list of client_ids for training and
        # for evaluation respectively.
        # this might seem like too much code for this purpose, we do it in this way so it's
        # easy to understand and revisit in the future if necessary.
        if task_fn == fit_clients:
            self._client_manager.update_id_list_to_use(self._client_manager.pool_ids.train_ids)
            num_to_sample = self.strategy.min_fit_clients
            tqdm_tile = f'Round #{rnd}'
        elif task_fn == evaluate_clients:
            # we ensure we sample all clients for val/test
            if is_testset:
                self._client_manager.update_id_list_to_use(self._client_manager.pool_ids.test_ids)
                num_to_sample = len(self._client_manager.pool_ids.test_ids)
                tqdm_tile = "Test"
            else:
                self._client_manager.update_id_list_to_use(self._client_manager.pool_ids.val_ids)
                num_to_sample = len(self._client_manager.pool_ids.val_ids)
                tqdm_tile = "Eval"
        else:
            raise NotImplementedError()

        # indicate that a new round starts so RCM should wait for VCM to be available
        self._client_manager.start_new_round(num_to_sample)

        with tqdm(total=num_to_sample, desc=tqdm_tile) as t:
            while len(results) < num_to_sample:

                # Get clients and their respective instructions from strategy
                client_instructions = get_instructions_fn(rnd=rnd, weights=self.weights,
                                                          client_manager=self._client_manager)
                if not client_instructions:
                    results = []
                    failures = []
                    for _ in range(num_to_sample):
                        failures.append(Exception("Empty client update"))
                    return results, failures

                # obtain results
                results_, failures_ = task_fn(client_instructions)

                # add to lists
                results += results_
                failures += failures_

                # shut down clients
                all_clients = self._client_manager.all()
                dis, err = shutdown(clients=[all_clients[k] for k in all_clients.keys()])
                t.set_postfix({'results': f"{len(results)}", 'failures':f"{len(failures)}"})
                t.update(len(results_))

        return results, failures

    def disconnect_all_clients(self) -> None:
        """Send shutdown signal to all clients."""
        all_clients = self._client_manager.all()
        _ = shutdown(clients=[all_clients[k] for k in all_clients.keys()])


def shutdown(clients: List[ClientProxy]) -> ReconnectResultsAndFailures:
    """Instruct clients to disconnect and never reconnect."""
    reconnect = Reconnect(seconds=None)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(reconnect_client, c, reconnect) for c in clients]
        concurrent.futures.wait(futures)
    # Gather results
    results: List[Tuple[ClientProxy, Disconnect]] = []
    failures: List[BaseException] = []
    for future in futures:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            result = future.result()
            results.append(result)
    return results, failures


def reconnect_client(
    client: ClientProxy, reconnect: Reconnect
) -> Tuple[ClientProxy, Disconnect]:
    """Instruct a single client to disconnect and (optionally) reconnect
    later."""
    disconnect = client.reconnect(reconnect)
    return client, disconnect


def fit_clients(
    client_instructions: List[Tuple[ClientProxy, FitIns]]
) -> FitResultsAndFailures:
    """Refine weights concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(fit_client, c, ins) for c, ins in client_instructions
        ]
        concurrent.futures.wait(futures)
    # Gather results
    results: List[Tuple[ClientProxy, FitRes]] = []
    failures: List[BaseException] = []
    for future in futures:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            # Potential success case
            result = future.result()
            if len(result[1].parameters.tensors) > 0:
                results.append(result)
            else:
                failures.append(Exception("Empty client update"))
    return results, failures


def fit_client(client: ClientProxy, ins: FitIns) -> Tuple[ClientProxy, FitRes]:
    """Refine weights on a single client."""
    fit_res = client.fit(ins)
    return client, fit_res


def evaluate_clients(
    client_instructions: List[Tuple[ClientProxy, EvaluateIns]]
) -> EvaluateResultsAndFailures:
    """Evaluate weights concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(evaluate_client, c, ins) for c, ins in client_instructions
        ]
        concurrent.futures.wait(futures)
    # Gather results
    results: List[Tuple[ClientProxy, EvaluateRes]] = []
    failures: List[BaseException] = []
    for future in futures:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            # Success case
            results.append(future.result())
    return results, failures


def evaluate_client(
    client: ClientProxy, ins: EvaluateIns
) -> Tuple[ClientProxy, EvaluateRes]:
    """Evaluate weights on a single client."""
    evaluate_res = client.evaluate(ins)
    return client, evaluate_res
