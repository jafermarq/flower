import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
from hydra.utils import instantiate

from flwr.common import (
    FitIns,
    Parameters,
    EvaluateIns,
    GetPropertiesIns,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


log = logging.getLogger(__name__)

CNFG_FIT = List[Tuple[ClientProxy, FitIns]]
CNFG_EVAL = List[Tuple[ClientProxy, EvaluateIns]]


def filter_by_client_enrollement(
    workload: Union[CNFG_FIT, CNFG_EVAL], keyword: str
) -> Union[CNFG_FIT, CNFG_EVAL]:
    """This helper function filters the outputs of `configure_fit` and
    `configure_evaluate` and discards sending the instructions to those clients that
    aren't enrolled in a given taks (i.e. either fit, evalute)"""

    ins = GetPropertiesIns({})
    config_final = []
    for client, fit_or_eval_ins in workload:
        prop = client.get_properties(ins, timeout=30)
        if prop.properties[keyword]:
            config_final.append((client, fit_or_eval_ins))
            log.info(f"Client {prop.properties['name']} enrolled in {keyword} round.")
        else:
            log.info(f"Client {prop.properties['name']} excluded from {keyword} round.")

    return config_final


class CustomFedAvg(FedAvg):
    """This is a strategy that behaves like FedAvg.

    However it restricst which clients
    do `fit()` and which do `evaluate()` based on their enrolment (defined in the client
    properties -- see the client class used in this example).
    In addition, this CustomFedAvg adds support to load the global model from a checkpoint
    and to save the status of the global model at the end of the round i.e. after running
    `evaluate`(which runs the global evaluation stage -- if defined).
    """

    def __init__(self, cfg_model, path_to_checkpoint, exp_dir, *args, **kwargs):
        self.exp_dir = exp_dir
        self.cfg_model = cfg_model
        init_params = None
        if path_to_checkpoint:
            init_params = self._get_initial_parameters(path_to_checkpoint)

        super().__init__(*args, **kwargs, initial_parameters=init_params)

    def _get_initial_parameters(self, path_to_checkpoint):
        """Instantiates the model, applies weights from checkpoint and returns model as
        parameters (i.e. the framework-agnostic representation used in the Flower
        strategies + commms)"""
        model = instantiate(self.cfg_model)
        model.load_state_dict(torch.load(path_to_checkpoint))
        log.info(f"Loaded weights from checkpoint: {path_to_checkpoint}")
        ndarrays = [val.cpu().numpy() for _, val in model.state_dict().items()]
        return ndarrays_to_parameters(ndarrays)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> CNFG_FIT:
        """Standard configure_fit, but drops (clients, instructions) pairs based on
        client enrolment."""
        # Call the parent class method to retain behaviour
        fit_config = super().configure_fit(server_round, parameters, client_manager)

        # Now discard based on client's enrollment status
        return filter_by_client_enrollement(fit_config, "fit")

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> CNFG_EVAL:
        """Standard configure_evaluate, but drops (clients, instructions) pairs based on
        client enrolment."""
        # Call the parent class method to retain behaviour
        eval_config = super().configure_evaluate(
            server_round, parameters, client_manager
        )

        # Now discard based on client's enrollment status
        return filter_by_client_enrollement(eval_config, "eval")

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Tuple[float, Dict[str, Scalar]] | None:
        # Call evaluate as it would normally be called. We will return
        # it's output so to not interrupt the normal flow
        evaluate_output = super().evaluate(server_round, parameters)

        parameters_ndarrays = parameters_to_ndarrays(parameters)
        # But before we do so, let's save the global model to disk
        # as a standar PyTorch state dict. The steps below are identical
        # to what we do in `evaluate()` we defined in `get_evaluate_fn`
        model = instantiate(self.cfg_model)
        params_dict = zip(model.state_dict().keys(), parameters_ndarrays)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})

        # now save as usual
        checkpoint_dir = f"{self.exp_dir}/global_model_round_{server_round}.pt"
        torch.save(state_dict, checkpoint_dir)
        log.info(f"Saved checkpoint in: {checkpoint_dir}")
        # Here you could introduce some logic to only save the model every N rounds
        # or if the global acc/loss has increased/decreased.
        return evaluate_output
