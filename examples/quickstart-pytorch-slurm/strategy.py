import logging
from typing import List, Tuple, Union
from flwr.common import FitIns, Parameters, EvaluateIns, GetPropertiesIns
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> CNFG_FIT:
        # Call the parent class method to retain behaviour
        fit_config = super().configure_fit(server_round, parameters, client_manager)

        # Now discard based on client's enrollment status
        return filter_by_client_enrollement(fit_config, "fit")

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> CNFG_EVAL:
        # Call the parent class method to retain behaviour
        eval_config = super().configure_evaluate(
            server_round, parameters, client_manager
        )

        # Now discard based on client's enrollment status
        return filter_by_client_enrollement(eval_config, "eval")
