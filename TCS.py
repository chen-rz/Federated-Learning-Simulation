import math
import random
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

import flwr.common
import pandas
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server import SimpleClientManager
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.strategy.strategy import Strategy

from constants import *


class TCS_ClientManager(SimpleClientManager):
    def sample(self, num_clients: int, min_num_clients=2, time_constr=500):
        # For model initialization
        if num_clients == 1:
            return [self.clients[str(random.randint(0, pool_size - 1))]]

        # Sample clients which meet the criterion
        param_dicts = []
        available_cids = []
        total_data_amount = 0
        P = dict()
        for _ in range(pool_size + 1):
            P[_] = []

        for n in range(pool_size):
            # Get each client's parameters
            param_dicts.append(
                self.clients[str(n)].get_properties(
                    flwr.common.GetPropertiesIns(config={}), 68400
                ).properties.copy()
            )

            # Calculate max tolerating number of peers
            t_upload_available = time_constr - param_dicts[n]["updateTime"]
            max_peer = math.floor(
                t_upload_available * sys_bandwidth * math.log2(
                    1 + sys_channelGain * param_dicts[n]["transPower"] / sys_bgNoise
                ) / sys_modelSize
            )
            if max_peer < 0:
                max_peer = 0
            if max_peer > pool_size:
                max_peer = pool_size

            param_dicts[n]["maxPeer"] = max_peer
            P[max_peer].append(n)

            param_dicts[n]["isSelected"] = False

        # TCS algorithm
        for j in range(1, pool_size + 1):
            S = []  # all clients that tolerate j peers
            K_j = []  # available cids under j-peer condition
            D_j = 0  # total data amount under j-peer condition

            # you are available since you tolerate more than j peers
            for m in range(j, pool_size + 1):
                S += P[m]

            # select the most data-rich j clients
            l = j
            while l > 0:
                l -= 1

                # S is empty
                if not S:
                    continue

                # argmax
                max_data_amount = 0
                index_with_max_data = -1
                for s in S:
                    if param_dicts[s]["dataSize"] > max_data_amount:
                        max_data_amount = param_dicts[s]["dataSize"]
                        index_with_max_data = s

                K_j.append(index_with_max_data)
                D_j += param_dicts[index_with_max_data]["dataSize"]
                S.remove(index_with_max_data)

            if D_j > total_data_amount:
                total_data_amount = D_j
                available_cids = K_j.copy()

        # Record client parameters
        for n in range(pool_size):
            param_dicts[n]["bandwidth"] = sys_bandwidth / len(available_cids)
            param_dicts[n]["transRate"] = \
                param_dicts[n]["bandwidth"] * math.log2(
                    1 + sys_channelGain * param_dicts[n]["transPower"] / sys_bgNoise
                )
            param_dicts[n]["uploadTime"] = sys_modelSize / param_dicts[n]["transRate"]
            param_dicts[n]["totalTime"] = \
                param_dicts[n]["updateTime"] + param_dicts[n]["uploadTime"]

        fit_round_time = 0
        for _ in available_cids:
            param_dicts[_]["isSelected"] = True
            if param_dicts[_]["totalTime"] > fit_round_time:
                fit_round_time = param_dicts[_]["totalTime"]

        return [self.clients[str(cid)] for cid in available_cids], \
            {
                "clients_selected": available_cids,
                "data_amount": total_data_amount,
                "time_elapsed": fit_round_time,
                "time_constraint": time_constr
            }, \
            param_dicts


class TCS(Strategy):
    # pylint: disable=too-many-arguments,too-many-instance-attributes,line-too-long
    def __init__(
            self,
            *,
            evaluate_fn: Optional[
                Callable[
                    [int, NDArrays, Dict[str, Scalar]],
                    Optional[Tuple[float, Dict[str, Scalar]]],
                ]
            ] = None,
            on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            accept_failures: bool = True,
            initial_parameters: Optional[Parameters] = None,
            fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
            evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:

        super().__init__()

        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn

    def __repr__(self) -> str:
        rep = f"TCS (accept_failures={self.accept_failures})"
        return rep

    def initialize_parameters(
            self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def evaluate(
            self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def configure_fit(
            self, server_round: int, parameters: Parameters,
            client_manager: TCS_ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit_clients config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Time constraint
        if server_round == 1:
            C_T_i = timeConstrGlobal / num_rounds
        else:
            with open(
                    "./output/fit_server/round_{}.txt".format(server_round - 1)
            ) as inputFile:
                fit_round_dict = eval(inputFile.readline())
                C_T_i = timeConstrGlobal / num_rounds + \
                        fit_round_dict["time_constraint"] - fit_round_dict["time_elapsed"]

        # Sample clients
        clients, fit_round_dict, param_dicts = client_manager.sample(
            num_clients=0, time_constr=C_T_i
        )

        # Record information of clients
        pandas.DataFrame.from_records(param_dicts).to_excel(
            "./output/fit_clients/fit_round_{}.xlsx".format(server_round)
        )

        # Record information of server
        with open(
                "./output/fit_server/round_{}.txt".format(server_round),
                mode='w'
        ) as outputFile:
            outputFile.write(str(fit_round_dict))

        # Check train_loss records
        if server_round > 1:
            with open(
                    "./output/fit_server/round_{}.txt".format(server_round - 1),
                    mode='r'
            ) as inputFile:
                clients_of_prev_round = eval(inputFile.readline())["clients_selected"]

            for _ in range(pool_size):
                # If the client was not selected in the previous round,
                # help it complete the records
                if _ not in clients_of_prev_round:
                    with open(
                            "./output/train_loss/client_{}.txt".format(_),
                            mode='a'
                    ) as outputFile:
                        outputFile.write("-1" + "\n")

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
            self, server_round: int, parameters: Parameters,
            client_manager: TCS_ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        clients, fit_round_dict, param_dicts = client_manager.sample(
            num_clients=0, time_constr=timeConstrGlobal
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit_clients results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated
