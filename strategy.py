import math
import random
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

import flwr.common
import pandas
import torchvision
from flwr.common import (
    EvaluateIns, EvaluateRes, FitIns, FitRes, GetParametersIns,
    ndarrays_to_parameters, parameters_to_ndarrays,
    MetricsAggregationFn, NDArrays, Parameters, Scalar,
)
from flwr.common.logger import log
from flwr.server import SimpleClientManager
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.strategy.strategy import Strategy

import client as clt
from constants import *
from dataset_utils import cifar10Transformation


class TCS_ClientManager(SimpleClientManager):
    def sample(self, num_clients: int, server_round=0, time_constr=500):
        # For model initialization
        if num_clients == 1:
            return [self.clients[str(random.randint(0, pool_size - 1))]]

        # For evaluation
        elif num_clients == -1:
            with open(
                    "./output/fit_server/round_{}.txt".format(server_round),
                    mode='r'
            ) as inputFile:
                cids_in_fit = eval(inputFile.readline())["clients_selected"]
            return [self.clients[str(cid)] for cid in cids_in_fit]

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


class QCS_ClientManager(SimpleClientManager):
    def sample(self, num_clients: int, server_round=0, time_constr=500):
        # For model initialization
        if num_clients == 1:
            # return [self.clients[str(random.randint(0, pool_size - 1))]]
            return [self.clients["0"]]

        # For evaluation
        elif num_clients == -1:
            with open(
                    "./output/fit_server/round_{}.txt".format(server_round),
                    mode='r'
            ) as inputFile:
                cids_in_fit = eval(inputFile.readline())["clients_selected"]
            return [self.clients[str(cid)] for cid in cids_in_fit]

        # Sample clients which meet the criterion
        param_dicts = []
        available_cids = []
        total_mu = 0
        P = dict()
        for _ in range(pool_size + 1):
            P[_] = []

        # Get info of previous round
        loss_of_prev_round = []
        if server_round == 1:
            init_parameters = self.clients["0"].get_parameters(
                ins=GetParametersIns(config={}),
                timeout=None
            ).parameters

            init_param_ndarrays = parameters_to_ndarrays(init_parameters)

            init_eval_func = clt.get_evaluate_fn(
                torchvision.datasets.CIFAR10(
                    root="./data", train=False, transform=cifar10Transformation()
                )
            )

            eval_res = init_eval_func(0, init_param_ndarrays, {})
            L_prev = eval_res[0]

        else:
            with open("./output/fit_server/round_{}.txt".format(server_round - 1)) \
                    as inputFile:
                cids_in_prev_round = eval(inputFile.readline())["clients_selected"]

            valid_loss_sum, valid_n_num = 0.0, 0
            for n in range(pool_size):

                with open("./output/train_loss/client_{}.txt".format(n)) as inputFile:
                    loss_of_prev_round.append(eval(inputFile.readlines()[-1]))

                if n in cids_in_prev_round:
                    assert loss_of_prev_round[-1] > 0
                    valid_loss_sum += loss_of_prev_round[-1]
                    valid_n_num += 1
                else:
                    assert loss_of_prev_round[-1] == -1

            L_prev = valid_loss_sum / valid_n_num

        with open(
                "./output/loss_avg/L_{}.txt".format(server_round - 1), mode='w'
        ) as outputFile:
            outputFile.write(str(L_prev))

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

            # Calculate \mu
            if server_round == 1:
                param_dicts[n]["mu"] = param_dicts[n]["dataSize"]
            else:
                with open("./output/loss_avg/L_{}.txt".format(server_round - 2)) \
                        as inputFile:
                    L_2 = eval(inputFile.readline())
                if n in cids_in_prev_round:
                    assert loss_of_prev_round[n] > 0
                    param_dicts[n]["mu"] = \
                        (L_2 - loss_of_prev_round[n]) * param_dicts[n]["dataSize"]
                else:
                    assert loss_of_prev_round[n] == -1
                    param_dicts[n]["mu"] = \
                        (L_2 - L_prev) * param_dicts[n]["dataSize"]

            param_dicts[n]["isSelected"] = False

        # QCS algorithm
        for j in range(1, pool_size + 1):
            S = []  # all clients that tolerate j peers
            K_j = []  # available cids under j-peer condition
            Q_j = 0  # total \mu under j-peer condition

            # you are available since you tolerate more than j peers
            for m in range(j, pool_size + 1):
                S += P[m]

            # select the best j clients
            l = j
            while l > 0:
                l -= 1

                # S is empty
                if not S:
                    continue

                # argmax
                max_mu = float("-inf")
                index_with_max_mu = -1
                for s in S:
                    if param_dicts[s]["mu"] > max_mu:
                        max_mu = param_dicts[s]["mu"]
                        index_with_max_mu = s

                K_j.append(index_with_max_mu)
                Q_j += param_dicts[index_with_max_mu]["mu"]
                S.remove(index_with_max_mu)

            if Q_j > total_mu:
                total_mu = Q_j
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
                "total_mu": total_mu,
                "time_elapsed": fit_round_time,
                "time_constraint": time_constr
            }, \
            param_dicts


class Random_ClientManager(SimpleClientManager):
    def sample(self, num_clients: int, server_round=0, time_constr=500):
        # For model initialization
        if num_clients == 1:
            return [self.clients[str(random.randint(0, pool_size - 1))]]

        # For evaluation
        elif num_clients == -1:
            with open(
                    "./output/fit_server/round_{}.txt".format(server_round),
                    mode='r'
            ) as inputFile:
                cids_in_fit = eval(inputFile.readline())["clients_selected"]
            return [self.clients[str(cid)] for cid in cids_in_fit]

        # Sample clients in a random way
        param_dicts = []

        with open(
                "./output/fit_server/round_{}.txt".format(server_round),
                mode='r'
        ) as inputFile:
            cid_num = len(eval(inputFile.readline())["clients_selected"])

        cids_tbd = list(range(pool_size))
        for _ in range(pool_size - cid_num):
            pop_idx = random.randint(0, len(cids_tbd) - 1)
            cids_tbd.pop(pop_idx)

        available_cids = cids_tbd.copy()
        assert len(available_cids) == cid_num

        for n in range(pool_size):
            # Get each client's parameters
            param_dicts.append(
                self.clients[str(n)].get_properties(
                    flwr.common.GetPropertiesIns(config={}), 68400
                ).properties.copy()
            )

            # Record client parameters
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
                "time_elapsed": fit_round_time,
                "time_constraint": time_constr
            }, \
            param_dicts


class Full_ClientManager(SimpleClientManager):
    def sample(self, num_clients: int, server_round=0, time_constr=500):
        # For model initialization
        if num_clients == 1:
            return [self.clients[str(random.randint(0, pool_size - 1))]]

        # All clients are selected
        param_dicts = []
        available_cids = list(range(pool_size))

        for n in range(pool_size):
            # Get each client's parameters
            param_dicts.append(
                self.clients[str(n)].get_properties(
                    flwr.common.GetPropertiesIns(config={}), 68400
                ).properties.copy()
            )

            # Record client parameters
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
                "time_elapsed": fit_round_time,
                "time_constraint": time_constr
            }, \
            param_dicts


class TCS_QCS(Strategy):
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
            client_manager: Union[
                TCS_ClientManager, QCS_ClientManager,
                Random_ClientManager, Full_ClientManager
            ]
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit_clients config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Check records of previous round
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
                    with open(
                            "./output/val_accu/client_{}.txt".format(_),
                            mode='a'
                    ) as outputFile:
                        outputFile.write("-1" + "\n")
                    with open(
                            "./output/val_loss/client_{}.txt".format(_),
                            mode='a'
                    ) as outputFile:
                        outputFile.write("-1" + "\n")

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
            num_clients=0, server_round=server_round, time_constr=C_T_i
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

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
            self, server_round: int, parameters: Parameters,
            client_manager: Union[
                TCS_ClientManager, QCS_ClientManager,
                Random_ClientManager, Full_ClientManager
            ]
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients: use same clients as in fit
        clients = client_manager.sample(
            num_clients=-1, server_round=server_round, time_constr=timeConstrGlobal
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
