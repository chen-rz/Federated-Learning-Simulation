import argparse
import logging
import os
import shutil
import time
from collections import OrderedDict
from pathlib import Path

import flwr as fl
import numpy as np
import ray
import torch
import torch.utils.data
import torchvision
from flwr.common.logger import log

from TCS import *
from dataset_utils import get_dataloader, cifar10Transformation
from utils import Net, train, test

parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")
parser.add_argument("--num_client_cpus", type=int, default=1)


# Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid: str, fed_dir_data: str, param_dict: dict):
        self.cid = cid
        self.fed_dir = Path(fed_dir_data)
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}

        # Instantiate model
        self.net = Net()

        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Random initialized parameters
        self.properties["cyclePerBit"] = param_dict["cyclePerBit"]
        self.properties["dataSize"] = param_dict["dataSize"]
        self.properties["frequency"] = param_dict["frequency"]
        self.properties["transPower"] = param_dict["transPower"]
        self.properties["updateTime"] = \
            num_rounds * self.properties["cyclePerBit"] * self.properties["dataSize"] \
            / self.properties["frequency"]

    def get_properties(self, config) -> Dict[str, Scalar]:
        return self.properties

    def get_parameters(self, config):
        return get_params(self.net)

    def fit(self, parameters, config):
        # Start timer
        start_time = time.perf_counter()
        log(logging.INFO, "Client %s starts training", self.cid)

        # Set model weights
        set_params(self.net, parameters)

        # Load data for this client and get trainloader
        num_workers = int(ray.get_runtime_context().get_assigned_resources()["CPU"])
        trainloader = get_dataloader(
            self.fed_dir,
            self.cid,
            is_train=True,
            batch_size=config["batch_size"],
            workers=num_workers,
        )

        # Send model to device
        self.net.to(self.device)

        # Train
        train_loss = train(
            self.net, trainloader, epochs=config["epochs"], device=self.device
        )

        # Record loss
        with open(
                "./output/train_loss/client_{}.txt".format(self.cid),
                mode='a'
        ) as outputFile:
            outputFile.write(str(train_loss) + "\n")

        # End timer
        end_time = time.perf_counter()
        log(
            logging.INFO,
            "Client %s ends training after %f sec, train loss: %f",
            self.cid, end_time - start_time, train_loss
        )

        # Return local model and statistics
        return get_params(self.net), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_params(self.net, parameters)

        # Load data for this client and get trainloader
        num_workers = int(ray.get_runtime_context().get_assigned_resources()["CPU"])
        valloader = get_dataloader(
            self.fed_dir, self.cid, is_train=False, batch_size=50, workers=num_workers
        )

        # Send model to device
        self.net.to(self.device)

        # Evaluate
        loss, accuracy = test(self.net, valloader, device=self.device)

        # Record loss and accuracy
        with open(
                "./output/val_loss/client_{}.txt".format(self.cid),
                mode='a'
        ) as outputFile:
            outputFile.write(str(loss) + "\n")
        with open(
                "./output/val_accu/client_{}.txt".format(self.cid),
                mode='a'
        ) as outputFile:
            outputFile.write(str(accuracy) + "\n")

        # Return statistics
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}


def fit_config(server_round: int) -> Dict[str, Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    _ = server_round  # placeholder
    config = {
        "epochs": 5,  # number of local epochs
        "batch_size": 64,
    }
    return config


def get_params(model) -> List[np.ndarray]:
    """Get model weights as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_params(model, params: List[np.ndarray]):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def get_evaluate_fn(test_set: torchvision.datasets.CIFAR10, ):
    """Return an evaluation function for centralized evaluation."""

    def evaluate(
            server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
    ):
        """Use the entire CIFAR-10 test set for evaluation."""

        # placeholder
        _ = server_round
        __ = config

        # determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = Net()
        set_params(model, parameters)
        model.to(device)

        testloader = torch.utils.data.DataLoader(test_set, batch_size=50)
        loss, accuracy = test(model, testloader, device=device)

        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate


# Start simulation (a _default server_ will be created)
if __name__ == "__main__":
    # parse input arguments
    args = parser.parse_args()

    fed_dir = "./data/cifar-10-batches-py/federated/"
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, transform=cifar10Transformation()
    )

    if Path("output/fit_clients/").exists():
        shutil.rmtree("output/fit_clients/")
    os.mkdir("output/fit_clients/")
    if Path("output/fit_server/").exists():
        shutil.rmtree("output/fit_server/")
    os.mkdir("output/fit_server/")
    if Path("output/train_loss/").exists():
        shutil.rmtree("output/train_loss/")
    os.mkdir("output/train_loss/")
    if Path("output/val_accu/").exists():
        shutil.rmtree("output/val_accu/")
    os.mkdir("output/val_accu/")
    if Path("output/val_loss/").exists():
        shutil.rmtree("output/val_loss/")
    os.mkdir("output/val_loss/")

    client_resources = {
        "num_cpus": args.num_client_cpus
    }  # each client will get allocated 1 CPU

    parameter_dict_list = []
    for _ in range(pool_size):
        parameter_dict_list.append(dict())
    with open("./parameters/cyclePerBit.txt") as inputFile:
        for _ in range(pool_size):
            parameter_dict_list[_]["cyclePerBit"] = eval(inputFile.readline())
    with open("./parameters/dataSize.txt") as inputFile:
        for _ in range(pool_size):
            parameter_dict_list[_]["dataSize"] = eval(inputFile.readline())
    with open("./parameters/frequency.txt") as inputFile:
        for _ in range(pool_size):
            parameter_dict_list[_]["frequency"] = eval(inputFile.readline())
    with open("./parameters/transPower.txt") as inputFile:
        for _ in range(pool_size):
            parameter_dict_list[_]["transPower"] = eval(inputFile.readline())


    def client_fn(cid: str):
        # create a single client instance
        return FlowerClient(cid, fed_dir, parameter_dict_list[int(cid)])


    # (optional) specify Ray config
    ray_init_args = {
        "include_dashboard": True,
        "log_to_driver": True
    }

    # Configure the strategy
    strategy = TCS(
        on_fit_config_fn=fit_config,
        evaluate_fn=get_evaluate_fn(testset),  # centralised evaluation of global model
    )

    # start simulation
    simulation = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=pool_size,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_manager=TCS_ClientManager(),
        ray_init_args=ray_init_args,
    )

    print(simulation)

    # Check records of last round
    with open(
            "./output/fit_server/round_{}.txt".format(num_rounds),
            mode='r'
    ) as last_inputFile:
        clients_of_last_round = eval(last_inputFile.readline())["clients_selected"]

    for _ in range(pool_size):
        # If the client was not selected in the last round,
        # help it complete the records
        if _ not in clients_of_last_round:
            with open(
                    "./output/train_loss/client_{}.txt".format(_),
                    mode='a'
            ) as last_outputFile:
                last_outputFile.write("-1" + "\n")
