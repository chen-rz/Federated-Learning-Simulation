import argparse
import os
import shutil
from pathlib import Path

import flwr as fl
import flwr.server.strategy

from strategy import *

parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")
parser.add_argument("--num_client_cpus", type=int, default=1)
parser.add_argument("--mode", type=str, default="TCS")

# Start simulation (a _default server_ will be created)
if __name__ == "__main__":
    # parse input arguments
    args = parser.parse_args()

    fed_dir = "./data/cifar-10-batches-py/federated/"
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, transform=cifar10Transformation()
    )

    path_to_init = ["fit_clients", "fit_server", "loss_avg",
                    "train_loss", "val_accu", "val_loss"]
    for _ in path_to_init:
        if Path("output/" + _ + "/").exists():
            shutil.rmtree("output/" + _ + "/")
        os.mkdir("output/" + _ + "/")

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
        return clt.FlowerClient(cid, fed_dir, parameter_dict_list[int(cid)])


    # (optional) specify Ray config
    ray_init_args = {
        "include_dashboard": True,
        "log_to_driver": True
    }

    # Configure the strategy
    if args.mode in ["TCS", "QCS"]:
        strategy = TCS_QCS(
            on_fit_config_fn=clt.fit_config,
            # centralised evaluation of global model
            evaluate_fn=clt.get_evaluate_fn(testset),
        )
    else:
        strategy = flwr.server.strategy.FedAvg(
            min_available_clients=pool_size,
            on_fit_config_fn=clt.fit_config,
            # centralised evaluation of global model
            evaluate_fn=clt.get_evaluate_fn(testset),
        )

    # Configure the client manager
    if args.mode == "TCS":
        client_manager = TCS_ClientManager()
    elif args.mode == "QCS":
        client_manager = QCS_ClientManager()
    else:
        client_manager = SimpleClientManager()

    # start simulation
    simulation = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=pool_size,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_manager=client_manager,
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
