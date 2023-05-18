import os.path
import random

from constants import *
from dataset_utils import get_cifar_10, do_fl_partitioning

# Download CIFAR-10 dataset
train_path, testset = get_cifar_10()

# Partition dataset
# Use a large `alpha` (e.g. 1000) to make it IID;
#   a small value (e.g. 1) will make it non-IID
# This will create a new directory called "federated" in the directory where CIFAR-10 lives.
# Inside it, there will be N=pool_size subdirectories each with its own train/set split.
fed_dir = do_fl_partitioning(
    train_path, pool_size=pool_size, alpha=1, num_classes=10, val_ratio=0.1
)

# Record dataset sizes (bit)
with open("./parameters/dataSize.txt", mode='w') as outputFile:
    for n in range(pool_size):
        outputFile.write(
            str(
                8 * os.path.getsize(
                    "./data/cifar-10-batches-py/federated/" + str(n) + "/train.pt"
                )
            )
            + "\n"
        )
print("Dataset initialization completed")

# Define CPU/GPU frequency
with open("./parameters/frequency.txt", mode='w') as outputFile:
    for n in range(pool_size):
        outputFile.write(str(random.uniform(100e6, 2e9)) + "\n")
print("CPU/GPU frequency initialization completed")

# Define cycles per bit
with open("./parameters/cyclePerBit.txt", mode='w') as outputFile:
    for n in range(pool_size):
        outputFile.write(str(random.uniform(50, 200)) + "\n")
print("Cycles per bit initialization completed")

# Define transmission power
with open("./parameters/transPower.txt", mode='w') as outputFile:
    for n in range(pool_size):
        outputFile.write(str(random.uniform(5e-4, 5e-3)) + "\n")
print("Transmission power initialization completed")
