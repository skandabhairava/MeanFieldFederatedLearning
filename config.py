import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLIENTS = 20
# CLIENT_FRAC = 0.5
CLIENT_FRAC = 1

LOCAL_EPOCHS = 1
BATCH_SIZE = 32
LR = 0.01

ROUNDS = 30

DIRICHLET_ALPHA = 0.5

LOG_DIR = "./logs"

# parallel
NUM_CPUS = 8
NUM_GPUS = 1

# client typing
CLIENT_TYPES = {
    "normal": 0.8,
    "byzantine_flip": 0.2,
}

RANDOM_SEED = 60123