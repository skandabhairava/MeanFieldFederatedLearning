import torch
from client_types import ClientTypes

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_VELOCITY = False

NUM_CLIENTS = 30
# CLIENT_FRAC = 0.5
CLIENT_FRAC = 0.5

LOCAL_EPOCHS = 5
BATCH_SIZE = 32
LR = 1e-4

ROUNDS = 30
ATTRACT_SPLIT_EVERY = 5

DIRICHLET_ALPHA = 1
# 0 -> HIGH NON-IID, 100 -> IID

LOG_DIR = "./logs"
WRITE_LOGS = True

# parallel
NUM_CPUS = 8
NUM_GPUS = 1

# client typing
CLIENT_TYPES = {
    ClientTypes.NORMAL: 1.0,

    # ClientTypes.NORMAL: 0.4,
    # ClientTypes.SUBTLE: 0.6,

    # ClientTypes.NORMAL: 0.4,
    # ClientTypes.BACKDOOR_0: 0.6,

    # ClientTypes.NORMAL: 0.4,
    # ClientTypes.LABEL_SWITCH: 0.6,
}

RANDOM_SEED = 60123