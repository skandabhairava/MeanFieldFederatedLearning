import torch
from client_types import ClientTypes

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLIENTS = 30
# CLIENT_FRAC = 0.5
CLIENT_FRAC = 1

LOCAL_EPOCHS = 5
BATCH_SIZE = 32
LR = 1e-4

ROUNDS = 30

DIRICHLET_ALPHA = 0.5
# 0 -> HIGH NON-IID, 100 -> IID

LOG_DIR = "./logs"
WRITE_LOGS = False

# parallel
NUM_CPUS = 8
NUM_GPUS = 1

# client typing
CLIENT_TYPES = {
    # ClientTypes.NORMAL: 1.0,

    ClientTypes.NORMAL: 0.9,
    ClientTypes.ALIE: 0.1
}

RANDOM_SEED = 60123