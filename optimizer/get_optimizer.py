import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from .shb import SHB
from .qhm import QHM

def get_optimizer(op_name: str, net: torch.nn.Module, lr: float, beta: float, gamma: float) -> Optimizer:
    if op_name == "shb":
        optimizer = SHB(net.parameters(), lr=lr, beta=beta)
    elif op_name == "nshb":
        optimizer = QHM(net.parameters(), lr=lr, beta=beta, gamma=1)
    elif op_name == "qhm":
        optimizer = QHM(net.parameters(), lr=lr, beta=beta, gamma=gamma)
    elif op_name == "adam":
        optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
    elif op_name == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=lr)
    elif op_name == "rmsprop":
        optimizer = optim.RMSprop(net.parameters())      
    elif op_name == "adamw":
        optimizer = optim.AdamW(net.parameters())
    else:
        raise ValueError(f"Invalid optimizer name '{op_name}'. Available options are: 'shb', 'nshb', 'qhm', 'adam', 'sgd', 'rmsprop', 'adamw'.")
    print(optimizer)
    return optimizer