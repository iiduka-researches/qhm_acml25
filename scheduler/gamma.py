import torch.optim as optim
from typing import List
from abc import ABC, abstractmethod


class GammaScheduler(ABC):
    """
    GammaScheduler is an abstract class for updating the momentum optimizer's gamma value.
    """
    def __init__(self, optimizer: optim.Optimizer, last_epoch: int = -1) -> None:
        """
        optimizer (optim.Optimizer): An optimizer with a gamma parameter.
        last_epoch (int): The current epoch number.
        """
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.base_gammas = [group['gamma'] for group in optimizer.param_groups]

    @abstractmethod
    def get_gamma(self) -> List[float]:
        pass

    def step(self) -> float:
        self.last_epoch += 1
        new_gammas = self.get_gamma()
        for param_group, gamma in zip(self.optimizer.param_groups, new_gammas):
            param_group['gamma'] = gamma
        return new_gammas[0]
    
    def state_dict(self) -> dict:
        return {
			'last_epoch': self.last_epoch,
			'base_gammas': self.base_gammas,
		}

    def load_state_dict(self, state_dict: dict) -> None:
        self.last_epoch = state_dict['last_epoch']
        self.base_betas = state_dict['base_gammas']

class GammaConstantScheduler(GammaScheduler):
    def __init__(self, optimizer: optim.Optimizer, last_epoch: int = -1) -> None:
        super().__init__(optimizer, last_epoch)
        self.constant_gamma = self.base_gammas[0]

    def get_gamma(self) -> List[float]:
        return [self.constant_gamma for _ in self.optimizer.param_groups]

class GammaStepDecayScheduler(GammaScheduler):
    def __init__(self, optimizer: optim.Optimizer, step_size: int, eta: float= 0.1, last_epoch: int = -1) -> None:
        """
        step_size (int): interval of epochs after which to increment internal counter.
        eta (int): the hyperparameter used in updating gamma(γ'_m = γ_0 * eta^{m}).
        """
        self.step_size = step_size
        self.eta = eta
        super().__init__(optimizer, last_epoch)
    
    def get_gamma(self) -> List[float]:
        if self.last_epoch == 0 or self.last_epoch % self.step_size != 0:
            return [group['gamma'] for group in self.optimizer.param_groups]
        return [group['gamma'] * self.eta for group in self.optimizer.param_groups]

def get_gamma_scheduler(sche_name: str, optimizer: optim.Optimizer, step_size: int = None) -> GammaScheduler:
    """
    sche_name (str): scheduler_name we use: {constant, step_decay}
    optimizer (optim.Optimizer): optimizer we use: {qhm, shb, nshb, sgd, adam, adamw, rmsprop}
    step_size (int): interval of epochs after which to increment internal counter.
    """
    if sche_name == "constant":
        gamma_scheduler = GammaConstantScheduler(optimizer)
    elif sche_name == "step_decay":
        gamma_scheduler = GammaStepDecayScheduler(optimizer, step_size=step_size, eta=0.5)
    else:
        raise ValueError(f"Invalid scheduler name '{sche_name}'. Available options are: 'constant', 'step_decay'.")
    return gamma_scheduler