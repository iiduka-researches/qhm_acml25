import torch.optim as optim
from typing import List, Callable
from abc import ABC, abstractmethod
from .func import decay_func_3

class BetaScheduler(ABC):
    """
    BetaScheduler is an abstract class for updating the momentum optimizer's beta value。
    """
    def __init__(self, optimizer: optim.Optimizer, last_epoch: int = -1) -> None:
        """
        optimizer (optim.Optimizer): An optimizer with a beta parameter
        last_epoch (int): The current epoch number
        """
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.base_betas = [group['beta'] for group in optimizer.param_groups]

    @abstractmethod
    def get_beta(self) -> List[float]:
        pass

    def step(self) -> float:
        self.last_epoch += 1
        new_betas = self.get_beta()
        for param_group, beta in zip(self.optimizer.param_groups, new_betas):
            param_group['beta'] = beta
        return new_betas[0]
    
    def state_dict(self) -> dict:
        return {
			'last_epoch': self.last_epoch,
			'base_betas': self.base_betas,
		}

    def load_state_dict(self, state_dict: dict) -> None:
        self.last_epoch = state_dict['last_epoch']
        self.base_betas = state_dict['base_betas']

class BetaConstantScheduler(BetaScheduler):
    def __init__(self, optimizer: optim.Optimizer, last_epoch: int = -1) -> None:
        super().__init__(optimizer, last_epoch)
        self.constant_beta = self.base_betas[0]

    def get_beta(self) -> List[float]:
        return [self.constant_beta for _ in self.optimizer.param_groups]

class BetaStepDecayScheduler(BetaScheduler):
    def __init__(self, optimizer: optim.Optimizer, step_size: int, eta: float= 0.1, last_epoch: int = -1) -> None:
        self.step_size = step_size
        self.eta = eta
        super().__init__(optimizer, last_epoch)
    
    def get_beta(self) -> List[float]:
        if self.last_epoch == 0 or self.last_epoch % self.step_size != 0:
            return [group['beta'] for group in self.optimizer.param_groups]
        return [group['beta'] * self.eta for group in self.optimizer.param_groups]

class BetaIncreaseScheduler(BetaScheduler):
    def __init__(self, optimizer: optim.Optimizer, beta_lambda: Callable[[int], float], step_size: int, last_epoch: int = -1) -> None:
        """
        beta_lambda (Callable[[int], float]): A function that takes an epoch (int) as input and returns a float
        step_size (int): interval of epochs after which to increment internal counter
        """
        self.beta_lambda = beta_lambda
        self.step_size = step_size
        self.m_last_epoch = 0
        super().__init__(optimizer, last_epoch)

        if not isinstance(beta_lambda, (list, tuple)):
            self.beta_lambdas = [beta_lambda] * len(optimizer.param_groups)
        else:
            if len(beta_lambda) != len(optimizer.param_groups):
                raise ValueError(f"Expected {len(optimizer.param_groups)} beta_lambdas, got {len(beta_lambda)}")
            self.beta_lambdas = list(beta_lambda)
        
        super().__init__(optimizer, last_epoch)

    def get_beta(self) -> List[float]:
        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return [1 - (1 - base_beta) * lmbda(self.m_last_epoch) for lmbda, base_beta in zip(self.beta_lambdas, self.base_betas)]
        else:
            self.m_last_epoch += 1
            return [1 - (1 - base_beta) * lmbda(self.m_last_epoch) for lmbda, base_beta in zip(self.beta_lambdas, self.base_betas)]
    
    def state_dict(self) -> dict:
        state = super().state_dict()
        state.update({
			'm_last_epoch': self.m_last_epoch,
			'step_size': self.step_size,
		})
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self.m_last_epoch = state_dict['m_last_epoch']
        self.step_size = state_dict['step_size']

def get_beta_scheduler(sche_name: str, optimizer: optim.Optimizer, step_size: int=None)-> BetaScheduler:
    """
    sche_name (str): scheduler_name we use: {constant, step_decay, increase}
    optimizer (optim.Optimizer): optimizer we use: {qhm, shb, nshb, sgd, adam, adamw, rmsprop}
    step_size (int): interval of epochs after which to increment internal counter
    """
    if sche_name == "constant":
        beta_scheduler = BetaConstantScheduler(optimizer)
    elif sche_name == "step_decay":
        beta_scheduler = BetaStepDecayScheduler(optimizer, step_size=step_size, eta=0.5)
    elif sche_name == "increase":
        beta_scheduler = BetaIncreaseScheduler(optimizer, beta_lambda=decay_func_3, step_size=step_size)
    else:
        raise ValueError(f"Invalid scheduler name '{sche_name}'. Available options are: 'constant', 'step_decay', 'increase'.")
    return beta_scheduler