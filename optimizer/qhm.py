import torch
from torch.optim import Optimizer
from typing import List, Union, Dict, Any

class QHM(Optimizer):
    def __init__(self, params: Union[List[torch.Tensor], List[Dict[str, Any]]], lr: float, beta: float=0, gamma: float=0, weight_decay: float=0) -> None:
        """
        params: parameters of model
        lr: learning rate(α_k)
        beta: decaying rate of d_k (β_k)
        gamma: decaying rate of m_k (γ_k)
        weight_decay: coefficient of weight decay (λ)
        """
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not (0.0 <= beta < 1.0):
            raise ValueError("Invalid beta value: {}".format(beta))
        if not (0.0 <= gamma <= 1.0):
            raise ValueError("Invalid gamma value: {}".format(gamma))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, beta=beta, gamma=gamma, weight_decay=weight_decay)
        super(QHM, self).__init__(params, defaults)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        super(QHM, self).__setstate__(state)

    def step(self) -> None:
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            beta = group['beta']
            gamma = group['gamma']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if 'd' not in state:
                    state['d'] = torch.zeros_like(p.data)
                if 'm' not in state:
                    state['m'] = torch.zeros_like(p.data)
                
                grad = p.grad.data
                if weight_decay != 0:
                    grad.add_(weight_decay, p.data)
                
                # update of d_k: d_k = (1 - β_k) * grad + β * d_{k-1}
                d = state['d']
                d.mul_(beta).add_(grad, alpha=(1-beta))
                
                # update of m_k: m_k = (1 - γ_k) * grad + γ_k * d_k
                m = state['m']
                new_m = (1-gamma) * grad + gamma * d
                m.copy_(new_m)
                
                # update of parameter: x_{k+1} = x_k - α_k * m_k
                p.data.add_(m, alpha=-lr)

                state['d'] = d
                state['m'] = m