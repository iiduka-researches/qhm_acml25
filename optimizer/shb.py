import torch
from torch.optim import Optimizer
from typing import List, Union

class SHB(Optimizer):
  def __init__(self, params: Union[List[torch.Tensor]], lr: float, beta: float=0, weight_decay:float=0) -> None:
    if lr < 0.0:
      raise ValueError("Invalid learning rate: {}".format(lr))
    if not 0.0 <= beta <= 1.0:
      raise ValueError("Invalid momentum value: {}".format(beta))
    if weight_decay < 0.0:
      raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

    defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay)
    super(SHB, self).__init__(params, defaults)

  def __setstate__(self, state) -> None:
    super(SHB, self).__setstate__(state)

  def step(self) -> None:
    for group in self.param_groups:
      weight_decay = group['weight_decay']
      beta = group['beta']
      lr = group['lr']

    for p in group['params']:
      if p.grad is None:
          continue
      state = self.state[p]
      if 'm' not in state:
          state['m'] = torch.zeros_like(p.data)
      
      grad = p.grad.data
      if weight_decay != 0:
          grad.add_(weight_decay, p.data)
      
      # update of m_k: m_k =  grad + β_k * m_{k-1}
      m = state['m']
      m.mul_(beta).add_(grad, alpha=1)
      
      # update of parameter: x_{k+1} = x_k - α_k * m_k
      p.data.add_(m, alpha=-lr)
      state['m'] = m