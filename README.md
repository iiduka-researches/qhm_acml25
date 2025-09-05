# Both Asymptotic and Non-Asymptotic Convergence of Quasi-Hyperbolic Momentum using Increasing Batch Size
Code for reproducing experiments in our paper.
Our experiments were based on the basic code for image classification.

## Abstract
Momentum methods were originally introduced for their superiority to stochastic gradient descent (SGD) in deterministic settings with convex objective functions. However, despite their widespread application to deep neural networks — a representative case of stochastic nonconvex optimization — the theoretical justification for their effectiveness in such settings remains limited. Quasi-hyperbolic momentum (QHM) is an algorithm that generalizes various momentum methods and has been studied to better understand the class of momentum-based algorithms as a whole. In this paper, we provide both asymptotic and non-asymptotic convergence results for mini-batch QHM with an increasing batch size. We show that achieving asymptotic convergence requires either a decaying learning rate or an increasing batch size. Since a decaying learning rate adversely affects non-asymptotic convergence, we demonstrate that using mini-batch QHM with an increasing batch size — without decaying the learning rate — can be a more effective strategy. Our experiments show that even a finite increase in batch size can provide benefits for training neural networks. The code is available at https://anonymous.4open.science/r/qhm_public.

## WandB Setup
Please replace `ooooo`(project name), `???`(experiment name), and `XXXXXX`(entity name) with your actual WandB project, experiment, and entitiy names.

```
wandb_project_name = "ooooo"
wandb_exp_name = "???"
wandb.init(config = args,
           project = wandb_project_name,
           name = wandb_exp_name,
           entity = "xxxxx",
           settings=wandb.Settings(start_method='fork'))
```

## Usage
You can select the hyperparameter(learning rate, momentum weights, and batch size) scheduler as follows.
```
python3 main.py --lr_sche_name="constant" --beta_sche_name="step_decay" --gamma_sche_name="step_decay" --batch_sche_name="exp"
```
