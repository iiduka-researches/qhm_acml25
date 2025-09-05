'''Train CIFAR10 or CIFAR100 with PyTorch.'''
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

from data.get_dataset import get_dataset
from scheduler.lr import get_lr_scheduler
from scheduler.beta import get_beta_scheduler
from scheduler.gamma import get_gamma_scheduler
from scheduler.batch import get_batch_scheduler
from optimizer.get_optimizer import get_optimizer
from models.get_model import get_model
from utils import progress_bar

steps = 0
def train(epoch: int) -> None:
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    global steps
    for batch_idx, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        steps += 1
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    training_acc = 100.*correct/total
    wandb.log({'training_acc': training_acc,
               'training_loss': train_loss/(batch_idx+1)})

max_test_acc = 0.0

def test() -> None:
    global max_test_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
            
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total
    if acc > max_test_acc:
        max_test_acc = acc

    wandb.log({'accuracy': acc,
               'max_test_accuracy': max_test_acc,
               'test_loss': test_loss / len(testloader)}) 

min_total_grad_norm = float('inf')

def calc_full_grad(net: nn.Module, train_set: torch.utils.data.Dataset, batch_size: int=1000) -> None:
    global min_total_grad_norm
    parameters = [p for p in net.parameters()]
    device = 'cuda:0'
    full_grad_params = []
    for p in parameters:
        full_grad_params.append(torch.zeros_like(p, device=device))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    N = len(train_set)

    for xx, yy in train_loader:
        xx = xx.to(device, non_blocking=True)
        yy = yy.to(device, non_blocking=True)
        net.zero_grad()
        loss = nn.CrossEntropyLoss(reduction='mean')(net(xx), yy)
        loss.backward()
        for fg, p in zip(full_grad_params, parameters):
            if p.grad is not None:
                fg.add_(p.grad, xx.shape[0] / N)
    
    sqsum  = torch.tensor(0.0, device=device)
    for fg in full_grad_params:
        sqsum += (fg**2).sum()
    total_grad_norm = sqsum.sqrt().item()

    if total_grad_norm < min_total_grad_norm:
        min_total_grad_norm = total_grad_norm
    wandb.log({'total_grad_norm': total_grad_norm,
               'min_total_grad_norm': min_total_grad_norm})

def schedulers_step(is_lr :bool=False, is_beta :bool=False, is_gamma :bool=False, is_batch=False) -> DataLoader:
    if is_lr:
        last_lr = lr_scheduler.get_last_lr()[0]
        lr_scheduler.step()
        wandb.log({'last_lr': last_lr})
    if is_beta:
        last_beta = beta_sheduler.step()
        wandb.log({'last_beta': last_beta})
    if is_gamma:
        last_gamma = gamma_sheduler.step()
        wandb.log({'last_gamma': last_gamma})
    if is_batch:
        last_batch, trainloader = batch_scheduler.step()
        wandb.log({'last_batch': last_batch})
    return trainloader

def schedulers_step_for_param_tracking(is_lr :bool=False, is_beta :bool=False, is_gamma :bool=False, is_batch=False) -> DataLoader:
    if is_lr:
        last_lr = lr_scheduler.get_last_lr()[0]
        lr_scheduler.step()
        wandb.log({'last_lr': last_lr})
    if is_beta:
        last_beta = beta_sheduler.step()
        wandb.log({'last_beta': last_beta})
    if is_gamma:
        last_gamma = gamma_sheduler.step()
        wandb.log({'last_gamma': last_gamma})
    if is_batch:
        last_batch, trainloader = batch_scheduler.step()
        wandb.log({'last_batch': last_batch})
    return trainloader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 or CIFAR100 Training')
    parser.add_argument('--dataset', default="CIFAR10", type=str, help="[CIFAR100, CIFAR10]")
    parser.add_argument('--model', default="ResNet18", type=str, help="[ResNet18, WideResNet-28-10, MobileNetv2]")
    parser.add_argument('--optimizer', default="qhm", type=str, help="[sgd, shb, nshb, qhm, sgd, adam, adamw, rmsprop]")
    parser.add_argument('--epochs', default=200, type=int, help="the number of epochs")
    parser.add_argument('--lr', default=0.1, type=float, help='the initial learning rate')
    parser.add_argument('--batch_size', default=8, type=int, help='the initial training batch size')
    parser.add_argument('--beta', default=0.9, type=float, help="the initial decaying rate of d_k (β_k)")
    parser.add_argument('--gamma', default=0.7, type=float, help="the initial decaying rate of m_k (γ_k)")
    parser.add_argument('--lr_sche_name', default="constant", type=str, help="[constant, decay, decay_squared, cosine, poly]")
    parser.add_argument('--beta_sche_name', default="constant", type=str, help="[constant, decay, step_decay, increase]")
    parser.add_argument('--gamma_sche_name', default="constant", type=str, help="[constant, decay, step_decay]")
    parser.add_argument('--batch_sche_name', default="constant", type=str, help="[constant, exp]")
    parser.add_argument('--step_size', default=20, type=int, help="the number of epoch to increase or decrease when using a decay or increase scheduler")

    args = parser.parse_args()
    wandb_project_name = "ooooo"
    wandb_exp_name = "???"
    wandb.init(config = args,
               project = wandb_project_name,
               name = wandb_exp_name,
               entity = "xxxxx",
               settings=wandb.Settings(start_method='fork'))
    
    trainset, trainloader, testset, testloader = get_dataset(args.dataset, args.batch_size)
    device = 'cuda:0'
    criterion = nn.CrossEntropyLoss()

    net = get_model(args.model, device)
    optimizer = get_optimizer(args.optimizer, net, args.lr, args.beta, args.gamma)
    lr_scheduler = get_lr_scheduler(args.lr_sche_name, optimizer, args.epochs)
    if args.optimizer in ["shb", "nshb", "qhm"]:
        beta_sheduler = get_beta_scheduler(args.beta_sche_name, optimizer, args.step_size)
    if args.optimizer == "qhm":
        gamma_sheduler = get_gamma_scheduler(args.gamma_sche_name, optimizer, args.step_size)
    batch_scheduler = get_batch_scheduler(args.batch_sche_name, trainset, args.batch_size, shuffle=True, num_workers=2, step_size=args.step_size)
    for epoch in range(args.epochs):
        train(epoch)
        test()
        calc_full_grad(net, trainset)
        trainloader = schedulers_step(is_lr=True, is_beta=(args.optimizer in ["shb", "nshb", "qhm"]), is_gamma=(args.optimizer == "qhm"), is_batch=True)
