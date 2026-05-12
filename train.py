import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import CIFAR10, CIFAR100, DATASET_NUM
from dataloader import DataAgentDataset
from lars import Lars
from model import ResNet18, ResNet34, ResNet50, ResNet101
from normalization import batch_Normalization
from PPO import PPO


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR training with PPO and Data Agent')

    # Training
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N', help='input batch size for testing')
    parser.add_argument('--num-epoch', default=200, type=int, help='number of epochs to train')
    parser.add_argument('--model', default='r18', type=str, help='model type (r18, r34, r50, r101)')
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar10', 'cifar100'], help='dataset to use')
    parser.add_argument('--dataset_path', type=str, default='./data', help='path to dataset')
    parser.add_argument('--gpus', type=str, default='0', help='GPU ids to use')
    parser.add_argument('--log-interval', type=int, default=100, help='how many batches to wait before logging')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--save-model', action='store_true', default=False, help='save the best checkpoint')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'], help='device to use')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='label smoothing for cross entropy')

    # Optimizer
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'lars', 'adam'], help='optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD/LARS momentum')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='W', help='weight decay')

    # OneCycle scheduler
    parser.add_argument('--max-lr', default=0.1, type=float, help='maximum learning rate for OneCycleLR')
    parser.add_argument('--div-factor', default=25, type=float, help='initial_lr = max_lr / div_factor')
    parser.add_argument('--final-div', default=10000, type=float, help='min_lr = initial_lr / final_div_factor')
    parser.add_argument('--pct-start', default=0.3, type=float, help='percentage of cycle spent increasing lr')

    # Data Agent sampling
    parser.add_argument('--ratio', default=0.5, type=float, help='Data Agent sampling ratio')
    parser.add_argument('--delta', default=0.875, type=float, help='Data Agent annealing parameter')

    # PPO
    parser.add_argument('--state-dim', type=int, default=2048, help='state dimension for PPO agent')
    parser.add_argument('--action-dim', type=int, default=1, help='action dimension for PPO agent')
    parser.add_argument('--use-orthogonal-init', action='store_true', default=True, help='use orthogonal initialization')
    parser.add_argument('--use-reward-norm', action='store_true', default=True, help='use reward normalization')
    parser.add_argument('--max_step', type=int, default=200, help='max steps for PPO learning-rate decay')
    parser.add_argument('--ppo-lr-a', type=float, default=3e-4, help='PPO actor learning rate')
    parser.add_argument('--ppo-lr-c', type=float, default=3e-4, help='PPO critic learning rate')
    parser.add_argument('--ppo-eps-clip', type=float, default=0.2, help='PPO clipping epsilon')
    parser.add_argument('--ppo-k-epochs', type=int, default=4, help='PPO update epochs per update')
    parser.add_argument('--ppo-gamma', type=float, default=0.99, help='PPO discount factor')
    parser.add_argument('--ppo-gae-lambda', type=float, default=0.95, help='PPO GAE lambda')
    parser.add_argument('--ppo-update-freq', type=int, default=10, help='update PPO every N batches')

    # Reward weighting
    parser.add_argument('--variance-adaptive-epsilon', type=float, default=1e-8,
                        help='epsilon for variance-adaptive reward weighting')
    return parser.parse_args()


args = parse_args()

if args.model.lower() in ['r18', 'r34']:
    args.state_dim = 512
elif args.model.lower() in ['r50', 'r101', 'r152']:
    args.state_dim = 2048
args.max_step = args.num_epoch

if 'CUDA_VISIBLE_DEVICES' in os.environ and os.environ['CUDA_VISIBLE_DEVICES']:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"==> Using CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
elif args.device == 'auto':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
elif args.device == 'cuda':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = 'cuda'
else:
    device = 'cpu'

print(f'==> Using device: {device}')

best_acc = 0
best_epoch = 0
start_epoch = 0

train_times = []
test_times = []
train_losses_history = []
ppo_actor_losses_history = []
ppo_critic_losses_history = []


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataset_stats(dataset_name):
    if dataset_name == 'cifar10':
        return (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    if dataset_name == 'cifar100':
        return (0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025)
    raise ValueError(f'Dataset {dataset_name} is not supported')


def build_datasets():
    stats = get_dataset_stats(args.dataset)
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])

    if args.dataset == 'cifar10':
        train_dataset = CIFAR10(root=args.dataset_path, train=True, cls_transform=train_transform, download=True)
        test_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_path, train=False, download=True, transform=test_transform)
        return train_dataset, test_dataset, 10

    train_dataset = CIFAR100(root=args.dataset_path, train=True, cls_transform=train_transform, download=True)
    test_dataset = torchvision.datasets.CIFAR100(
        root=args.dataset_path, train=False, download=True, transform=test_transform)
    return train_dataset, test_dataset, 100


def get_model(model_name, num_classes):
    model_name = model_name.lower()
    if model_name == 'r18':
        return ResNet18(num_classes=num_classes, return_features=True)
    if model_name == 'r34':
        return ResNet34(num_classes=num_classes, return_features=True)
    if model_name == 'r50':
        return ResNet50(num_classes=num_classes, return_features=True)
    if model_name == 'r101':
        return ResNet101(num_classes=num_classes, return_features=True)
    raise ValueError(f'Model {model_name} is not supported')


def build_optimizer(model):
    optimizer_name = args.optimizer.lower()
    if optimizer_name == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
    if optimizer_name == 'lars':
        return Lars(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    raise ValueError(f'Optimizer {args.optimizer} is not supported')


def build_scheduler(last_epoch=-1):
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        args.max_lr,
        steps_per_epoch=len(trainloader),
        epochs=args.num_epoch,
        div_factor=args.div_factor,
        final_div_factor=args.final_div,
        pct_start=args.pct_start,
        last_epoch=last_epoch,
    )


def compute_reward(outputs, losses, indices_device):
    losses_detached = losses.detach()
    loss_upper_bound = torch.quantile(losses_detached, 0.95)
    losses_clipped = torch.clamp(losses_detached, max=loss_upper_bound)

    min_loss = losses_clipped.min()
    max_loss = losses_clipped.max()
    normalized_losses = (losses_clipped - min_loss) / (max_loss - min_loss + 1e-8)

    with torch.no_grad():
        probs = F.softmax(outputs, dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
        max_entropy = torch.log(torch.tensor(outputs.size(1), dtype=torch.float32, device=outputs.device))
        confidence_reward = entropy / max_entropy

    loss_variance = torch.var(normalized_losses)
    confidence_variance = torch.var(confidence_reward)
    reward_weight = loss_variance / (loss_variance + confidence_variance + args.variance_adaptive_epsilon)
    reward_weight = torch.clamp(reward_weight, 0.0, 1.0)

    mask_values = MASK_PRUNER[indices_device].squeeze()
    reward = reward_weight * (normalized_losses * mask_values)
    reward = reward + (1 - reward_weight) * (confidence_reward * mask_values)

    if args.use_reward_norm:
        reward = reward_norm(reward)

    return reward, confidence_reward, reward_weight, loss_variance, confidence_variance


setup_seed(args.seed)

print('==> Preparing data...')
trainset_ada, testset, num_classes = build_datasets()
trainset = DataAgentDataset(trainset_ada, ratio=args.ratio, num_epoch=args.num_epoch, delta=args.delta)
trainset.dataset.is_magnitude = True

trainloader = DataLoader(
    trainset,
    batch_size=args.batch_size,
    sampler=trainset.pruning_sampler(),
    num_workers=16,
    pin_memory=True,
)
testloader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=16, pin_memory=True)

print('==> Building model and PPO agent...')
model = get_model(args.model, num_classes).to(device)
if device == 'cuda' and len(args.gpus.split(',')) > 1:
    model = nn.DataParallel(model).to(device)

agent = PPO(args)
optimizer = build_optimizer(model)
lr_scheduler = build_scheduler()

criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, reduction='none')
test_criterion = nn.CrossEntropyLoss()
reward_norm = batch_Normalization()
MASK_PRUNER = torch.ones(DATASET_NUM[args.dataset], 1, device=device)


def train(epoch):
    print(f'Epoch: {epoch}')
    model.train()

    train_loss = 0
    correct = 0
    total = 0
    start_time = time.time()

    ppo_step_count = 0
    total_ppo_actor_loss = 0
    total_ppo_critic_loss = 0
    ppo_update_count = 0

    for batch_idx, batch_data in enumerate(trainloader):
        if isinstance(batch_data, (list, tuple)) and len(batch_data) == 4:
            inputs, targets, indices, weights = batch_data
        else:
            inputs, targets, indices = batch_data[:3]
            weights = torch.ones_like(targets, dtype=torch.float32)

        inputs = inputs.to(device)
        targets = targets.to(device)
        weights = weights.to(device)
        indices_cpu = indices.long().cpu()
        indices_device = indices_cpu.to(device)

        outputs, feature = model(inputs)
        losses = criterion(outputs, targets)

        action = agent.action(feature)
        action_squeezed = action.squeeze().to(device)

        batch_size = indices_device.size(0)
        replace_index = (indices_device, torch.zeros(batch_size, dtype=torch.long, device=device))
        MASK_PRUNER.index_put_(replace_index, action_squeezed)
        trainset.__setscore__(indices_cpu.numpy(), action_squeezed.detach().cpu().numpy())

        loss_to_backward = (losses * weights).mean()
        reward, confidence_reward, reward_weight, loss_variance, confidence_variance = compute_reward(
            outputs, losses, indices_device)

        if batch_idx % (args.log_interval * 10) == 0:
            print(f'  Confidence Reward Stats - Avg: {confidence_reward.mean():.4f}, '
                  f'Min: {confidence_reward.min():.4f}, Max: {confidence_reward.max():.4f}')
            print(f'  Variance Adaptive Weights - r: {reward_weight:.4f}, Loss Var: {loss_variance:.4f}, '
                  f'Confidence Var: {confidence_variance:.4f}, Epsilon: {args.variance_adaptive_epsilon}')

        agent.store_reward(reward.detach().cpu().numpy())

        optimizer.zero_grad()
        loss_to_backward.backward()
        optimizer.step()
        lr_scheduler.step()

        ppo_step_count += 1
        if ppo_step_count >= args.ppo_update_freq:
            actor_loss, critic_loss = agent.update()
            total_ppo_actor_loss += actor_loss
            total_ppo_critic_loss += critic_loss
            ppo_update_count += 1
            agent.clear_buffer()
            ppo_step_count = 0

        train_loss += loss_to_backward.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(inputs)}/{len(trainloader.dataset)} '
                  f'({100. * batch_idx / len(trainloader):.0f}%)]\t'
                  f'Loss: {loss_to_backward.item():.6f}, Acc: {100. * correct / total:.2f}%')

    if len(agent.states) > 0:
        res = agent.update()
        if res:
            actor_loss, critic_loss = res
            total_ppo_actor_loss += actor_loss
            total_ppo_critic_loss += critic_loss
            ppo_update_count += 1
        agent.clear_buffer()

    end_time = time.time()
    train_times.append(end_time - start_time)

    avg_train_loss = train_loss / len(trainloader)
    avg_ppo_actor_loss = total_ppo_actor_loss / ppo_update_count if ppo_update_count > 0 else 0
    avg_ppo_critic_loss = total_ppo_critic_loss / ppo_update_count if ppo_update_count > 0 else 0

    print(f'Train Epoch: {epoch}, Loss: {avg_train_loss:.4f}, '
          f'PPO Actor Loss: {avg_ppo_actor_loss:.4f}, PPO Critic Loss: {avg_ppo_critic_loss:.4f}, '
          f'Acc: {100. * correct / total:.2f}%, Time: {end_time - start_time:.2f}s')

    return avg_train_loss, avg_ppo_actor_loss, avg_ppo_critic_loss


def test(epoch):
    global best_acc, best_epoch

    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    start_time = time.time()

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs, _ = model(inputs)
            loss = test_criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    end_time = time.time()
    test_times.append(end_time - start_time)

    acc = 100.0 * correct / total
    print(f'Test Epoch: {epoch}, Loss: {test_loss / len(testloader):.4f}, '
          f'Acc: {acc:.2f}%, Time: {end_time - start_time:.2f}s')

    if acc > best_acc:
        print(f'Saving... Best accuracy improved from {best_acc:.2f}% to {acc:.2f}%')
        best_acc = acc
        best_epoch = epoch
        if args.save_model:
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            torch.save(state, f'./checkpoint_ppo_{args.dataset}_{args.model}.pth')


def save_results():
    results = {
        'algorithm': 'PPO',
        'dataset': args.dataset,
        'model': args.model,
        'best_accuracy': best_acc,
        'best_epoch': best_epoch,
        'total_epochs': args.num_epoch,
        'avg_train_time': np.mean(train_times),
        'avg_test_time': np.mean(test_times),
        'data_agent_savings': trainset.total_save(),
        'ppo_hyperparams': {
            'lr_a': args.ppo_lr_a,
            'lr_c': args.ppo_lr_c,
            'eps_clip': args.ppo_eps_clip,
            'k_epochs': args.ppo_k_epochs,
            'gamma': args.ppo_gamma,
            'gae_lambda': args.ppo_gae_lambda,
            'update_freq': args.ppo_update_freq,
        },
    }

    results_path = f'results_ppo_{args.dataset}_{args.model}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'==> Results saved to {results_path}')


if __name__ == '__main__':
    print(f'==> Starting PPO training: dataset={args.dataset}, model={args.model}')
    print(f'==> PPO hyperparameters: lr_a={args.ppo_lr_a}, lr_c={args.ppo_lr_c}, '
          f'eps_clip={args.ppo_eps_clip}, k_epochs={args.ppo_k_epochs}')

    for epoch in range(start_epoch, args.num_epoch):
        lr_scheduler = build_scheduler(last_epoch=epoch * len(trainloader) - 1)

        train_loss, ppo_actor_loss, ppo_critic_loss = train(epoch)
        train_losses_history.append(train_loss)
        ppo_actor_losses_history.append(ppo_actor_loss)
        ppo_critic_losses_history.append(ppo_critic_loss)

        test(epoch)
        agent.lr_decay(epoch)

    print('\n==> Training completed')
    print(f'Best Test Accuracy: {best_acc:.2f}% (Epoch {best_epoch})')
    print('Total saved sample forwarding by Data Agent:', trainset.total_save())
    print(f'Train times per epoch: {train_times}')
    print(f'Test times per epoch: {test_times}')
    print(f'Average train time: {np.mean(train_times):.2f} seconds')
    print(f'Average test time: {np.mean(test_times):.2f} seconds')
    print(f'Total train time: {np.sum(train_times):.2f} seconds')
    print(f'Total test time: {np.sum(test_times):.2f} seconds')

    save_results()
