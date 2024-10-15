import torch
from torch import nn

import wandb

import sys
sys.path.append('..')


dtype = torch.float
device = torch.device('cuda:0')

n = 10
r = 4
T = 100
rand_ratio = 0.25
type = 2 # 0galore 1adamw 2golore
epoch = 200000
lr = 3e-5
seed = 42
sigma = 1
optimizer = 'SGD'
upload = False
grad_accumulation = 1

if upload:
    wandb.init(project="inexact")
    config = {
        'n':n,
        'r':r,
        'T':T,
        'rand_ratio':rand_ratio,
        'lr': lr,
        'type': type,
        'seed':seed,
        'optimizer':optimizer
    }
    wandb.config.update(config)
torch.manual_seed(seed)

A = torch.cat([torch.eye(n - r), torch.zeros(n - r, r)], dim = 1)
B = torch.cat([torch.randn(n - r, n - r) * sigma, torch.zeros(n - r, r)], dim = 1)
B = torch.cat([B, torch.zeros(r, n)], dim = 0)
C = torch.cat([torch.zeros(r, n - r), torch.eye(r)], dim = 1)
C = torch.cat([torch.zeros(n - r, n), C], dim = 0)

A, B, C = A.to(device), B.to(device), C.to(device)


class Net(nn.Module):
    def __init__(self, n, r):
        super().__init__()
        self.X = sigma / 2 * torch.cat([torch.eye(n - r), torch.zeros(n - r, r)], dim = 1)
        self.X = torch.cat([self.X, torch.randn(r, n) * sigma], dim = 0)
        self.X = nn.Parameter(self.X)
    def forward(self, A, B):
        tmp = A @ self.X
        return 0.5 * (tmp * tmp).sum() + (B * self.X).sum()

from galore_adam import AdamW as GaLoreAdamW
from peft_pretraining.galore_torch.sgd import SGD
from exp.Golore import ReLoRaModel

if type == 0:
    net = Net(n, r).to(device)
    param_groups = [{'params': net.parameters(), 'rank': r, 'update_proj_gap': T, 'scale': 1, 'proj_type': 'left'}]
    if optimizer == 'AdamW': optimizer = GaLoreAdamW(param_groups, lr = lr)
    else: optimizer = SGD(param_groups, lr = lr)

if type == 1:
    net = Net(n, r).to(device)
    if optimizer == 'AdamW': optimizer = torch.optim.AdamW(net.parameters(), lr = lr)
    else: optimizer = torch.optim.SGD(net.parameters(), lr = lr)


if type == 2:
    net = Net(n, r)
    net = ReLoRaModel(net, target_modules=['X'], r = r).to(device)
    params = [        
        {
            "params": [p for n, p in net.named_parameters() if p.requires_grad]
        }
    ]
    from golore_adamw import AdamW
    if optimizer == 'AdamW': optimizer = AdamW(net.parameters(), lr = lr)
    else: optimizer = torch.optim.SGD(net.parameters(), lr = lr)
    for name, module in net.named_parameters():
        # print(name)
        if name == 'X.weight':
            weight3 = module
        if name == 'X.lora_B.weight':
            lora_B = module
        if name == 'X.lora_A.weight':
            lora_A = module

L = []

import tqdm
bar = tqdm.tqdm(range(epoch))

from loguru import logger


steps = -1
updata_steps = 0
loss_sum = 0
Loss = 0
for idx in range(epoch):

    bar.update(1)
    steps += 1
    if type == 2:
        reset_relora = updata_steps % T == 0
        net._config.forward_type = reset_relora
    l = net(A, B)
    l.backward()
    Loss = l.item()
    with torch.no_grad():
        # pos = torch.rand(n, n) <= 0.5
        times = 1
        if steps % T == 0: times = grad_accumulation
        for _ in range(times):
            xi = torch.eye(r)
            # xi[pos] = -1.0
            xi *= sigma
            xi = torch.cat([torch.zeros(r, n - r), xi], dim = 1)
            xi = torch.cat([torch.zeros(n - r, n), xi], dim = 0)
            import random
            if random.random() <= 0.5: xi = -xi
            xi = xi.to(device)
            xi /= grad_accumulation
            if type == 0:
                net.X.grad += xi
            if type == 1:
                net.X.grad += xi
            if type == 2:
                if reset_relora:
                    weight3.grad += xi
                else:
                    lora_B.grad -= lora_A.T @ xi

    loss_sum += Loss

    # print(net.X.grad)

    if type == 2 and reset_relora:
        use_rand = steps / epoch >= rand_ratio
        net.merge_and_reinit(optimizer, use_rand)
    
    updata_steps += 1
    optimizer.step()
    optimizer.zero_grad()

    if updata_steps % 1000 == 0:
        logger.info(f'Loss: {Loss}')
    if updata_steps % 100 == 0:
        if upload:
            wandb.log({
                'loss_avg': loss_sum / updata_steps,
                'loss':Loss
            })

# import matplotlib.pyplot as plt
# import numpy as np

# plt.xlabel('step')
# plt.ylabel('loss')

# L1 = np.array(L1)
# plt.plot(L1, label = 'GaloreAdamW')
# L2 = np.array(L2)
# plt.plot(L2, label = 'AdamW')
# L3= np.array(L3)
# plt.plot(L3, label = 'GoloreAdamW')
# plt.legend()

# plt.show()
# print(L.shape)