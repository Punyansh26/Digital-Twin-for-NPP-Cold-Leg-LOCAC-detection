import torch
import torch.nn as nn
from torch.cuda.amp import autocast

h = torch.randn(4, 25000, 32, 8, device='cuda', requires_grad=True)
norm = nn.LayerNorm(32 * 8).cuda()
with autocast():
    out = norm(h.view(*h.shape[:-2], -1))
    loss = out.sum()

loss.backward()
print(norm.weight.grad.shape)
