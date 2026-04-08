import torch

from models import SimplePolicy

policy = SimplePolicy(7)
inp = torch.randn(1, 3, 210, 160)
out = policy.get_action(inp)
print(out.log_prob(torch.tensor([1.0])))
