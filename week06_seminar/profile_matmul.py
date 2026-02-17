import torch

device = "cuda"
dim = 4096

A = torch.randn(dim, dim, device=device)
B = torch.randn(dim, dim, device=device)

torch.cuda.synchronize()

for _ in range(10):
    C = torch.matmul(A, B)

torch.cuda.synchronize()
