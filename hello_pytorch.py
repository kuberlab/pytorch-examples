from __future__ import print_function
import torch

if torch.cuda.is_available():
    print("GPU based")
    a = torch.LongTensor(10).fill_(3).cuda()
    print(type(a))
    print(a.cpu())
else:
    print("CPU based")
    x = torch.Tensor(5, 3)
    print(x)
