import os
import torch

tensor = torch.zeros([1, 16, 18, 64])


for _ in range(20):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    tensor = tensor.cpu()

    start.record()
    tensor = tensor.cuda()
    end.record()

    torch.cuda.synchronize()

    print(start.elapsed_time(end))
