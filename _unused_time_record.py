import os
import torch

tensor1 = torch.zeros([10000000])
tensor2 = torch.zeros([10000000])

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
tensor1 = tensor1.cuda()

end.record()
torch.cuda.synchronize()
print(start.elapsed_time(end))

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
tensor2 = tensor2.cuda()

end.record()
torch.cuda.synchronize()
print(start.elapsed_time(end))