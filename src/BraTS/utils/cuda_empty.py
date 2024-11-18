import torch

# Clear cache
torch.cuda.empty_cache()

# Optionally, free up any unreferenced memory
torch.cuda.memory_summary(device=None, abbreviated=False)
