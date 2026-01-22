import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Device Count: {torch.cuda.device_count()}")
    x = torch.randn(100, 100).cuda()
    print("Tensor allocation on GPU successful")
else:
    print("Running on CPU")