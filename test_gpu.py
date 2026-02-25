import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

import torch
import time

print(torch.__version__)

# 测试一个简单的计算
x = torch.randn(1, 3).cuda()
print(x + x)

# 确保在 GPU 上运行
device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(2048, 2048).to(device)
y = torch.randn(2048, 2048).to(device)

# 预热
for _ in range(10):
    _ = torch.matmul(x, y)

# 计时测试
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    _ = torch.matmul(x, y)
torch.cuda.synchronize()
end = time.time()

print(f"100次 2048x2048 矩阵乘法耗时: {(end - start)*1000:.2f} ms")