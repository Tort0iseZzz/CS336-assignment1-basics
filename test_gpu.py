import torch
import time

# 打印详细环境信息，确认驱动是否真的在工作
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device Name: {torch.cuda.get_device_name(0)}")
print(f"Device Capability: {torch.cuda.get_device_capability(0)}")

# 核心：看看这版 Torch 是否真的支持 sm_120
print(f"Supported Architectures: {torch.cuda.get_arch_list()}")

# 准备测试数据
device = "cuda"
# 使用 float16 或 bfloat16，这是 Blackwell 的强项！
# RTX 5060 在运行 fp32 时可能并不比旧显卡快多少
dtype = torch.float16 

x = torch.randn(2048, 2048, device=device, dtype=dtype)
y = torch.randn(2048, 2048, device=device, dtype=dtype)

# 增加预热次数（新架构需要更多时间来稳定频率和 JIT 缓存）
for _ in range(50):
    _ = torch.matmul(x, y)

torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    _ = torch.matmul(x, y)
torch.cuda.synchronize()
end = time.time()

print(f"100次 2048x2048 (FP16) 矩阵乘法耗时: {(end - start)*1000:.2f} ms")