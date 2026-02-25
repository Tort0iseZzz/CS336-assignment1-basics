

# 1. 加载模型
checkpoint = torch.load("checkpoint.pt")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 2. 基础生成循环
input_ids = tokenizer.encode("Once upon a time")
for _ in range(max_new_tokens):
    # 你的代码：前向传播、获取末尾 logits、采样、拼接
    ...