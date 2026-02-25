import numpy as np
from tqdm import tqdm
from cs336_basics.bpe_tokenizer import Tokenizer # 确保路径正确

# 1. 加载你刚训练好的词表
# 假设你的 Tokenizer 实现支持从这两个文件初始化
tokenizer = Tokenizer.from_files(
    vocab_filepath="./models/tokenizer/vocab.json", 
    merges_filepath="./models/tokenizer/merges.txt",
    special_tokens={"<|endoftext|>": 256}
)

def process_file(input_txt, output_bin):
    print(f"\n开始处理: {input_txt}")
    
    # 统计总行数以便显示进度条（可选，但推荐）
    print("正在扫描文件行数...")
    with open(input_txt, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    all_ids = []
    
    # 2. 使用 tqdm 按行读取并编码
    with open(input_txt, "r", encoding="utf-8") as f:
        # 使用 unit="line" 让进度条显示每秒处理多少行
        for line in tqdm(f, total=total_lines, desc="Encoding progress", unit="line"):
            line = line.strip()
            if line:
                # 编码当前行
                ids = tokenizer.encode(line)
                all_ids.extend(ids)
                # 提示：如果你的故事之间需要特殊分隔符，可以在这里加上：
                # all_ids.append(tokenizer.special_tokens['<|endoftext|>'])
    
    print(f"编码完成，正在转换为 NumPy 格式...")
    ids_array = np.array(all_ids, dtype=np.uint16)
    
    print(f"正在写入二进制文件: {output_bin}")
    ids_array.tofile(output_bin)
    print(f"成功! 文件已保存。总 Token 数: {len(all_ids)}")

# 执行转换
if __name__ == "__main__":
    process_file("./data/TinyStoriesV2-GPT4-train.txt", "./data/train.bin")
    process_file("./data/TinyStoriesV2-GPT4-valid.txt", "./data/val.bin")