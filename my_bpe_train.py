import os
from cs336_basics.bpe_tokenizer import train_bpe, save_tokenizer_assets, Tokenizer

if __name__ == "__main__":
    # 配置路径
    INPUT_FILE_LARGE = "data/TinyStoriesV2-GPT4-train.txt"
    INPUT_FILE_SMALL = "data/TinyStoriesV2-GPT4-valid.txt"
    INPUT_FILE = INPUT_FILE_LARGE
    VOCAB_SAVE_PATH = "models/tokenizer/vocab.json"
    MERGES_SAVE_PATH = "models/tokenizer/merges.txt"
    
    # 确保目录存在
    os.makedirs(os.path.dirname(VOCAB_SAVE_PATH), exist_ok=True)

    # 训练配置
    # 注意：TinyStories 很大，如果全量训练太慢，可以先用一个小子集测试
    VOCAB_SIZE = 32768
    SPECIAL_TOKENS = ["<|endoftext|>"]
    NUM_PROCESSES=32

    print(f"🚀 开始在 {INPUT_FILE} 上训练 BPE...")
    
    # 调用你实现的 train_bpe
    vocab, merges = train_bpe(
        input_path=INPUT_FILE,
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        num_processes=NUM_PROCESSES
    )

    # 保存结果
    print(f"🚀 训练完成！正在保存结果到 {VOCAB_SAVE_PATH} 和 {MERGES_SAVE_PATH}...")
    save_tokenizer_assets(vocab, merges, VOCAB_SAVE_PATH, MERGES_SAVE_PATH)
    print("保存完成！")