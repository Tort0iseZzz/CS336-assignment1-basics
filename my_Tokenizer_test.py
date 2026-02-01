import os
import regex as re
from cs336_basics.bpe_tokenizer import train_bpe, save_tokenizer_assets, Tokenizer

if __name__ == "__main__":
    VOCAB_SAVE_PATH = "models/tokenizer/vocab.json"
    MERGES_SAVE_PATH = "models/tokenizer/merges.txt"
    SPECIAL_TOKENS = ["<|endoftext|>"]

    # 加载刚刚训练好的分词器
    print(f"🚀 正在根据数据 {VOCAB_SAVE_PATH} 和 {MERGES_SAVE_PATH} 创建Tokenizer...")
    tokenizer = Tokenizer.from_files(VOCAB_SAVE_PATH, MERGES_SAVE_PATH, SPECIAL_TOKENS)

    test_str = " loved" # 注意前面有一个空格
    # 打印预分词结果
    for match in re.finditer(tokenizer.PAT, test_str):
        print(f"{test_str} 的预分词片段: '{match.group()}'")

    # 测试一段 TinyStories 中的典型文本
    TEST_TEXT1 = "Once upon a time, there was a little girl named Lily😊."
    TEST_TEXT2 = " loved"
    TEST_TEXT3 = "a sdhueihfiegfwyuegd dhdu sdoi i so do this\n<|endoftext|>"
    test_text = TEST_TEXT3
    ids = tokenizer.encode(test_text)
    tokens = [tokenizer.decode([i]) for i in ids]

    print(f"原始文本: {test_text}")
    print(f"id结果: {ids}")
    print(f"切分结果: {tokens}")
    print(f"压缩比: {len(test_text) / len(ids):.2f} (字符/Token)")