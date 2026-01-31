import os
import multiprocessing
import regex as re
from collections import Counter, defaultdict
from typing import BinaryIO, Iterable, Iterator
import json


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def process_single_chunk(input_path, start, end, special_tokens_bytes, COMPLIED_PAT):
    """worker function for child process"""
    local_counts = Counter()
    with open(input_path, "rb") as f:
        # 1. read from start till end
        f.seek(start)
        chunk_bytes = f.read(end - start)

        # 2. replace \r\n to \n
        # this is mainly due to Windows vs Unix
        chunk_bytes = chunk_bytes.replace(b"\r\n", b"\n").replace(b"\r", b"\n")

        # 3. Remove special tokens
        # re.escape here auto translate '|' into '\|'
        combined_special_regex = b'|'.join([re.escape(token) for token in special_tokens_bytes])
        segments = re.split(combined_special_regex, chunk_bytes)

        for segment in segments:
            if not segment:
                continue
            # Decode
            text = segment.decode("utf-8", errors="ignore")

            # 4. pre-tokenize
            for match in re.finditer(COMPLIED_PAT, text):
                word_str = match.group()
                # words -> tuple
                # eg. "hello" -> (b'h', b'e', b'l', b'l', b'o')
                # word_tuple = tuple(bytes([b]) for b in word_str.encode("utf-8"))
                word_tuple = tuple(bytes([b]) for b in word_str.encode("utf-8"))
                local_counts[word_tuple] += 1
    return local_counts

def merge_at_word(word_tuple, pair, new_token):
    """
    在一个单词元组中，将所有的 (p1, p2) 替换为 new_token。
    例如: ('a', 'b', 'c', 'a', 'b') -> ('ab', 'c', 'ab')
    """
    new_word = []
    i = 0
    p1, p2 = pair
    while i < len(word_tuple):
        if i < len(word_tuple) - 1 and word_tuple[i] == p1 and word_tuple[i+1] == p2:
            new_word.append(new_token)
            i += 2
        else:
            new_word.append(word_tuple[i])
            i += 1
    return tuple(new_word)

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes = 32):
    
    ##### Constants #####
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""  
    COMPLIED_PAT = re.compile(PAT) 
    ####################

    with open(input_path, "rb") as f:
        # cut the file into several chunks
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # convert special tokens into bytes
    special_tokens_bytes = [
        t.encode("utf-8") if isinstance(t, str) else t 
        for t in special_tokens
    ]

    # * Parallelism *
    # 1. prepare parameters
    tasks = [
        (input_path, boundaries[i], boundaries[i+1], special_tokens_bytes, COMPLIED_PAT)
        for i in range(len(boundaries) - 1)
    ]
    
    # 2. multiprocessing_pool
    # starmap 会自动将 tasks 中的元组展开作为 process_single_chunk 的参数
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(process_single_chunk, tasks)
    
    # 3. merge the results
    # now word_counts: { (b'h', b'e', b'l', b'l', b'o'): 1000, (b't', b'h', b'e'): 500, ... }
    word_counts = Counter()
    for res in results:
        word_counts.update(res)

    # * merge bytes pairs *
    # create pair_counts and pair2words (function as cache)
    pair_counts = Counter()
    pair2words = defaultdict(set) # set here to avoid repeating
    for word_tuple, cnt in word_counts.items():
        for i in range(len(word_tuple) - 1):
            pair = (word_tuple[i], word_tuple[i+1])
            pair_counts[pair] += cnt
            pair2words[pair].add(word_tuple)

    merges = [] # list[tuple[bytes, bytes]]

    # start generate
    generate_count = vocab_size - 256 - len(special_tokens)
    for _ in range(generate_count):
        if not pair_counts: break
        
        # 1. find max pair
        best_pair = max(
            pair_counts.keys(), 
            key=lambda p: (pair_counts[p], p)
        )

        # 2. merge the pair
        new_token = best_pair[0] + best_pair[1]
        merges.append(best_pair)

        # 3. update words that include best_pair
        words_to_update = list(pair2words[best_pair])
        
        for old_word in words_to_update:
            word_count = word_counts[old_word]
            
            # --- A. 'remove old word' ---
            for i in range(len(old_word) - 1):
                p = (old_word[i], old_word[i+1])
                pair_counts[p] -= word_count
                if pair_counts[p] <= 0: del pair_counts[p]
                pair2words[p].discard(old_word) # will do nothing if there is no old_word
            
            # --- B. generate new word ---
            new_word = merge_at_word(old_word, best_pair, new_token)
            
            # --- C. update word_counts ---
            del word_counts[old_word]
            word_counts[new_word] = word_count
            
            # --- D. update pair_counts and pair2words ---
            for i in range(len(new_word) - 1):
                p = (new_word[i], new_word[i+1])
                pair_counts[p] += word_count
                pair2words[p].add(new_word)

    # calculate vocab
    vocab = dict()
    for i in range(256):
        vocab[i] = bytes([i])
    
    current_id = 256

    
    for st in special_tokens_bytes:
        vocab[current_id] = st
        current_id += 1


    for p1, p2 in merges:
        token = p1 + p2
        vocab[current_id] = token
        current_id += 1

    return (vocab, merges)

def save_tokenizer_assets(vocab, merges, vocab_path, merges_path):
    """
    将 BPE 分词器的资源保存到磁盘。
    
    参数:
        vocab: dict[int, bytes] - ID 到字节的映射
        merges: list[tuple[bytes, bytes]] - 合并规则列表
        vocab_path: 词汇表保存路径 (e.g., 'vocab.json')
        merges_path: 合并规则保存路径 (e.g., 'merges.txt')
    """
    # --- 保存 Vocabulary ---
    # 核心点：JSON 不支持 bytes，必须转为字符串
    # 我们使用 latin-1，因为它能一对一映射 0-255 的所有字节
    serializable_vocab = {
        token_id: token_bytes.decode('latin-1') 
        for token_id, token_bytes in vocab.items()
    }
    
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_vocab, f, indent=4, ensure_ascii=False)
    
    # --- 保存 Merges ---
    # 核心点：保持顺序，且每行存储一对合并规则
    with open(merges_path, 'w', encoding='utf-8') as f:
        for pair in merges:
            # 同样使用 latin-1 确保字节被安全转换成字符
            p0 = pair[0].decode('latin-1')
            p1 = pair[1].decode('latin-1')
            f.write(f"{p0} {p1}\n")

    print(f"✅ 已成功保存词汇表至: {vocab_path}")
    print(f"✅ 已成功保存合并规则至: {merges_path}")


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes],
                merges: list[tuple[bytes, bytes]],
                special_tokens: list[str] | None = None):
        self.vocab: dict[int,bytes] = vocab
        self.merges: list[tuple[bytes, bytes]] = merges
        self.special_tokens = None
        if special_tokens:
            self.special_tokens = special_tokens
        # to search for the rank of a pair in O(1)
        self.merges_dict: dict[tuple[bytes, bytes], int] = {pair: i for i, pair in enumerate(self.merges)}
        # to cache word_str
        self.word_cache: dict[str, list[int]] = dict()
        # to search for the id of bytes in O(1)
        self.vocab2id:dict[bytes, int] = {b: i for i, b in vocab.items()}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):  
        # 1. 加载 Vocab
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 转换回 dict[int, bytes]
            vocab = {int(k): v.encode('latin-1') for k, v in data.items()}

        # 2. 加载 Merges
        merges = []
        if os.path.exists(merges_filepath):
            with open(merges_filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.rstrip('\n')
                    if not line: continue
                    # 按空格切分，并还原回字节
                    parts = line.split(' ')
                    if len(parts) == 2:
                        merges.append((parts[0].encode('latin-1'), parts[1].encode('latin-1')))
        
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str)-> list[int]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""  
        COMPLIED_PAT = re.compile(PAT)
        
        result_ids: list[int] = []

        # * pre-tokenize
        # Decode
        for match in re.finditer(COMPLIED_PAT, text):
            word_str = match.group()
            # cached words
            if word_str in self.word_cache:
                ids = self.word_cache[word_str]
                result_ids += ids
                continue

            # words -> tuple
            # eg. "hello" -> (b'h', b'e', b'l', b'l', b'o')
            # word_tuple = tuple(bytes([b]) for b in word_str.encode("utf-8"))
            word_tuple = tuple(bytes([b]) for b in word_str.encode("utf-8"))

            # apply the merges
            while True:
                pairs = []
                for i in range(len(word_tuple) - 1):
                    pairs.append((word_tuple[i], word_tuple[i+1]))
                
                # 找到这些 pair 中在 merges_dict 里 rank 最小的一个
                target_pair = min(pairs, key=lambda p: self.merges_dict.get(p, float('inf')))
                
                if target_pair not in self.merges_dict:
                    break  # 没有可以继续合并的了
                    
                # 执行合并：将序列中所有的 bigram 替换为合并后的 bytes
                replacement = target_pair[0] + target_pair[1]
                word_tuple = merge_at_word(word_tuple, target_pair, replacement)
                
                if len(word_tuple) == 1:
                    break
            
            # get the ids
            ids = []
            for token_bytes in word_tuple:
                ids.append(self.vocab2id[token_bytes])
            result_ids += ids
            self.word_cache[word_str] = ids

        return result_ids

    def encode_iterable(self, iterable: Iterable[str])->Iterator[int]:
        remainder = ""
        
        # 预先准备好匹配特殊 token 的正则，用于寻找安全边界
        #special_pat = None
        #if self.special_tokens:
        #   special_pat = "|".join(re.escape(t) for t in self.special_tokens)

        for chunk in iterable:
            # 1. 拼接上一次剩下的“尾巴”
            current_text = remainder + chunk
            
            # 2. 寻找安全切分点
            last_safe_index = max(
                current_text.rfind(" "), 
                current_text.rfind("\n"),
                current_text.rfind("\r")
            )
            
            # 如果存在特殊 token，还要考虑不要切断特殊 token
            # 这里简单起见，如果没找到空格，我们就不切分，继续积累 remainder
            if last_safe_index != -1:
                # 安全的部分
                ready_to_encode = current_text[:last_safe_index + 1]
                # 剩下的尾巴
                remainder = current_text[last_safe_index + 1:]
    
                # 3. 调用 encode 处理安全部分，并逐个 yield ID
                for token_id in self.encode(ready_to_encode):
                    yield token_id
            else:
                # 没找到安全边界，全存入 remainder 等待下一块
                remainder = current_text
                
        # 最后，处理最后剩下的部分
        if remainder:
            for token_id in self.encode(remainder):
                yield token_id

    def decode(self, ids:list[int])->str:
        # collect all bytes
        all_bytes = b"".join(self.vocab[i] for i in ids)
        # decode them all
        return all_bytes.decode("utf-8", errors="replace")