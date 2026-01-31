import os
import multiprocessing
import regex as re
from collections import Counter, defaultdict
from typing import BinaryIO


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
            # Decode  # star: we shouldn't do that!
            # text = segment.decode("utf-8", errors="ignore")

            # 4. pre-tokenize
            for match in re.finditer(COMPLIED_PAT, segment):
                word_str = match.group()
                # words -> tuple
                # eg. "hello" -> (b'h', b'e', b'l', b'l', b'o')
                # word_tuple = tuple(bytes([b]) for b in word_str.encode("utf-8"))
                word_tuple = tuple(bytes([b]) for b in word_str)
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
    PAT = rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""  
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

    return(vocab, merges)