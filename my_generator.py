import math
import torch
import argparse
from torch import Tensor
from torch.nn.parameter import Parameter
from collections.abc import Callable, Iterable
from typing import Optional
from jaxtyping import Bool, Float, Int
from cs336_basics.transformer_arch import Transformer_lm
from cs336_basics.bpe_tokenizer import Tokenizer

def softmax_with_temperature(in_features: Tensor, dim: int, t: float) -> Tensor:
    # softmax(xi, t) = exp(xi/t - max) / sum_j(exp(xj/t - max))
    # substract max from all xi is to avoid inf

    # temperature scaling
    in_features = in_features / t

    # 1. find the max_x
    max_x = torch.max(in_features, dim=dim, keepdim=True)[0]

    # 2. e^(x - max)
    exp_x = torch.exp(in_features - max_x)

    # 3. sum(exp(x))
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)

    return exp_x / sum_exp


def top_p_sampling(probs, p):
    """
    Args:
        probs: logits after softmax, (vocab_size,) or (batch, vocab_size)
        p: (0 < p <= 1)
    Returns:
        next_token: the sampled token ID
    """
    # 1. sort the probs
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    
    # 2. Cumulative Sum
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 3. find the position that sum_probs > p
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    
    # 4. clear lower probs
    sorted_probs[sorted_indices_to_remove] = 0.0
    
    # 5. re-normalize
    sorted_probs = sorted_probs / torch.sum(sorted_probs, dim=-1, keepdim=True)
    
    # 6. random sampling
    next_token_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)

    # 7. to original (before sorted) idx
    next_token_raw_idx = torch.gather(sorted_indices, -1, next_token_sorted_idx)
    
    return next_token_raw_idx


def load_trained_model(checkpoint_path, model_cfg):
    # 1. 按照训练时的参数初始化一个空的模型实例
    model = Transformer_lm(**model_cfg) 
    
    # 2. 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 3. 加载状态字典
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval() # 切换到评估模式
    return model


@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens, top_p, temperature=0.2, eoftoken=256):
    # prompt_ids: (1, seq_len)
    generated = prompt_ids
    hit_eof = False
    
    for _ in range(max_new_tokens):
        # 如果序列超过了模型支持的最大长度，进行截断
        curr_input = generated[:, -model.max_seq_len:] 
        
        # 1. forward
        logits = model.forward(curr_input) # (1, seq_len, vocab_size)
        
        # 2. get the logits
        next_token_logits = logits[:, -1, :]
        
        # 3. softmax with temperature scaling
        probs = softmax_with_temperature(next_token_logits, dim=-1, t=temperature)

        # 4. Top-p (Nucleus)
        next_token = top_p_sampling(probs, top_p)
        
        # 5. if hit endoftext token, break
        if next_token == eoftoken: 
            hit_eof = True
            break
            
        # 6. cat
        generated = torch.cat((generated, next_token), dim=1)   
        
    return generated, hit_eof


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS336 Assignment 1 Text Generator")
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--prompts", type=str, default="Once upon a time")
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--max_length", type=int, default=200)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()
    debug = args.debug

    tokenizer = Tokenizer.from_files(
        vocab_filepath="./models/tokenizer/vocab.json", 
        merges_filepath="./models/tokenizer/merges.txt",
        special_tokens={"<|endoftext|>": 256}
    )

    if debug:
        print("...tokenizer initialized...")

    checkpoint_path = "./models/tiny/latest_checkpoint.pt"
    model_cfg = {
        "vocab_size": 32800,
        "num_layers": 6,
        "num_heads": 8,
        "d_model": 512,
        "context_length": args.context_length
    }
    model = load_trained_model("./models/tiny/latest_checkpoint.pt", model_cfg)

    if debug:
        print("...model loaded...")

    prompt_text = args.prompts
    prompt_ids = torch.tensor([tokenizer.encode(prompt_text)]) # 形状: (1, seq_len)

    generated_ids, hit_eof = generate(model, prompt_ids, args.max_length, args.top_p, args.temperature)

    output_ids = generated_ids[0].tolist()
    output_text = tokenizer.decode(output_ids)

    print(f"Output: \n{output_text}")

    if hit_eof:
        print("\n [THE END]")
    else:
        print("\n [TO BE CONTINUED]\nMaximum token limit reached. The story remains suspended in the latent space.")
        print("Tip: Try increasing 'max_length' to see the rest of the tale!")
