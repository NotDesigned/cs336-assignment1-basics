"""
Generating Text from Model as chat completion
Support Temperature Scaling & necleus samping
"""

import torch 
from cs336_basics.modules import *
from cs336_basics.tokenizer import *
from jaxtyping import Bool, Float, Int
# 给定一个对话前缀，需要 Prefill， 然后KV Caching 解码 
# 考虑先实现原始的 n^2
def sample(logit: Float[Tensor, "B V"], temperature: float, p: Optional[float]) -> Int[Tensor, "B"]:
    if temperature == 0:
        return logit.argmax(dim=-1, keepdim=False)
    probs = (logit/temperature).softmax(dim=-1)
    
    if p != None and p < 1:
        if p < 0:
            raise ValueError("p cannot less than zero")
        device = logit.device
        B, V = logit.shape
        probs_sort, indice = torch.sort(probs, descending=True, dim=-1)

        num = (torch.cumsum(probs_sort, dim=-1) >= p ).int().argmax(dim=-1)
        position = torch.arange(V, device=device).unsqueeze(0) # [1, V]
        keep = (position <= num.unsqueeze(1)) # [B, 1] -> [B, V]
        probs_sort[~keep] = 0 
        probs_sort = probs_sort / probs_sort.sum(dim=-1, keepdim=True)

        sorted_indice_sample = torch.multinomial(probs_sort, num_samples=1)
        return indice.gather(dim = -1, index=sorted_indice_sample).squeeze(-1)

    else:
        return torch.multinomial(probs, num_samples=1).squeeze(-1) 
    

def chat_completion(model, prompt: Int[Tensor, "B S"], tk: BPE_Tokenizer, max_generated:int, temperature: float, p: Optional[float]) -> Int[Tensor, "B G"]:
    device = prompt.device
    B, S = prompt.shape 
    prefix = torch.empty((B, max_generated+ S), device=device, dtype=torch.int)
    prefix[..., :S] = prompt
    
    for i in range(max_generated):
        
        logit:Tensor = model(prefix[..., : S+i]) [..., -1, :]
        
        token = sample(logit, temperature, p)
        
        prefix[..., i+S] = token
    
    generated = prefix[..., S:]
    mask = (generated == tk.reverse_vocab[b'<|endoftext|>'] ).int()
    num =  mask.argmax(dim=-1) 
    num[~mask.any(dim=-1)] = generated.shape[-1]
    position = torch.arange(generated.shape[-1], device=device).unsqueeze(0) 
    keep = (position <= num.unsqueeze(1))
    generated[~keep] = -1
    return generated

def test_completion(model, prompt:str, tokenizer: BPE_Tokenizer, max_generated:int, temperature: float, p: Optional[float]):
    tokens = tokenizer.encode(prompt)
    tokens = torch.as_tensor(tokens).unsqueeze(0)
    return tokenizer.decode(chat_completion(model, tokens, tokenizer, max_generated, temperature, p).squeeze(0).tolist())