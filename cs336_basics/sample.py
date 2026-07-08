"""
Generating Text from Model as chat completion
Support Temperature Scaling & necleus samping
"""

import torch 
from modules import *
# 给定一个对话前缀，需要 Prefill， 然后KV Caching 解码 （考虑先实现原始的 n^2）


def chat_completion(model, prompt: list[int], max_generated:int, temperature: float, p: Optional[Float]) -> Tensor:
    device = get_device()
    prefix = torch.as_tensor(prompt, device=device)
    
    