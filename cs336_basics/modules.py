import torch
import torch.nn as nn
from einops import rearrange, einsum
import einx

class Linear(nn.Module):
    def __init__(self, in_features:int, out_features:int, device:torch.device | None =None, dtype:torch.dtype | None =None):
        super().__init__()
        w = torch.empty(out_features, in_features, dtype=dtype, device=device)
        nn.init.trunc_normal_(w, std=1, a=-3, b=3)
        self.parameter:torch.nn.Parameter = nn.Parameter(w)
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return einx.dot("b ... i, o i -> b ... o", x, self.parameter)

class Embedding(nn.Module):
    def __init__(self, num_embedding:int, embedding_dim:int, device:torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.num_embedding = num_embedding
        self.embedding_dim = embedding_dim 
        w = torch.empty(num_embedding, embedding_dim, dtype=dtype, device=device)
        nn.init.trunc_normal_(w, a=-3, b=3)
        self.parameter=nn.Parameter(w)
    
    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        return self.parameter[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model:int, eps:float=1e-5, device:torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.eps=eps
        w = torch.empty(d_model, dtype=dtype, device=device)
        nn.init.trunc_normal_(w, a=-3, b=3)
        self.parameter=nn.Parameter(w)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        in_dtype=x.dtype
        x = x.float()
        
        RMS = torch.sqrt(einx.mean("... [a]", x**2) + self.eps)
        x = einx.divide("... a, ... -> ... a", x, RMS)
        return einx.multiply("... a, a -> ... a", x, self.parameter).to(in_dtype)

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

 