import torch
import torch.nn as nn
from einops import rearrange, einsum
import einx

class Linear(nn.Module):
    def __init__(self, in_features:int, out_features:int, device:torch.device | None =None, dtype:torch.dtype | None =None):
        super().__init__()
        self.parameter = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype, device=device))
        nn.init.trunc_normal_(self.parameter, a=-3, b=3)
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return einx.dot("b ... i, o i -> b ... o", x, self.parameter)

class Embedding(nn.Module):
    def __init__(self, num_embedding:int, embedding_dim:int, device:torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.num_embedding = num_embedding
        self.embedding_dim = embedding_dim 
        self.parameter = nn.Parameter(torch.empty(num_embedding, embedding_dim, dtype=dtype, device=device))
        nn.init.trunc_normal_(self.parameter, a=-3, b=3)
    
    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        return self.parameter[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model:int, eps:float=1e-5, device:torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.eps=eps
        self.parameter = nn.Parameter(torch.empty(d_model, dtype=dtype, device=device))
        nn.init.trunc_normal_(self.parameter, a=-3, b=3)
    
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

class SwiGLU(nn.Module):
    def __init__(self, d_model:int, d_ff:int, device:torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_ff = d_ff
        self.w1 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))
        self.w3 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.silu = SiLU()
        nn.init.trunc_normal_(self.w1, a=-3, b=3)
        nn.init.trunc_normal_(self.w2, a=-3, b=3)
        nn.init.trunc_normal_(self.w3, a=-3, b=3)
    
    def forward(self, x: torch.Tensor):
        return einx.dot("b ... d_ff, d_model d_ff -> b ... d_model", 
            self.silu(einx.dot("b ... d_model, d_ff d_model -> b ... d_ff", x, self.w1)) * 
            einx.dot("b ... d_model, d_ff d_model -> b ... d_ff", x, self.w3), 
            self.w2
        )

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta:float, d_k:int, max_seq_len:int, device:torch.device | None = None):
        """
        theta: base frequency.
        Position rotates while different dimension rotates in different speed.
        theta_{i,k} = i / (theta^{(2k-2)/d}) 
        For small k -> rotate fast
        For large k -> rotate slow
        """
        super().__init__()
        # Precompute sin and cos
        assert d_k % 2 == 0
        pos = torch.arange(max_seq_len, device=device)
        inv_freq = theta ** (-torch.arange(start=0, end=d_k, step=2, device=device)/d_k)
        
        angle = einx.multiply("a, b -> a b", pos, inv_freq)

        sin = torch.sin(angle)
        cos = torch.cos(angle)

        self.register_buffer("sin_freq", sin, persistent=False)
        self.register_buffer("cos_freq", cos, persistent=False)
    
    def forward(self, x: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Float[Tensor, "... sequence_length d_k"]
            position (torch.Tensor): Int[Tensor, "... sequence_length"]

        Returns:
            torch.Tensor: Float[Tensor, "... sequence_length d_k"]
        """
        
        x_even = x[...,0::2]
        x_odd = x[...,1::2]
        
        cos = self.cos_freq[position]
        sin = self.sin_freq[position]
        
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd  = x_even * sin + x_odd * cos
        
        x_rot = torch.empty_like(x)
        x_rot[..., 0::2] = x_rot_even
        x_rot[..., 1::2] = x_rot_odd
        return x_rot