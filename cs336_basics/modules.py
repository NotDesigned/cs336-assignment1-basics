import math
from typing import Optional
from collections import OrderedDict

import torch
import torch.nn as nn
from einops import rearrange, einsum
from torch import Tensor
from jaxtyping import Bool, Float, Int
import einx

class Linear(nn.Module):
    def __init__(self, in_features:int, out_features:int, device:torch.device | None =None, dtype:torch.dtype | None =None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype, device=device))
        nn.init.trunc_normal_(self.weight, a=-3, b=3)
    
    def forward(self, x : Tensor) -> Tensor:
        return einx.dot("b ... i, o i -> b ... o", x, self.weight)

class Embedding(nn.Module):
    def __init__(self, num_embedding:int, embedding_dim:int, device:torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.num_embedding = num_embedding
        self.embedding_dim = embedding_dim 
        self.weight = nn.Parameter(torch.empty(num_embedding, embedding_dim, dtype=dtype, device=device))
        nn.init.trunc_normal_(self.weight, a=-3, b=3)
    
    def forward(self, token_ids: torch.LongTensor) -> Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model:int, eps:float=1e-5, device:torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.eps=eps
        self.weight = nn.Parameter(torch.empty(d_model, dtype=dtype, device=device))
        nn.init.trunc_normal_(self.weight, a=-3, b=3)
    
    def forward(self, x: Tensor) -> Tensor: 
        in_dtype=x.dtype
        x = x.float()
        
        RMS = torch.sqrt(einx.mean("... [a]", x**2) + self.eps)
        x = einx.divide("... a, ... -> ... a", x, RMS)
        return einx.multiply("... a, a -> ... a", x, self.weight).to(in_dtype)

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x:Tensor) -> Tensor:
        return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model:int, d_ff:int, device:torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff) # nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.w2 = Linear(d_ff, d_model) # nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))
        self.w3 = Linear(d_model, d_ff) # nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.silu = SiLU()
        nn.init.trunc_normal_(self.w1.weight, a=-3, b=3)
        nn.init.trunc_normal_(self.w2.weight, a=-3, b=3)
        nn.init.trunc_normal_(self.w3.weight, a=-3, b=3)
    
    def forward(self, x: Tensor):
        return einx.dot("b ... d_ff, d_model d_ff -> b ... d_model", 
            self.silu(einx.dot("b ... d_model, d_ff d_model -> b ... d_ff", x, self.w1.weight)) * 
            einx.dot("b ... d_model, d_ff d_model -> b ... d_ff", x, self.w3.weight), 
            self.w2.weight
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
    
    def forward(self, x: Tensor, position: Tensor) -> Tensor:
        """

        Args:
            x (Tensor): Float[Tensor, "... sequence_length d_k"]
            position (Tensor): Int[Tensor, "... sequence_length"]

        Returns:
            Tensor: Float[Tensor, "... sequence_length d_k"]
        """
        
        x_even = x[...,0::2]
        x_odd = x[...,1::2]
        
        # "position d_k"
        cos = self.cos_freq[position]
        sin = self.sin_freq[position]
        
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd  = x_even * sin + x_odd * cos
        
        x_rot = torch.empty_like(x)
        x_rot[..., 0::2] = x_rot_even
        x_rot[..., 1::2] = x_rot_odd
        return x_rot

def Softmax(x: Tensor, dim:int):
    # Subtract the maximum according dim i.
    max_element =  x.amax(dim, keepdim=True)
    
    exp_x = torch.exp(x-max_element)
    
    ret = exp_x / exp_x.sum(dim=dim, keepdim=True)
    return ret 

def scaled_dot_product_attention(x_q:Float[Tensor, "B ... Q D_k"], x_k:Float[Tensor, "B ... K D_k"], 
                                 x_v:Float[Tensor, "B ... K D_v"], mask:Optional[Bool[Tensor, "B ... Q K"]]):
    d_k = x_q.shape[-1]
    score = einx.dot("B ... Q [D_k], B ... K [D_k] -> B ... Q K", x_q, x_k) / math.sqrt(d_k)
    if mask is not None:
        score.masked_fill_(~mask, -torch.inf)
    score = Softmax(score, dim=-1)
    ret = einx.dot("B ... Q [K], B ... [K] D_v -> B ... Q D_v", score, x_v)
    return ret

class MultiHead_Self_Attention(nn.Module):
    def __init__(self, d_model:int, num_heads:int, rope: Optional[RotaryPositionalEmbedding] = None, device:torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        assert d_model % num_heads == 0
        d_k = d_v = d_model // num_heads
        self.q_proj = Linear(d_model, d_k*num_heads)  # nn.Parameter(torch.empty((d_k*num_heads, ), device=device, dtype=dtype))
        self.k_proj = Linear(d_model, d_k*num_heads)  # nn.Parameter(torch.empty((d_k*num_heads, d_model), device=device, dtype=dtype))
        self.v_proj = Linear(d_model, d_v*num_heads) # nn.Parameter(torch.empty((d_v*num_heads, d_model), device=device, dtype=dtype))
        self.output_proj = Linear(d_v*num_heads, d_model) # nn.Parameter(torch.empty((d_model, d_v*num_heads), device=device, dtype=dtype))
        self.num_heads=num_heads
        self.d_model = d_model
        self.rope=rope
    
    def forward(self, x: Float[Tensor, "B ... S D"], token_positions:Optional[Tensor]=None):
        Q = einx.dot("B ... S [D], (H d_k) [D] -> (B H) ... S d_k", x, self.q_proj.weight, H=self.num_heads)
        K = einx.dot("B ... S [D], (H d_k) [D] -> (B H) ... S d_k", x, self.k_proj.weight, H=self.num_heads)
        V = einx.dot("B ... S [D], (H d_v) [D] -> (B H) ... S d_v", x, self.v_proj.weight, H=self.num_heads)
        S = x.shape[-2]
        mask = torch.tril(torch.ones(S, S, dtype=torch.bool)) # Q \times K , k <= q. 
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(0, S, device=x.device)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        ret = scaled_dot_product_attention(Q, K, V, mask=mask)
        ret = einx.id("(B H) ... S d_v -> B ... S (H d_v)", ret, H=self.num_heads)
        ret = einx.dot("B ... S D2, D D2-> B ... S D", ret, self.output_proj.weight)
        return ret

class TransformerBlock(nn.Module):
    def __init__(self, d_model:int, num_heads:int, d_ff:int, theta:float = 10000, max_seq_len:int = 1024, device:torch.device | None = None, dtype: torch.dtype | None = None):
        """
        Args:
            d_model (int): Dimensionality of embedding in attention
            num_heads (int): Number of heads in the multihead attention
            d_ff (int): Dim of FeedForward inner network
        """
        super().__init__()
        assert d_model % num_heads == 0
        d_k = d_model // num_heads
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbedding(theta=theta, d_k=d_k, max_seq_len=max_seq_len, device=device)
        self.attn = MultiHead_Self_Attention(d_model, num_heads, rope=self.rope, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        
    def forward(self, x: Float[Tensor, "B ... S D"]) -> Tensor:
        # PreNorm
        y = self.attn(self.ln1(x)) + x
        y = self.ffn(self.ln2(y)) + y
        return y

class TransformerLM(nn.Module):
    def __init__(self, vocab_size:int, num_layers:int, d_model:int, num_heads:int, d_ff:int, content_length:int, rope_theta:float = 10000, device:torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.token_embeddings = Embedding(num_embedding=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        self.layers = torch.nn.Sequential(OrderedDict({
            f'{i}':TransformerBlock(d_model, num_heads, d_ff, rope_theta, content_length, device, dtype)
            for i in range(num_layers)
        }))
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, token_indices:Int[Tensor, "B S"]) -> Float[Tensor, "B S V"]:
        embeddings = self.token_embeddings(token_indices)
        ret = self.layers(embeddings) # B S D
        return self.lm_head(self.ln_final(ret)) # B S D, V D -> B S V

def calc_params_flops(V:int,S:int,L:int,D:int,H:int,D_:int) -> tuple[int, int, dict, dict]:
    # for flops, each transformer block is 6 SDD' + 8 SD^2 + 4 S^2D
    flops = L*(6*S*D*D_+8*S*D*D+4*S*S*D)
    # output embedding
    flops += 2*S*D*V
    # We ignore token embedding and softmax for flops calculation
    
    # transformer blocks
    params = L* (
        2 * D # 2*RMSNorm
        + 0 # rope has no params
        + 4 * D*D
        + 3 * D*D_ # SwiGLU
    )
    # Token embedding
    params += V*D
    # Final Norm, linear projection
    params += D + D*V
    full_params = {
        'FFN': 3* L * D * D_,
        'Attention': 4 * L* D*D,
        'Embeddings': 2 *D*V
    }
    full_flops = {
        'FFN': L * 6 * D * D_ * S,
        'Attention': L * (8 * S * D * D + 4 * S * S * D)
    }
    return params, flops, full_params, full_flops

