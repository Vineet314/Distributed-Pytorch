'''
This script builds an LLM model based on the user's CLI inputs.
Credits:
    - This code is highly inspired by Andrej Karpathy's work on his nanoGPT : https://github.com/karpathy/nanoGPT/
    - Thanks to Vizuara AI Labs for their detailed explanations of MLA : https://youtu.be/m1x8vA_Tscc

Available settings to choose from : 
1. Attention Type (with  KV caching): 
   - Multi Head Attention (mha)
   - Multi Query Attention (mqa)
   - Grouped Query Attention (gqa)
   - Multi Head Latent Attention (mla)
   - (Work in progress) Flash Multi Head Latent Attention (fmla)

2. Positional Encodings:
   - Learnable PE
   - Sinusoidal PE
   - Rotary PE (RoPE)

3. Feed Forward Layers:
   - Dense Network Layer (moe=False): Fully Connected, MLP layer
   - Sparse Network Layer (moe=True): Mixture of Exprts
        - Load Balancing with Auxilary Loss function (aux_free = False) 
        - Shared Expert Isolation                    (n_shared = 0) 
        - Fine Grained Expert Segmentation           (set up_dim, n_exp, n_act accordingly)
        - Aux Loss Free Load Balancing               (aux_free = True)  
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal
from dataclasses import dataclass 

@dataclass
class LLMconfig:
    # token params
    vocab_size : int
    block_size : int
    n_embd : int
    pos_emb : str | Literal['learn','sin','rope']

    # Neural Network
    up_dim  : int
    non_linearity : str | Literal['elu','lrelu','relu', 'gelu', 'swish', 'mish', 'silu', 'selu','celu','tanh','sigmoid']
    dropout : float
    n_layer : int

    # MoE
    moe : bool

    n_exp : int
    n_shared : int  
    n_act : int      ### INCLUDES THE SHARED EXPERTS
    coeff : float

    aux_free : bool
    alpha : float   # complementry aux loss coeff
    gamma: float    # bias update speed
    
    # Attention
    attn : str | Literal['mha', 'mqa', 'gqa', 'mla']
    # kv_cache : bool
    n_head : int
    n_kv_heads : int 
        # Only for mla 
    q_latent_dim  : int | None
    kv_latent_dim : int | None
    rope_head_dim : int | None

    @staticmethod
    def apply_rotary_emb(x:torch.Tensor, freqs_cis:torch.Tensor)->torch.Tensor:
        ''' Applies RoPE to either the query or the key whose embeddings are to be rotated two at a time.'''

        # H below is either the number of total query heads(nh)
        # hs is the embedding dimension for the query/key, given by n_embd//nh
        B,T,H,_ = x.size()
        x_ = x.float().reshape(B, T, H, -1, 2)          # (B, T, H, hs)       -> (B, T, H, hs//2, 2)    -> creates the two pairs in the embd dim
        x_re, x_im = x_.unbind(-1)                      # (B, T, H, hs//2, 2) -> (B, T, H, hs//2)       -> splits those two pairs
        freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2) # (T, hs//2)          -> (1, T, 1, hs//2)       -> this has dtype complex64, so last dim has two parts, real and imaginary
        # freqs_cis has two parts : real and imaginary (cosθ, sinθ)
        # import code ; code.interact(local=locals())
        # Perform the rotation (vector * rotation matrix)
        x_re_out = x_re*freqs_cis.real - x_im*freqs_cis.imag    # (B, T, H, hs//2) * (1, T, 1, hs//2) - (B, T, H, hs//2) * (1, T, 1, hs//2) -> (B, T, H, hs//2)
        x_im_out = x_re*freqs_cis.imag + x_im*freqs_cis.real    # (B, T, H, hs//2) * (1, T, 1, hs//2) + (B, T, H, hs//2) * (1, T, 1, hs//2) -> (B, T, H, hs//2)
        
        # Stack the real and imaginary parts back together
        x_out = torch.stack([x_re_out, x_im_out], dim=-1).flatten(3) # (B, T, H, hs//2), (B, T, H, hs//2) -> (B, T, H, hs)

        return x_out.type_as(x)

class GQA(nn.Module):
    """ Grouped-Query Attention with or without RoPE """

    def __init__(self, config:LLMconfig):
        super().__init__()
        if config.attn == 'mha' : config.n_kv_heads = config.n_head
        elif config.attn == 'mqa' : config.n_kv_heads = 1
        else : assert config.n_head % config.n_kv_heads == 0, "n_head must be divisible by n_kv_heads"
        
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.config = config
        self.head_size = config.n_embd // config.n_head

        # k,q,v in a btach
        self.c_attn = nn.Linear(config.n_embd, config.n_embd + 2 * config.n_kv_heads * self.head_size)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout  = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x:torch.Tensor, freqs_cis:torch.Tensor|None, kv_cache=None):
        B, T, C = x.size()
        nh, nkvh, hs = self.config.n_head , self.config.n_kv_heads, self.head_size

        q_proj_size = C # n_embd
        kv_proj_size = nkvh * hs
        q, k, v = self.c_attn(x).split([q_proj_size, kv_proj_size, kv_proj_size], dim=2)
        q:torch.Tensor = q.view(B, T, nh, hs) # (B, T, nh, hs)
        k:torch.Tensor = k.view(B, T, nkvh, hs) # (B, T, n_kvh, hs)
        v:torch.Tensor = v.view(B, T, nkvh, hs).transpose(1, 2) # (B, n_kvh, T, hs)

        if self.config.pos_emb == 'rope':
        # Apply RoPE
            q = LLMconfig.apply_rotary_emb(q, freqs_cis)
            k = LLMconfig.apply_rotary_emb(k, freqs_cis)

        q,k = q.transpose(1, 2), k.transpose(1, 2) # (B, nh, T, hs) # (B, n_kvh, T, hs)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)

        updated_kv_cache = (k, v)

        if nkvh != nh:
            num_repeats = nh // nkvh
            k = k.repeat_interleave(num_repeats, dim=1)
            v = v.repeat_interleave(num_repeats, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.config.dropout if self.training else 0, is_causal=True)
        y = y.transpose(1,2).contiguous().view(B,T,C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        return y, updated_kv_cache

class NaiveMHLA(nn.Module):
    """ A fully parallel implementation of the MHLA algorithm without the RoPE. No for loops."""
    def __init__(self, config:LLMconfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "num of heads must be a divisor of n_embd"
        self.head_size = config.n_embd // config.n_head
        self.config = config

        self.W_dq  = nn.Linear(config.n_embd,        config.q_latent_dim,  bias=False)
        self.W_uq  = nn.Linear(config.q_latent_dim,  config.n_embd,        bias=False)
        self.W_dkv = nn.Linear(config.n_embd,        config.kv_latent_dim, bias=False)
        self.W_uk  = nn.Linear(config.kv_latent_dim, config.n_embd,        bias=False)
        self.W_uv  = nn.Linear(config.kv_latent_dim, config.n_embd,        bias=False)
        self.W_o   = nn.Linear(config.n_embd,        config.n_embd,        bias=False)
        
        # self.ln  = nn.LayerNorm(config.kv_latent_dim)
        self.dropout = nn.Dropout(config.dropout)

        self.register_buffer('_k_absorbed_inference', None)
        self.register_buffer('_v_absorbed_inference', None)

    def _precompute_absorbed_matrices(self):
        """Precomputes k_absorbed and v_absorbed for efficient inference."""
        # Just to be safe
        if (self._k_absorbed_inference is not None) and (self._v_absorbed_inference is not None):
            return 
        
        nh , n_kvl, hs = self.config.n_head, self.config.kv_latent_dim, self.head_size
        with torch.no_grad():
            self._k_absorbed_inference = (self.W_dq.weight.T @ self.W_uq.weight.T  @ self.W_uk.weight).view(nh, hs, n_kvl).unsqueeze(0)
            self._v_absorbed_inference = (self.W_uv.weight.T @ self.W_o.weight.T).view(n_kvl, nh, hs).transpose(0,1).unsqueeze(0)    

    def forward(self, x:torch.Tensor, freqs_cis:torch.Tensor|None, kv_cache=None) -> tuple[torch.Tensor, torch.Tensor]:

        B, T, C = x.size()
        nh, n_kvl, hs = self.config.n_head, self.config.kv_latent_dim, self.config.n_embd//self.config.n_head

        # k_eff and v_eff based on training or inference
        if self.training:
            k_eff = (self.W_dq.weight.T @ self.W_uq.weight.T  @ self.W_uk.weight).view(nh, hs, n_kvl).unsqueeze(0)
            v_eff = (self.W_uv.weight.T @ self.W_o.weight.T).view(n_kvl, nh, hs).transpose(0,1).unsqueeze(0)
        else:
            if (self._k_absorbed_inference is None) or (self._v_absorbed_inference is None):
                self._precompute_absorbed_matrices()
            k_eff = self._k_absorbed_inference
            v_eff = self._v_absorbed_inference
        
        new_c_kv = self.W_dkv(x) # down projection : (B,T,C) -> (B,T,n_kvl)

        if kv_cache is None:
            c_kv = new_c_kv # (B,T,n_kvl) ; initiate cache
        else:
            c_kv = torch.cat([kv_cache, new_c_kv], dim=1) # append cache
        
        updated_kv_cache = c_kv

        T_full = c_kv.size(1) # Current total sequence length (including cache)

        q:torch.Tensor = self.W_uq(self.W_dq(x)) # query projection : (B,T,C) -> (B,T,n_ql) -> (B,T,C)
        q = q.view(B, T, nh, hs).transpose(1, 2) # (B,T,C) -> (B,T,nh,hs) -> (B, nh, T, hs)

        attn:torch.Tensor = (q @ k_eff @ c_kv.transpose(1,2).unsqueeze(1)) / math.sqrt(hs)

        query_indices = torch.arange(T, device=x.device).unsqueeze(1) + (T_full - T)
        key_indices   = torch.arange(T_full, device=x.device).unsqueeze(0)
        mask = (query_indices >= key_indices).unsqueeze(0).unsqueeze(0) # (1,1,T,T_full)
        attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = self.dropout(F.softmax(attn, dim=-1)) # (B, nh, T, T_full)

        # final output : attn @ C_kv @ v_abs 
        # (B, nh, T, T) * (B, 1, T, n_kvl) * (1, nh, n_kvl, hs) = (B, nh, T, hs)
        y:torch.Tensor = attn @ c_kv.unsqueeze(1) @ v_eff #(B, nh, T, hs)
        y = self.dropout(y.transpose(1,2).contiguous().view(B,T,C))

        return y, updated_kv_cache

class FullMHLA(nn.Module):
    """
    A fully parallel implementation of Multi-Head Latent Attention (MLA)
    with Decoupled Rotary Position Embeddings (RoPE), as described in DeepSeek-V2.
    """
     
    def __init__(self, config:LLMconfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "num of heads must be a divisor of n_embd"
        self.config = config
        self.W_dq  = nn.Linear(config.n_embd, config.q_latent_dim , False)
        self.dropout = nn.Dropout(config.dropout)
        
        # (NoPE)
        self.head_size = config.n_embd // config.n_head
        self.W_uq  = nn.Linear(config.q_latent_dim , config.n_embd, False)
        self.W_dkv = nn.Linear(config.n_embd, config.kv_latent_dim, False)
        self.W_uk  = nn.Linear(config.kv_latent_dim, config.n_embd, False)
        self.W_uv  = nn.Linear(config.kv_latent_dim, config.n_embd, False)

        # (RoPE)
        self.W_qr  = nn.Linear(config.q_latent_dim, config.n_head * config.rope_head_dim,  False)
        self.W_kr  = nn.Linear(config.n_embd, config.rope_head_dim, False)

        # (Out)
        self.W_o = nn.Linear(config.n_embd, config.n_embd ,False)

        # Absroption during inference
        self.register_buffer('_k_absorbed_inference', None, persistent=False)
        self.register_buffer('_v_absorbed_inference', None, persistent=False)

    def _precompute_absorbed_matrices(self):
        """Precomputes k_absorbed and v_absorbed for efficient inference."""
        # Just to be safe
        if (self._k_absorbed_inference is not None) and (self._v_absorbed_inference is not None):
            return 
        
        nh, nlkv, hs, nlq = self.config.n_head, self.config.kv_latent_dim, self.config.n_embd//self.config.n_head, self.config.q_latent_dim
        with torch.no_grad():
            self._k_absorbed_inference = (self.W_uq.weight.view(1,nlq,nh,hs).transpose(1,2) @ self.W_uk.weight.view(1,nh,hs,nlkv))
            self._v_absorbed_inference = (self.W_uv.weight.T @ self.W_o.weight.T).view(nlkv, nh, hs).transpose(0,1).unsqueeze(0)    

    def forward(self, x:torch.Tensor, freqs_cis:torch.Tensor|None, kv_cache=None):
        B,T,C = x.size()
        nh,nlkv,nlq = self.config.n_head, self.config.kv_latent_dim, self.config.q_latent_dim
        hs = C//nh
        dhr = self.config.rope_head_dim
        
        c_q:torch.Tensor = self.W_dq(x)  # (B,T,nlq)

 #------------ NoPE--------------

        # Define the absorbed matrices
        if self.training:
            k_eff = (self.W_uq.weight.view(1,nlq,nh,hs).transpose(1,2) @ self.W_uk.weight.view(1,nh,hs,nlkv))
            v_eff = (self.W_uv.weight.T @ self.W_o.weight.T).view(nlkv, nh, hs).transpose(0,1).unsqueeze(0)  
        else:
            if (self._k_absorbed_inference is None) or (self._v_absorbed_inference is None):
                self._precompute_absorbed_matrices()
            k_eff = self._k_absorbed_inference
            v_eff = self._v_absorbed_inference

        new_c_kv = self.W_dkv(x) # down projection : (B,T,C) -> (B,T,n_kvl)

        if kv_cache is None: # first pass
            c_kv = new_c_kv # (B,T,n_kvl) ; initiate cache
        else:
            c_kv = torch.cat([kv_cache['c_kv'], new_c_kv], dim=1) # append cache

        T_full = c_kv.size(1) # Current total sequence length (including cache)

        attn_c = c_q.unsqueeze(1) @ k_eff @ c_kv.transpose(-1,-2).unsqueeze(1)

 #------------ RoPE--------------

        c_kr:torch.Tensor = self.W_kr(x).unsqueeze(2)        # (B,T,1,dhr)
        k_r = LLMconfig.apply_rotary_emb(c_kr, freqs_cis).transpose(1,2)  # (B,1,T,dhr), to be cached

        # initate KV cache
        if kv_cache is not None:
            k_r = torch.cat([kv_cache['k_r'], k_r], dim=2)

        c_qr:torch.Tensor = self.W_qr(c_q).view(B,T,nh,dhr) # (B,T,nh,dhr) # because rope expects (B,T,H,dh)
        q_r = LLMconfig.apply_rotary_emb(c_qr, freqs_cis).transpose(1,2) # (B,nh,T,dhr)
        
        attn_r = q_r @ k_r.transpose(-1,-2)

 #------------ Out--------------

        attn = (attn_c + attn_r)/math.sqrt(hs+dhr)

        query_indices = torch.arange(T, device=x.device).unsqueeze(1) + (T_full - T)
        key_indices = torch.arange(T_full, device=x.device).unsqueeze(0)
        mask = (query_indices >= key_indices).unsqueeze(0).unsqueeze(0) # (1,1,T,T_full)
        attn = attn.masked_fill(mask == 0, float('-inf')) 

        attn = self.dropout(F.softmax(attn, dim=-1)) # (B, nh, T, T_full)

        # final output : attn @ C_kv @ v_abs 
        # (B, nh, T, T) * (B, 1, T, n_kvl) * (1, nh, n_kvl, hs) = (B, nh, T, hs)
        y:torch.Tensor = attn @ c_kv.unsqueeze(1) @ v_eff #(B, nh, T, hs)
        y = self.dropout(y.transpose(1,2).contiguous().view(B,T,C))

        updated_kv_cache = {'c_kv': c_kv, 'k_r': k_r}

        return y, updated_kv_cache

class Attention(nn.Module):
    """ Routes the attention mechanism according to the config"""

    def __init__(self, config:LLMconfig):
        super().__init__()
        self.config = config
        if config.attn in ('mha','mqa','gqa'):
            self.attn = GQA(config)
        
        elif config.attn == 'mla':
            if config.pos_emb != 'rope':
                self.attn = NaiveMHLA(config)
            else:
                self.attn = FullMHLA(config)
                
    def forward(self, x:torch.Tensor, freqs_cis:torch.Tensor|None = None, kv_cache=None):
        return self.attn(x, freqs_cis, kv_cache)

class MLP(nn.Module):
    """ A simple feed-forward network block. """
    def __init__(self, config: LLMconfig):
        super().__init__()
        non_linearity_map = {
            'relu': nn.ReLU(), 'gelu': nn.GELU(), 'swish': nn.SiLU(), 'mish': nn.Mish(),
            'silu': nn.SiLU(), 'selu': nn.SELU(), 'celu': nn.CELU(), 'elu': nn.ELU(),
            'lrelu': nn.LeakyReLU(negative_slope=0.01), 'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid()}

        self.c_fc = nn.Linear(config.n_embd, config.up_dim, bias=False)
        self.non_linearity = non_linearity_map.get(config.non_linearity, nn.GELU())
        self.c_proj = nn.Linear(config.up_dim, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.non_linearity(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Expert(nn.Module):
    """ A single feed-forward network expert. """
    def __init__(self, config:LLMconfig):
        super().__init__()
        self.expert = MLP(config)
        
    def forward(self, x):
        return self.expert(x)

class MoE(nn.Module):
    '''
    This class implements the DeepSeekMoE layer, featuring shared and routed experts.
    It uses an Auxiliary-Loss-Free load balancing strategy with a dynamic bias term.
    Ref: https://arxiv.org/pdf/2412.19437
    '''

    def __init__(self, config: LLMconfig):
        super().__init__()
        self.config = config
        
        # first `n_shared` are shared, the rest are routed
        self.n_shared = config.n_shared
        self.n_routed = config.n_exp - config.n_shared
        
        # Number of experts to activate from the ROUTED pool
        self.n_act_routed = config.n_act - config.n_shared
        assert self.n_act_routed > 0, "Number of active experts must be greater than shared experts"

        self.experts = nn.ModuleList([Expert(config) for _ in range(config.n_exp)])
        self.gate = nn.Linear(config.n_embd, self.n_routed, bias=False)
        
        if config.aux_free:
            self.register_buffer('expert_bias', torch.zeros(self.n_routed))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ Forward pass for the DeepSeekMoE layer with Aux-Loss-Free Balancing. """
        B, T, C = x.shape
        x_flat = x.view(-1, C)  # Shape: (B*T, C)
        n_tokens = x_flat.shape[0]

        # ___________ SHARED EXPERT PATH ___________

        shared_output = torch.zeros_like(x_flat)
        if self.n_shared > 0:
            for i in range(self.n_shared):
                shared_output += self.experts[i](x_flat) # bypass the router

        #  ___________ ROUTED EXPERT PATH ___________

        router_logits = self.gate(x_flat)

        if self.config.aux_free:        
            # Add Bias and then select topk
            biased_router_logits = router_logits + self.expert_bias
            topk_biased_logits, topk_indices = torch.topk(biased_router_logits, self.n_act_routed, dim=1)

            # Gating weights are based on un-biased logits
            topk_original_logits = torch.gather(router_logits, 1, topk_indices) 
            topk_gates = F.softmax(topk_original_logits, dim=1)

            # Calculate expert load and update bias during training only
            if self.training:
                with torch.no_grad(): # Ensure this logic doesn't affect gradients
                    ones = torch.ones_like(topk_indices, dtype=x_flat.dtype)
                    fi_counts = torch.zeros(self.n_routed, device=x.device).scatter_add_(0, topk_indices.flatten(), ones.flatten())
                    fi = fi_counts / n_tokens
                    ideal_load = 1.0 / self.n_routed

                    overloaded_mask = fi > ideal_load
                    underloaded_mask = fi < ideal_load

                    self.expert_bias[overloaded_mask] -= self.config.gamma
                    self.expert_bias[underloaded_mask] += self.config.gamma
            
            # COMPLEMENTARY LOSS
            router_probs = F.softmax(router_logits, dim=1)
            pi = router_probs.mean(dim=0)
            # We need fi for the loss, but only want to calculate it once.
            # During inference, we can skip the calculation if not already done.
            if not self.training:
                ones = torch.ones_like(topk_indices, dtype=torch.float)
                fi_counts = torch.zeros(self.n_routed, device=x.device).scatter_add_(0, topk_indices.flatten(), ones.flatten())
            fi = fi_counts / n_tokens
            # complementary_loss
            aux_loss = self.config.alpha * self.n_routed * torch.sum(pi*fi)

        else:
            router_probs = F.softmax(router_logits, dim=1)
            pi = router_probs.mean(dim=0)
            
            topk_logits, topk_indices = torch.topk(router_logits, self.n_act_routed, dim=1)
            ones = torch.ones_like(topk_indices, dtype=torch.float)
            fi_counts = torch.zeros(self.n_routed, device=x.device).scatter_add_(0, topk_indices.flatten(), ones.flatten())
            fi = fi_counts / n_tokens

            aux_loss = self.config.coeff * self.n_routed * torch.sum(pi * fi)

            topk_gates = F.softmax(topk_logits, dim=1)  

        # Dispatch
        routed_output = torch.zeros_like(x_flat)

        for i in range(self.n_routed):
            token_indices, topk_slot = (topk_indices == i).nonzero(as_tuple=True)
            if token_indices.numel() > 0:
                tokens_for_expert = x_flat[token_indices]
                gates_for_expert = topk_gates[token_indices, topk_slot].unsqueeze(1)

                # access the expert using an offset of `n_shared`
                expert_output = self.experts[i + self.n_shared](tokens_for_expert)
                
                weighted_output = expert_output * gates_for_expert
                routed_output.index_add_(0, token_indices, weighted_output)
        
        # combine to output
        y = (shared_output + routed_output).view(B, T, C)
        return y, aux_loss

class Block(nn.Module):
    """ A single Transformer block combining attention and MLP. """
    def __init__(self, config:LLMconfig):
        super().__init__()
        self.is_moe = config.moe
        self.attn = Attention(config)
        self.ln1  = nn.LayerNorm(config.n_embd)
        self.ln2  = nn.LayerNorm(config.n_embd)
        if config.moe:
            self.moe = MoE(config)
        else:
            self.mlp = MLP(config)

    def forward(self, x:torch.Tensor, freqs_cis:torch.Tensor|None = None, kv_cache=None):
        # Layer Norm + Attention
        attn_output, updated_kv_cache = self.attn.forward(self.ln1(x), freqs_cis, kv_cache)
        x = x + attn_output

        if self.is_moe: 
            moe_output, aux_loss = self.moe(self.ln2(x))
            x = x + moe_output
        else:
            aux_loss = 0.0
            x = x + self.mlp(self.ln2(x))

        return x, updated_kv_cache, aux_loss

class LLM(nn.Module):
    """ A simple Large language model """
    def __init__(self, config:LLMconfig):
        super().__init__()
        self.config = config
        self.head_size = config.n_embd//config.n_head
        self.tkn_emb = nn.Embedding(config.vocab_size, config.n_embd)
        if config.pos_emb == 'learn':
            self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        elif config.pos_emb == 'sin':
            pos_emb  = torch.zeros(config.block_size, config.n_embd)
            position = torch.arange(0, config.block_size, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, config.n_embd, 2).float() * (-math.log(10000.0) / config.n_embd))
            pos_emb[:, 0::2] = torch.sin(position * div_term)
            pos_emb[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pos_emb', pos_emb)
        elif config.pos_emb == 'rope':
            self.register_buffer("freqs_cis", self._precompute_freqs_cis())
    
        self.transformer = nn.ModuleDict(dict(
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)))
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.tkn_emb.weight = self.lm_head.weight # weight tying
        self.apply(self._init_weights)

    def _precompute_freqs_cis(self):
        """Precomputes the rotary frequencies for RoPE."""
        d = self.config.rope_head_dim if self.config.attn=='mla' else self.head_size
        assert d % 2 == 0, "head dimension must be even"
        
        theta = 1.0 / (10000.0 ** (torch.arange(0, d, 2).float() / d)) # 1.0 / (base^(2i/d))
        seq = torch.arange(self.config.block_size)
        freqs = torch.outer(seq, theta)

        # Convert to complex numbers: r * e^(i*theta)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis
        
    def _init_weights(self, module):
        """Initializes model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        """Returns the total number of parameters and active parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        
        active_params = 0

        active_params += self.tkn_emb.weight.numel()      # embeddings
        if self.config.pos_emb == 'learn': active_params += self.pos_emb.weight.numel()
        active_params += self.transformer.ln_f.weight.numel() + self.transformer.ln_f.bias.numel()

        for block in self.transformer.h:
            active_params += sum(p.numel() for p in block.attn.parameters())   # ----|
            active_params += sum(p.numel() for p in block.ln1.parameters())    #     |---> Always active
            active_params += sum(p.numel() for p in block.ln2.parameters())    # ----|

            if block.is_moe:

                active_params += sum(p.numel() for p in block.moe.gate.parameters())                # ----|
                for i in range(block.moe.n_shared):                                                 #     |---> Always active
                    active_params += sum(p.numel() for p in block.moe.experts[i].parameters())      # ----|

                if block.moe.n_routed > 0:
                    # Calculate params for one routed expert, multiply by the number of active ones
                    params_per_routed_expert = sum(p.numel() for p in block.moe.experts[block.moe.n_shared].parameters())
                    active_params += block.moe.n_act_routed * params_per_routed_expert
            
            else: # In case a block is not MoE
                active_params += sum(p.numel() for p in block.mlp.parameters())

        return n_params, active_params

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}]

        # Create AdamW optimizer and use the fused version if it is available
        try:
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, fused=True)
            # print("Using Fused AdamW")
        except:
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate)
        return optimizer

    def forward(self, idx: torch.Tensor, targets=None, kv_caches=None):
        B, T = idx.size()
        start_pos = 0

        if kv_caches is not None and kv_caches[0] is not None:
            if self.config.attn in ('mha', 'mqa', 'gqa'):
                start_pos = kv_caches[0][0].shape[-2]
            elif self.config.attn == 'mla':
                if self.config.pos_emb == 'rope':
                    start_pos = kv_caches[0]['c_kv'].shape[1]
                else:
                    start_pos = kv_caches[0].shape[1]

        tkn_emb = self.tkn_emb(idx)  # Shape: (B, T, n_embd)
        
        x = tkn_emb # Default value for x
        freqs_cis = None

        if self.config.pos_emb == 'rope':
            freqs_cis = self.freqs_cis[start_pos : start_pos + T]
        
        elif self.config.pos_emb == 'learn':
            pos = torch.arange(start_pos, start_pos + T, dtype=torch.long, device=idx.device)
            x = tkn_emb + self.pos_emb(pos)

        elif self.config.pos_emb == 'sin':
            pos = torch.arange(start_pos, start_pos + T, dtype=torch.long, device=idx.device)
            x = tkn_emb + self.pos_emb[pos]

        x = self.transformer.drop(x)

        if kv_caches is None:
            kv_caches = [None] * self.config.n_layer
        
        updated_kv_caches = []
        total_aux_loss = 0.0
        for i, block in enumerate(self.transformer.h):
            # The block now returns an auxiliary loss from the MoE layer
            x, updated_kv_cache, aux_loss = block(x, freqs_cis, kv_caches[i])
            updated_kv_caches.append(updated_kv_cache)
            total_aux_loss += aux_loss

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            main_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            # Add the accumulated auxiliary loss to the main loss
            # We divide by the number of layers because loss is accumulated from each MoE block
            loss = main_loss + total_aux_loss / self.config.n_layer
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss, updated_kv_caches
    
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, topk: int | None = None):
        self.eval()
        kv_caches = [None] * self.config.n_layer

        for i in range(max_new_tokens):
            if i == 0:
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
                input_for_forward = idx_cond
            else:
                input_for_forward = idx[:, -1:]

            if kv_caches[0] is not None:
                if self.config.attn in ('mha', 'mqa', 'gqa'):
                    cache_len = kv_caches[0][0].shape[-2]
                elif self.config.attn == 'mla':
                     cache_len = kv_caches[0]['c_kv'].shape[1] if self.config.pos_emb == 'rope' else kv_caches[0].shape[1]

                if cache_len >= self.config.block_size:
                    # Keep the most recent (block_size - 1) tokens to make space for the new one
                    keep_len = self.config.block_size - 1
                    for layer_idx in range(self.config.n_layer):
                        layer_cache = kv_caches[layer_idx]
                        if self.config.attn in ('mha', 'mqa', 'gqa'):
                            k, v = layer_cache
                            kv_caches[layer_idx] = (k[..., -keep_len:, :], v[..., -keep_len:, :])
                        elif self.config.attn == 'mla':
                            if self.config.pos_emb == 'rope':
                                layer_cache['c_kv'] = layer_cache['c_kv'][:, -keep_len:, :]
                                layer_cache['k_r']  = layer_cache['k_r'][:, :, -keep_len:, :] # Seq len is dim 2
                            else: # c_kv
                                kv_caches[layer_idx] = layer_cache[:, -keep_len:, :]

            # The forward pass now returns three items; we only need logits and caches for generation
            logits, _, kv_caches = self.forward(input_for_forward, kv_caches=kv_caches)
            logits = logits[:, -1, :]

            if temperature > 0:
                logits = logits / temperature
            if topk is not None:
                v, _ = torch.topk(logits, min(topk, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        self.train()
        return idx