'''
This scripts builds upon the Naive MHLA implemetnation by implementing the Rotray Positional Encodings (RoPE)
The script is implemented as per the DeepSeek's introduction of the Decoupled Rotray Postional Encodings in 
DeepSeek V2 : https://arxiv.org/abs/2405.04434
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LLMconfig:
    # hyperparameters
    block_size : int # what is the maximum context length for predictions?
    vocab_size : int
    q_latent_dim : int 
    kv_latent_dim : int 
    rope_head_dim : int 
    n_embd : int 
    n_head : int 
    n_layer: int 
    dropout: float 
   
    @staticmethod
    def apply_rotary_emb(x:torch.Tensor, freqs_cis:torch.Tensor)->torch.Tensor:
        ''' Applies RoPE to either the query or the key whose embeddings are to be rotated two at a time.'''

        # H below is either the number of total query heads(nh)
        # hs is the embedding dimension for the query/key, given by n_embd//nh
        B,T,H,_ = x.size()
        x_ = x.float().reshape(B, T, H, -1, 2)          # (B, T, H, hs)       -> (B, T, H, hs//2, 2)    -> creates the two pairs in the embd dim
        x_re, x_im = x_.unbind(-1)                      # (B, T, H, hs//2, 2) -> (B, T, H, hs//2)       -> splits those two pairs
        freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2) # (T, hs//2)          -> (1, T, 1, hs//2)       -> this has dtype complex64, so last dim has two parts, real and imaginary
        # freq_cis has two parts : real and imaginary (cosθ, sinθ)
        
        # Perform the rotation (vector * rotation matrix)
        x_re_out = x_re*freqs_cis.real - x_im*freqs_cis.imag    # (B, T, H, hs//2) * (1, T, 1, hs//2) - (B, T, H, hs//2) * (1, T, 1, hs//2) -> (B, T, H, hs//2)
        x_im_out = x_re*freqs_cis.imag + x_im*freqs_cis.real    # (B, T, H, hs//2) * (1, T, 1, hs//2) + (B, T, H, hs//2) * (1, T, 1, hs//2) -> (B, T, H, hs//2)
        
        # Stack the real and imaginary parts back together
        x_out = torch.stack([x_re_out, x_im_out], dim=-1).flatten(3) # (B, T, H, hs//2), (B, T, H, hs//2) -> (B, T, H, hs)

        return x_out.type_as(x)

class CausalSelfAttention(nn.Module):
    """
    A fully parallel implementation of Multi-Head Latent Attention (MLA)
    with Decoupled Rotary Position Embeddings (RoPE), as described in DeepSeek-V2.
    """
     
    def __init__(self, config:LLMconfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "num of heads must be a divisor of n_embd"
        self.config = config
        self.W_dq  = nn.Linear(config.n_embd, config.q_latent_dim , False)
        # self.ln = nn.LayerNorm()
        self.dropout = nn.Dropout(config.dropout)
        
        # (NoPE)
        self.head_size = config.n_embd // config.n_head
        # self.W_uq  = nn.Linear(config.q_latent_dim , config.n_embd, False)
        self.W_uq  = nn.Parameter(0.01*torch.randn(1,config.n_head, config.q_latent_dim, self.head_size), requires_grad=True)
        self.W_dkv = nn.Linear(config.n_embd, config.kv_latent_dim, False)
        # self.W_uk  = nn.Linear(config.kv_latent_dim, config.n_embd, False)
        self.W_uk  = nn.Parameter(0.01*torch.randn(1, config.n_head, self.head_size, config.kv_latent_dim), requires_grad=True)
        # self.W_uv  = nn.Linear(config.kv_latent_dim, config.n_embd, False)
        self.W_uv  = nn.Parameter(0.01*torch.randn(config.kv_latent_dim, config.n_embd), requires_grad=True)

        # (RoPE)
        self.W_qr  = nn.Linear(config.q_latent_dim, config.n_head * config.rope_head_dim,  False)
        self.W_kr  = nn.Linear(config.n_embd, config.rope_head_dim, False)

        # Output projection
        # self.W_o = nn.Linear(config.n_embd, config.n_embd ,False)
        self.W_o = nn.Parameter(0.01*torch.randn(config.n_embd, config.n_embd), requires_grad=True)

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
            self._k_absorbed_inference = (self.W_uq @ self.W_uk)
            self._v_absorbed_inference = (self.W_uv @ self.W_o).view(1, nh, nlkv, hs)

    def forward(self, x:torch.Tensor, freqs_cis:torch.Tensor, kv_cache=None):
        B,T,C = x.size()
        nh,nlkv,nlq = self.config.n_head, self.config.kv_latent_dim, self.config.q_latent_dim
        hs = C//nh
        dhr = self.config.rope_head_dim
        
        c_q:torch.Tensor = self.W_dq(x)  # (B,T,nlq)

#------------ NoPE--------------

        # Define the absorbed matrices
        if self.training:
            k_eff = (self.W_uq @ self.W_uk)
            v_eff = (self.W_uv @ self.W_o).view(1, nh, nlkv, hs)
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

class MLP(nn.Module):
    """ A simple feed-forward network block. """
    def __init__(self, config: LLMconfig):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
class Block(nn.Module):
    """ A single Transformer block combining attention and MLP. """
    def __init__(self, config: LLMconfig):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp  = MLP(config)
        self.ln1  = nn.LayerNorm(config.n_embd)
        self.ln2  = nn.LayerNorm(config.n_embd)

    def forward(self, x, freqs_cis, kv_cache=None):
        # Self-attention with residual connection
        attn_output, updated_kv_cache = self.attn(self.ln1(x), freqs_cis, kv_cache=kv_cache)
        x = x + attn_output
        # Feed-forward network with residual connection
        x = x + self.mlp(self.ln2(x))
        return x, updated_kv_cache

class LLM(nn.Module):
    """ A simple Large language model """
    def __init__(self, config:LLMconfig):
        super().__init__()
        self.config = config

        self.tkn_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.transformer = nn.ModuleDict(dict(
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.tkn_emb.weight  = self.lm_head.weight # Weight tying

        # Precompute RoPE frequencies
        self.register_buffer("freqs_cis", self._precompute_freqs_cis())
        self.apply(self._init_weights)
    
    def _precompute_freqs_cis(self):
        """Precomputes the rotary frequencies for RoPE."""
        d = self.config.rope_head_dim
        assert d % 2 == 0, "rope_head_dim must be even"
        
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

    def configure_optimizers(self, weight_decay, learning_rate, device, prints=False):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        # Create AdamW optimizer and use the fused version if it is available
        use_fused = "cuda" in device
        try:
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), fused=use_fused)
        except:
            use_fused = False
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate)
        print(f"using fused AdamW: {use_fused}")
        
        if prints:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        return optimizer

    def get_num_params(self):
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(self, idx, targets=None, kv_caches=None):
        B,T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, max is {self.config.block_size}"

        # Determine start position for RoPE based on cache
        start_pos = kv_caches[0]['k_r'].shape[2] if kv_caches is not None else 0
        freqs_cis = self.freqs_cis[start_pos : start_pos + T]

        x = self.transformer.drop(self.tkn_emb(idx))

        updated_kv_caches = []
        for i, block in enumerate(self.transformer.h):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, updated_cache = block(x, freqs_cis, kv_cache=layer_cache)
            updated_kv_caches.append(updated_cache)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # For generation, we only need logits of the last token
            logits = self.lm_head(x[:, [-1], :])
            loss = None
            
        return logits, loss, updated_kv_caches
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Autoregressive generation of new tokens with sliding window KV cache.
        Allows for generation longer than the model's block_size.
        """
        self.eval() # Set avoids dropout
        kv_caches = None
        
        for _ in range(max_new_tokens):
            # Truncate the prompt if it's longer than the block size.
            if kv_caches is None:
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            else:
                idx_cond = idx[:, -1:]

            if kv_caches is not None:
                # Using c_kv as reference for checking seq len.
                cached_len = kv_caches[0]['c_kv'].shape[1]
                if cached_len >= self.config.block_size:
                    keep_len = self.config.block_size - 1
                    for i in range(len(kv_caches)):
                        kv_caches[i]['c_kv'] = kv_caches[i]['c_kv'][:, -keep_len:, :]
                        kv_caches[i]['k_r']  = kv_caches[i]['k_r'][:, :, -keep_len:, :] # k_r cache has sequence length at dim 2
            
            logits, _, kv_caches = self(idx_cond, kv_caches=kv_caches)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        self.train() # back into training you go
        return idx
    