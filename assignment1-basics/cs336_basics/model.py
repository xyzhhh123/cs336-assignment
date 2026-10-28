import torch
import torch.nn as nn
import math

class Linear(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        limit = (6.0 / (d_in + d_out)) ** 0.5
        self.weight = nn.Parameter(torch.empty(d_out, d_in).uniform_(-limit, limit))

    def forward(self, x):
        return x @ self.weight.T

class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, d_model) * 0.02)

    def forward(self, x):
        return self.weight[x]

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
            dtype = x.dtype
            xf = x.float()
            out = self.weight * xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            return out.to(dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x):
        return self.w2(silu(self.w1(x)) * self.w3(x))

class Rope(nn.Module):
    def __init__(self, d_model, theta, max_seq_len):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for RoPE"
        self.d_model = d_model
        freqs = theta ** (-torch.arange(0, self.d_model, 2, dtype=torch.float32) / self.d_model)
        m = torch.arange(max_seq_len, dtype=torch.float32)
        angle = m[:, None] * freqs[None, :]
        cos_cache = torch.cos(angle)
        sin_cache = torch.sin(angle)
        self.register_buffer("cos_cache", cos_cache, persistent=False)
        self.register_buffer("sin_cache", sin_cache, persistent=False)
    
    def forward(self, x, token_positions):
        cos_selected = self.cos_cache[token_positions]
        sin_selected = self.sin_cache[token_positions]
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        new_even = x_even * cos_selected - x_odd * sin_selected
        new_odd = x_even * sin_selected + x_odd * cos_selected
        return torch.stack([new_even, new_odd], dim=-1).reshape(*x.shape[:-1], -1)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len=None, theta=None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

        if max_seq_len is not None and theta is not None:
            self.rope = Rope(self.head_dim, theta, max_seq_len)
        else:
            self.rope = None

    def forward(self, x, token_positions=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q_heads = q.reshape(*q.shape[:-1], self.num_heads, self.head_dim).permute(0, -2, -3, -1)
        k_heads = k.reshape(*k.shape[:-1], self.num_heads, self.head_dim).permute(0, -2, -3, -1)
        v_heads = v.reshape(*v.shape[:-1], self.num_heads, self.head_dim).permute(0, -2, -3, -1)

        if token_positions is None:
            seq_len = q_heads.shape[2]
            token_positions = torch.arange(seq_len, device=q_heads.device)

        if self.rope is not None:
            q_heads = self.rope(q_heads, token_positions)
            k_heads = self.rope(k_heads, token_positions)

        L = q_heads.shape[2]
        mask = ~torch.triu(torch.ones(L, L, dtype=torch.bool, device=q_heads.device), diagonal=1)

        attn_out = scaled_dot_product_attention(q_heads, k_heads, v_heads, mask).permute(0,2,1,3).reshape(*x.shape[:-1],-1)
        out = self.output_proj(attn_out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta)
        self.ffn = SwiGLU(d_model, d_ff)
        self.ln2 = RMSNorm(d_model)

    def forward(self, x, token_positions=None):
        x = x + self.attn(self.ln1(x), token_positions)
        return x + self.ffn(self.ln2(x))

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        blocks = []
        for _ in range(num_layers):
            block = TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta)
            blocks.append(block)
        self.layers = nn.ModuleList(blocks)
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x, token_positions=None):
        x = self.token_embeddings(x)
        for block in self.layers:
            x = block(x, token_positions)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits

def silu(x):
    return x * torch.sigmoid(x)

def softmax(x, dim=-1):
    x_max = x.max(dim=dim, keepdim=True)[0]
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    score = Q @ K.transpose(-1, -2)
    scale = 1.0 / math.sqrt(Q.shape[-1])
    score = score * scale
    if mask is not None:
        score = score.masked_fill(~mask, -1e9)
    return softmax(score, dim=-1) @ V