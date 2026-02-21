"""
Baseline Transformer Model

This is a deliberately inefficient implementation for educational purposes.
Students will optimize this in optimized_model.py.

Inefficiencies:
1. Unfused RMSNorm (separate rms computation, division, scaling)
2. Unfused RoPE (separate sin/cos computation and rotation)
3. Unfused SwiGLU (separate silu and multiply kernels)
4. Vanilla attention is used instead of Flash Attention
5. Vanilla F.cross_entropy is used
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class TransformerConfig:
    vocab_size: int = 32000
    hidden_dim: int = 1024
    num_heads: int = 16
    num_layers: int = 12
    intermediate_dim: int = 4096
    max_seq_len: int = 2048
    dropout: float = 0.0
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    
    def __post_init__(self):
        assert self.hidden_dim % self.num_heads == 0, \
            f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma

    Reference: https://arxiv.org/abs/1910.07467
    """
    
    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        x_squared = x * x
        mean_squared = x_squared.mean(dim=-1, keepdim=True)
        rms = torch.sqrt(mean_squared + self.eps)
        x_normed = x / rms
        output = x_normed * self.weight
        return output


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE).

    Reference: https://arxiv.org/abs/2104.09864
    """
    
    def __init__(self, head_dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        """Build sin/cos cache up to seq_len."""
        positions = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        self.register_buffer('cos', emb.cos().unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer('sin', emb.sin().unsqueeze(0).unsqueeze(0), persistent=False)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary positional embedding to q and k.
        
        Args:
            q: (B, num_heads, S, head_dim)
            k: (B, num_heads, S, head_dim)
            seq_len: sequence length (must be <= max_seq_len)
            
        Returns:
            q_rotated, k_rotated with same shapes
        """
        assert seq_len <= self.max_seq_len, \
            f"seq_len ({seq_len}) exceeds max_seq_len ({self.max_seq_len})"
        
        cos = self.cos[:, :, :seq_len, :]
        sin = self.sin[:, :, :seq_len, :]

        q_rotated = self._apply_rotary(q, cos, sin)
        k_rotated = self._apply_rotary(k, cos, sin)
        
        return q_rotated, k_rotated
    
    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary embedding to tensor x.
        """
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + rotated * sin


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with vanilla implementation and RoPE.
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads

        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        self.rope = RotaryPositionalEmbedding(
            head_dim=self.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )

        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, S, H = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = self.rope(q, k, S)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attention_mask is None:
            causal_mask = torch.triu(
                torch.ones(S, S, dtype=torch.bool, device=x.device), 
                diagonal=1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        else:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)

        out = out.transpose(1, 2).contiguous().view(B, S, H)
        out = self.out_proj(out)

        return out


class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU Feed-Forward Network.

    SwiGLU(x) = W_down @ (silu(W_gate @ x) * (W_up @ x))

    Reference: https://arxiv.org/abs/2002.05202
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.intermediate_dim = config.intermediate_dim

        self.gate_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.up_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.down_proj = nn.Linear(config.intermediate_dim, config.hidden_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        gate = F.silu(gate)
        up = self.up_proj(x)
        intermediate = gate * up
        output = self.down_proj(intermediate)
        return output


class TransformerBlock(nn.Module):
    """Single transformer block."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.attn = MultiHeadAttention(config)
        self.ln2 = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.ffn = SwiGLUFeedForward(config)
    
    def forward(
        self, 
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attention_mask)
        x = x + self.ffn(self.ln2(x))
        return x


class BaselineTransformer(nn.Module):
    """
    Transformer language model.
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        self.ln_f = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)

        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (B, S) token indices
            attention_mask: optional attention mask
            
        Returns:
            logits: (B, S, vocab_size)
        """
        B, S = input_ids.shape
        x = self.embedding(input_ids)

        for layer in self.layers:
            x = layer(x, attention_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
    
    def compute_loss(
        self, 
        input_ids: torch.Tensor, 
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for language modeling.
        """
        logits = self.forward(input_ids)

        logits = logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()

        loss = F.cross_entropy(
            logits.view(-1, self.config.vocab_size),
            labels.view(-1),
            ignore_index=-100,
        )

        return loss


def get_model_config() -> TransformerConfig:
    return TransformerConfig(
        vocab_size=32000,
        hidden_dim=2048,
        num_heads=32,
        num_layers=24,
        intermediate_dim=5120,
        max_seq_len=4096,
    )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    config = get_model_config()
    model = BaselineTransformer(config)

    print(f"Config: {config}")
    print(f"Parameters: {count_parameters(model):,}")

    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()

    logits = model(input_ids)
    loss = model.compute_loss(input_ids, labels)
    loss.backward()
