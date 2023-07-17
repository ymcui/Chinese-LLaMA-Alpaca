import torch
from torch import nn
from typing import Optional, Tuple, Union
import transformers
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, rotate_half
import math

try:
    from xformers import ops as xops
except ImportError:
    xops = None
    print(
        "Xformers is not installed correctly. If you want to use memory_efficient_attention use the following command to install Xformers\npip install xformers."
    )


STORE_KV_BEFORE_ROPE = False
USE_MEM_EFF_ATTENTION = False
ALPHA = 1.0


def apply_rotary_pos_emb_single(q, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed


def xformers_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    if STORE_KV_BEFORE_ROPE is False:
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None
    else:
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        past_key_value = (key_states, value_states) if use_cache else None

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)
        position_ids = torch.arange(kv_seq_len, dtype=torch.long, device=cos.device)
        position_ids = position_ids.unsqueeze(0).view(-1, kv_seq_len)
        key_states = apply_rotary_pos_emb_single(key_states, cos, sin, position_ids)

    if xops is not None and USE_MEM_EFF_ATTENTION:
        attn_weights = None
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        attn_bias = None if (query_states.size(1)==1 and key_states.size(1)>1) else xops.LowerTriangularMask()
        attn_output = xops.memory_efficient_attention(
            query_states, key_states, value_states, attn_bias=attn_bias, p=0)
    else:
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            )

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


old_init = transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.__init__

def adaptive_ntk_init(self, dim, max_position_embeddings=2048, base=10000, device=None):
    self.dim = dim
    self.alpha = ALPHA
    if isinstance(ALPHA,(float,int)):
        base = base * ALPHA ** (dim / (dim-2))
        self.base = base
    elif ALPHA=='auto':
        self.base = base
    else:
        raise ValueError(ALPHA)
    old_init(self, dim, max_position_embeddings, base, device)
    ntk_inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
    self.register_buffer("ntk_inv_freq", ntk_inv_freq, persistent=False)

def adaptive_ntk_forward(self, x, seq_len=None):
    if seq_len > self.max_seq_len_cached:
        if isinstance(self.alpha,(float,int)):
            self.max_seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device, dtype=self.ntk_inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.ntk_inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
            return (
                self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
                self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            )
        elif self.alpha=='auto':
            t = torch.arange(seq_len, device=x.device, dtype=self.ntk_inv_freq.dtype)
            dim = self.dim
            alpha = (seq_len / 1024 - 1) * 1.1
            base = self.base * alpha ** (dim / (dim-2))
            ntk_inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(x.device) / dim ))

            freqs = torch.einsum("i,j->ij", t, ntk_inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            cos_cached = emb.cos()[None, None, :, :]
            sin_cached = emb.sin()[None, None, :, :]
            return (
                cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
                sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype)
            )
    else:
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype)
        )


def apply_attention_patch(
        use_memory_efficient_attention=False,
        store_kv_before_rope=False
        ):
    global USE_MEM_EFF_ATTENTION, STORE_KV_BEFORE_ROPE
    if use_memory_efficient_attention is True and xops is not None:
        USE_MEM_EFF_ATTENTION = use_memory_efficient_attention
    print("USE_MEM_EFF_ATTENTION: ",USE_MEM_EFF_ATTENTION)
    STORE_KV_BEFORE_ROPE = store_kv_before_rope
    print("STORE_KV_BEFORE_ROPE:", STORE_KV_BEFORE_ROPE)
    transformers.models.llama.modeling_llama.LlamaAttention.forward = xformers_forward


def apply_ntk_scaling_patch(alpha: Union[float,str]):
    global ALPHA
    ALPHA = alpha
    try:
        ALPHA = float(ALPHA)
    except ValueError:
        if ALPHA!="auto":
            raise ValueError(f"Alpha can only be a float or 'auto', but given {ALPHA}")
    print(f"Apply NTK scaling with ALPHA={ALPHA}")

    transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.__init__ = adaptive_ntk_init
    transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.forward = adaptive_ntk_forward