import torch
from torch import nn
from typing import Optional, Tuple, Union
import transformers
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
import math

def apply_memory_efficient_attnetion():

    try:
        from xformers import ops as xops
    except ImportError:
        xops = None
        print(
            "Xformers is not installed correctly. If you want to use memory_efficient_attention to accelerate training use the following command to install Xformers\npip install xformers."
        )

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
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        if xops is not None:
            attn_weights = None
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)
            attn_output = xops.memory_efficient_attention(
                query_states, key_states, value_states, attn_bias=xops.LowerTriangularMask(), p=0)
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

    transformers.models.llama.modeling_llama.LlamaAttention.forward = xformers_forward


def apply_ntk_scaling(alpha: Union[float,str]):
    old_init = transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.__init__

    def adaptive_ntk_init(self, dim, max_position_embeddings=2048, base=10000, device=None):
        self.dim = dim
        self.alpha = alpha
        if isinstance(alpha,(float,int)):
            base = base * alpha ** (dim / (dim-2))
            max_position_embeddings = 32768
            self.base = base
        elif alpha=='auto':
            self.base = base
        else:
            raise ValueError(alpha)
        old_init(self, dim, max_position_embeddings, base, device)

    def adaptive_ntk_forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            if isinstance(self.alpha,(float,int)):
                self.max_seq_len_cached = seq_len
                t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
                freqs = torch.einsum("i,j->ij", t, self.inv_freq)
                # Different from paper, but it uses a different permutation in order to obtain the same calculation
                emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
                self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
                self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
                return (
                    self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
                    self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
                )
            elif self.alpha=='auto':
                t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
                inv_freq = self.inv_freq
                dim = self.dim
                alpha = seq_len / 1024 - 1
                base = self.base * alpha ** (dim / (dim-2))
                inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(x.device) / dim ))

                freqs = torch.einsum("i,j->ij", t, inv_freq)
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

    transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.__init__ = adaptive_ntk_init
    transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.forward = adaptive_ntk_forward