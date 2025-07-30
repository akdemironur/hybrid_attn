from typing import Optional

import torch
import torch.nn.functional as F

from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb

from flux_hybrid_attention.perturbated_attention import perturbated_attention


class HybridNAGPAGFluxAttnProcessor2_0:
    """Hybrid NAG/PAG attention processor for Flux Transformer."""

    def __init__(
        self,
        nag_scale: float = 1.0,
        nag_tau=2.5,
        nag_alpha=0.25,
        hybrid_alpha=0.25,
        encoder_hidden_states_length: int = None,
    ):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "HybridNAGPAGFluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
        self.nag_scale = nag_scale
        self.nag_tau = nag_tau
        self.nag_alpha = nag_alpha
        self.hybrid_alpha = hybrid_alpha
        self.encoder_hidden_states_length = encoder_hidden_states_length

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if self.nag_scale > 1.0:
            if encoder_hidden_states is not None:
                assert len(hidden_states) == batch_size * 0.5
            apply_guidance = True
        else:
            apply_guidance = False

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # attention
        if apply_guidance and encoder_hidden_states is not None:
            query = query.tile(2, 1, 1)
            key = key.tile(2, 1, 1)
            value = value.tile(2, 1, 1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(
                    encoder_hidden_states_query_proj
                )
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(
                    encoder_hidden_states_key_proj
                )

            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

            encoder_hidden_states_length = encoder_hidden_states.shape[1]

        else:
            assert self.encoder_hidden_states_length is not None
            encoder_hidden_states_length = self.encoder_hidden_states_length

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        if not apply_guidance:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=False
            )
            hidden_states = hidden_states.transpose(1, 2).reshape(
                batch_size, -1, attn.heads * head_dim
            )
            hidden_states = hidden_states.to(query.dtype)

        else:
            origin_batch_size = batch_size // 2
            query, query_negative = torch.chunk(query, 2, dim=0)
            key, key_negative = torch.chunk(key, 2, dim=0)
            value, value_negative = torch.chunk(value, 2, dim=0)

            hidden_states_negative = perturbated_attention(
                query_negative,
                key_negative,
                value_negative,
                self.nag_scale - 1,
                self.hybrid_alpha,
            )
            hidden_states_negative = hidden_states_negative.transpose(1, 2).reshape(
                origin_batch_size, -1, attn.heads * head_dim
            )
            hidden_states_negative = hidden_states_negative.to(query.dtype)

            hidden_states = F.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=False
            )
            hidden_states = hidden_states.transpose(1, 2).reshape(
                origin_batch_size, -1, attn.heads * head_dim
            )
            hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            if apply_guidance:
                encoder_hidden_states_negative, hidden_states_negative = (
                    hidden_states_negative[:, : encoder_hidden_states.shape[1]],
                    hidden_states_negative[:, encoder_hidden_states.shape[1] :],
                )
                hidden_states_positive = hidden_states
                hidden_states_guidance = (
                    hidden_states_positive * self.nag_scale
                    - hidden_states_negative * (self.nag_scale - 1)
                )
                norm_positive = torch.norm(
                    hidden_states_positive, p=2, dim=-1, keepdim=True
                ).expand(*hidden_states_positive.shape)
                norm_guidance = torch.norm(
                    hidden_states_guidance, p=2, dim=-1, keepdim=True
                ).expand(*hidden_states_positive.shape)

                scale = norm_guidance / norm_positive
                hidden_states_guidance = (
                    hidden_states_guidance
                    * torch.minimum(scale, scale.new_ones(1) * self.nag_tau)
                    / scale
                )

                hidden_states = (
                    hidden_states_guidance * self.nag_alpha
                    + hidden_states_positive * (1 - self.nag_alpha)
                )

                encoder_hidden_states = torch.cat(
                    (encoder_hidden_states, encoder_hidden_states_negative), dim=0
                )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states

        else:
            if apply_guidance:
                image_hidden_states_negative = hidden_states_negative[
                    :, encoder_hidden_states_length:
                ]
                image_hidden_states = hidden_states[:, encoder_hidden_states_length:]

                image_hidden_states_positive = image_hidden_states
                image_hidden_states_guidance = (
                    image_hidden_states_positive * self.nag_scale
                    - image_hidden_states_negative * (self.nag_scale - 1)
                )
                norm_positive = torch.norm(
                    image_hidden_states_positive, p=2, dim=-1, keepdim=True
                ).expand(*image_hidden_states_positive.shape)
                norm_guidance = torch.norm(
                    image_hidden_states_guidance, p=2, dim=-1, keepdim=True
                ).expand(*image_hidden_states_positive.shape)

                scale = norm_guidance / norm_positive
                image_hidden_states_guidance = (
                    image_hidden_states_guidance
                    * torch.minimum(scale, scale.new_ones(1) * self.nag_tau)
                    / scale
                )
                # scale = torch.nan_to_num(scale, 10)
                # image_hidden_states_guidance[scale > self.nag_tau] = image_hidden_states_guidance[scale > self.nag_tau] / (norm_guidance[scale > self.nag_tau] + 1e-7) * norm_positive[scale > self.nag_tau] * self.nag_tau

                image_hidden_states = (
                    image_hidden_states_guidance * self.nag_alpha
                    + image_hidden_states_positive * (1 - self.nag_alpha)
                )

                hidden_states_negative[:, encoder_hidden_states_length:] = (
                    image_hidden_states
                )
                hidden_states[:, encoder_hidden_states_length:] = image_hidden_states
                hidden_states = torch.cat(
                    (hidden_states, hidden_states_negative), dim=0
                )

            return hidden_states
