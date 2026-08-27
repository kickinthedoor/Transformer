import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    pos_enc = np.array([
        [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
        for pos in range(max_len)
    ])
    pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])
    pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])
    return torch.from_numpy(pos_enc[np.newaxis, ...]).float()

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        self.dropout = dropout

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        # Transposes from (B, S, H, Depth) to (B, H, S, Depth)
        return x.view(batch_size, -1, self.num_heads, self.depth).permute(0, 2, 1, 3)

    def forward(self, v, k, q, mask=None, return_attention=False):
        b = q.size(0)

        # 1. Linear projections and split into multi-head shapes: (B, H, S, Depth)
        q = self.split_heads(self.wq(q), b)
        k = self.split_heads(self.wk(k), b)
        v = self.split_heads(self.wv(v), b)

        # 2. Prepare Mask for SDPA (True = Attend, False = Ignore)
        attn_mask = None
        if mask is not None:
            # Invert padding mask: padding 1s (ignore) become False
            attn_mask = ~mask.bool()
            # If 3D mask [B, S, S], unsqueeze head dimension -> [B, 1, S, S]
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)

        # 3. Fast SDPA Path (Triggers FlashAttention/CuDNN)
        if not return_attention:
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False
            )
            # Reshape back to (B, Seq_Len, d_model)
            concat = out.permute(0, 2, 1, 3).contiguous().view(b, -1, self.d_model)
            return self.fc(concat), None

        # 4. Fallback Path for explicit attention matrix plotting
        scaled_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.depth)
        if mask is not None:
            # Filled with -inf where mask is True (padding positions)
            scaled_logits = scaled_logits.masked_fill(mask.bool(), float("-inf"))

        attn_weights = F.softmax(scaled_logits, dim=-1)
        out = torch.matmul(attn_weights, v)
        concat = out.permute(0, 2, 1, 3).contiguous().view(b, -1, self.d_model)
        return self.fc(concat), attn_weights


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout=rate)
        self.ffn = nn.Sequential(nn.Linear(d_model, dff), nn.ReLU(), nn.Linear(dff, d_model))
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop1, self.drop2 = nn.Dropout(rate), nn.Dropout(rate)

    def forward(self, x, mask=None, return_attention=False):
        norm_x = self.norm1(x)
        attn_out, w = self.mha(norm_x, norm_x, norm_x, mask, return_attention=return_attention)
        x = x + self.drop1(attn_out)

        norm_x = self.norm2(x)
        x = x + self.drop2(self.ffn(norm_x))

        if return_attention:
            return x, w.detach().cpu()
        return x, None


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads, dropout=rate)
        self.mha2 = MultiHeadAttention(d_model, num_heads, dropout=rate)
        self.ffn = nn.Sequential(nn.Linear(d_model, dff), nn.ReLU(), nn.Linear(dff, d_model))
        self.norm1, self.norm2, self.norm3 = [nn.LayerNorm(d_model, eps=1e-6) for _ in range(3)]
        self.drop1, self.drop2, self.drop3 = [nn.Dropout(rate) for _ in range(3)]

    def forward(self, x, enc_out, combined_mask, enc_mask, return_attention=False):
        norm_x = self.norm1(x)
        a1, w1 = self.mha1(norm_x, norm_x, norm_x, combined_mask, return_attention=return_attention)
        x = x + self.drop1(a1)

        norm_x = self.norm2(x)
        a2, w2 = self.mha2(enc_out, enc_out, norm_x, enc_mask, return_attention=return_attention)
        x = x + self.drop2(a2)

        norm_x = self.norm3(x)
        x = x + self.drop3(self.ffn(norm_x))

        if return_attention:
            return x, w1.detach().cpu(), w2.detach().cpu()
        return x, None, None


class Transformer(nn.Module):
    def __init__(self, num_enc_layers, num_dec_layers, d_model, num_heads, dff, src_vocab, tgt_vocab, max_pe_src, max_pe_tgt, rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(src_vocab, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model)

        self.register_buffer("pe_src", get_positional_encoding(max_pe_src, d_model))
        self.register_buffer("pe_tgt", get_positional_encoding(max_pe_tgt, d_model))

        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_enc_layers)])
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_dec_layers)])

        self.encoder_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.decoder_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.final_layer = nn.Linear(d_model, tgt_vocab)
        self.drop = nn.Dropout(rate)

        max_len = max(max_pe_src, max_pe_tgt)
        causal_mask = torch.triu(
            torch.ones((max_len, max_len), dtype=torch.bool),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, inp, tar, enc_mask=None, dec_mask=None, return_attention=False):
        batch_size, tar_seq_len = tar.shape
        c_mask = self.causal_mask[:, :, :tar_seq_len, :tar_seq_len]

        if dec_mask is not None:
            if dec_mask.size(-1) > tar_seq_len:
                dec_mask = dec_mask[..., :tar_seq_len]
            combined_mask = c_mask | dec_mask.bool()
        else:
            combined_mask = c_mask

        # Pre-scale factor as a float constant
        scale_factor = math.sqrt(self.d_model)

        attention_weights = {} if return_attention else None

        # Encoder Pass
        e_seq_len = inp.size(1)
        x_enc = self.drop(self.src_emb(inp) * scale_factor + self.pe_src[:, :e_seq_len, :])
        for i, layer in enumerate(self.enc_layers):
            x_enc, w_enc = layer(x_enc, enc_mask, return_attention=return_attention)
            if return_attention:
                attention_weights[f'enc_layer_{i+1}_self_attn'] = w_enc
        x_enc = self.encoder_norm(x_enc)

        # Decoder Pass
        d_seq_len = tar_seq_len
        x_dec = self.drop(self.tgt_emb(tar) * scale_factor + self.pe_tgt[:, :d_seq_len, :])

        for i, layer in enumerate(self.dec_layers):
            x_dec, w1, w2 = layer(x_dec, x_enc, combined_mask, enc_mask, return_attention=return_attention)
            if return_attention:
                attention_weights[f'dec_layer_{i+1}_self_attn'] = w1
                attention_weights[f'dec_layer_{i+1}_cross_attn'] = w2

        x_dec = self.decoder_norm(x_dec)
        logits = self.final_layer(x_dec)

        if return_attention:
            return logits, attention_weights
        return logits