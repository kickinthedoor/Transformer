import torch
import matplotlib.pyplot as plt
import seaborn as sns

from generation import translate_sentence


TITLE_MAP = {"cross": "Decoder Cross-Attention", "self": "Decoder Self-Attention", "encoder": "Encoder Self-Attention"}
AXIS_LABELS = {
    "cross": ("Source Tokens", "Target Tokens"),
    "self": ("Target Tokens (Key)", "Target Tokens (Query)"),
    "encoder": ("Source Tokens (Key)", "Source Tokens (Query)"),
}


def plot_attention(
    model,
    src_sentence,
    src_tokenizer,
    tgt_tokenizer,
    device,
    attn_type="cross",
    layer_idx=None,
    head_idx=None,
    save_path=None
):
    """
    Plots attention weights for a source sentence. attn_type: "cross" (decoder
    attending to source), "self" (decoder attending to its own previous
    tokens), or "encoder" (encoder source tokens attending to each other).

    layer_idx/head_idx both given: a single full-size heatmap with token
    labels and a colorbar, for close inspection of one (layer, head).
    Otherwise: a compact, unlabeled grid across every layer and head at once,
    for scanning the whole model in one figure.
    """
    model.eval()
    pad_id = getattr(src_tokenizer, 'pad_id', 0)

    src_tokens = src_tokenizer.encode(src_sentence)
    inp = torch.tensor([src_tokens], dtype=torch.long, device=device)
    enc_mask = (inp == pad_id).unsqueeze(1).unsqueeze(2)

    translated_text = translate_sentence(model, src_sentence, src_tokenizer, tgt_tokenizer, device)
    tgt_tokens = tgt_tokenizer.encode(translated_text)
    tar = torch.tensor([tgt_tokens], dtype=torch.long, device=device)
    dec_mask = (tar == pad_id).unsqueeze(1).unsqueeze(2).float()

    with torch.no_grad():
        _, attn_weights = model(inp, tar, enc_mask, dec_mask, return_attention=True)

    src_labels = [src_tokenizer.decode([t]) for t in src_tokens]
    tgt_labels = [tgt_tokenizer.decode([t]) for t in tgt_tokens]

    if attn_type == "encoder":
        num_layers = len(model.enc_layers)
        key_prefix = "enc_layer"
        key_suffix = "self_attn"
        row_labels, col_labels = src_labels, src_labels
    else:
        num_layers = len(model.dec_layers)
        key_prefix = "dec_layer"
        key_suffix = "cross_attn" if attn_type == "cross" else "self_attn"
        row_labels = tgt_labels
        col_labels = src_labels if attn_type == "cross" else tgt_labels

    num_heads = attn_weights[f"{key_prefix}_1_{key_suffix}"].shape[1]

    if layer_idx is not None and head_idx is not None:
        actual_layer = num_layers if layer_idx == -1 else (layer_idx + 1 if layer_idx >= 0 else num_layers + layer_idx + 1)
        matrix = attn_weights[f"{key_prefix}_{actual_layer}_{key_suffix}"].squeeze(0)[head_idx].cpu().numpy()
        matrix = matrix[:len(row_labels), :len(col_labels)]

        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(
            matrix,
            xticklabels=col_labels,
            yticklabels=row_labels,
            cmap="viridis",
            annot=False,
            cbar=True,
            ax=ax,
            square=True
        )

        xlabel, ylabel = AXIS_LABELS[attn_type]
        ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

        plt.title(f"{TITLE_MAP[attn_type]} Matrix (Layer {actual_layer}, Head {head_idx + 1})", fontsize=12, pad=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to {save_path}")

        plt.show()
        return

    fig, axes = plt.subplots(
        num_layers, num_heads,
        figsize=(num_heads * 2.5, num_layers * 2.5),
        squeeze=False
    )

    for layer_i in range(num_layers):
        layer_attn = attn_weights[f"{key_prefix}_{layer_i + 1}_{key_suffix}"].squeeze(0)
        for head_i in range(num_heads):
            ax = axes[layer_i][head_i]
            matrix = layer_attn[head_i].cpu().numpy()[:len(row_labels), :len(col_labels)]

            sns.heatmap(matrix, cmap="viridis", cbar=False, ax=ax, square=True)
            ax.set_xticks([])
            ax.set_yticks([])
            if layer_i == 0:
                ax.set_title(f"Head {head_i + 1}", fontsize=9)
            if head_i == 0:
                ax.set_ylabel(f"Layer {layer_i + 1}", fontsize=9, rotation=0, ha='right', va='center')

    plt.suptitle(
        f"{TITLE_MAP[attn_type]} — All {num_layers} Layers × {num_heads} Heads",
        fontsize=13
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Grid saved to {save_path}")

    plt.show()
