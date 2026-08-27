import torch


def translate_sentence(
    model,
    src_sentence: str,
    de_tokenizer,
    en_tokenizer,
    device,
    max_len: int = 50
) -> str:
    """
    Translates a single source sentence using Greedy Search decoding.
    """
    model.eval()
    pad_id = getattr(de_tokenizer, 'pad_id', 0)

    # 1. Tokenize & format input tensor
    src_tokens = de_tokenizer.encode(src_sentence)
    inp = torch.tensor([src_tokens], dtype=torch.long, device=device)
    mask_inp = (inp == pad_id).unsqueeze(1).unsqueeze(2)

    tgt_tokens = [en_tokenizer.sos_id]

    with torch.no_grad():
        for _ in range(max_len):
            tar = torch.tensor([tgt_tokens], dtype=torch.long, device=device)
            dec_pad = (tar == pad_id).unsqueeze(1).unsqueeze(2).float()

            # Forward pass
            out = model(inp, tar, mask_inp, dec_pad)

            # Greedy token selection
            next_word = torch.argmax(out[:, -1, :], dim=-1).item()

            if next_word == en_tokenizer.eos_id:
                break

            tgt_tokens.append(next_word)

    clean_ids = [t for t in tgt_tokens if t not in (en_tokenizer.sos_id, en_tokenizer.eos_id, pad_id)]
    return en_tokenizer.decode(clean_ids)


def translate_sentence_beam_search(
    model,
    src_sentence: str,
    de_tokenizer,
    en_tokenizer,
    device,
    beam_size: int = 4,
    alpha: float = 0.6,
    max_len: int = 50
) -> str:
    """
    Translates a single source sentence using Batched Beam Search decoding
    with Google NMT length penalty (alpha).
    """
    model.eval()
    pad_id = getattr(de_tokenizer, 'pad_id', 0)

    # 1. Prepare source input
    src_tokens = de_tokenizer.encode(src_sentence)
    inp = torch.tensor([src_tokens], dtype=torch.long, device=device)
    mask_inp = (inp == pad_id).unsqueeze(1).unsqueeze(2)

    # Track beam hypotheses: list of (sequence_list, log_prob_score)
    beams = [([en_tokenizer.sos_id], 0.0)]
    completed_beams = []

    with torch.no_grad():
        for _ in range(max_len):
            # If all active beams are completed or empty, stop early
            if not beams:
                break

            active_beams = []

            # Separate completed beams from active ones
            for seq, score in beams:
                if seq[-1] == en_tokenizer.eos_id:
                    completed_beams.append((seq, score))
                else:
                    active_beams.append((seq, score))

            if not active_beams:
                break

            # Batch all active beams together for a single forward pass
            curr_beam_size = len(active_beams)
            batch_inp = inp.repeat(curr_beam_size, 1)
            batch_mask_inp = mask_inp.repeat(curr_beam_size, 1, 1, 1)

            seqs = [b[0] for b in active_beams]
            scores = torch.tensor([b[1] for b in active_beams], device=device)

            tar = torch.tensor(seqs, dtype=torch.long, device=device)
            dec_pad = (tar == pad_id).unsqueeze(1).unsqueeze(2).float()

            # Single forward pass for all beams
            out = model(batch_inp, tar, batch_mask_inp, dec_pad)
            log_probs = torch.log_softmax(out[:, -1, :], dim=-1)  # [curr_beam_size, vocab_size]

            # Compute cumulative scores for top-k candidates per beam
            topk_log_probs, topk_indices = torch.topk(log_probs, beam_size, dim=-1)

            new_candidates = []
            for i in range(curr_beam_size):
                for k in range(beam_size):
                    next_token = topk_indices[i, k].item()
                    next_score = scores[i].item() + topk_log_probs[i, k].item()
                    new_candidates.append((seqs[i] + [next_token], next_score))

            # Keep top beam_size overall active candidates
            beams = sorted(new_candidates, key=lambda x: x[1], reverse=True)[:beam_size]

    # Include remaining active beams into completed pool
    completed_beams.extend(beams)

    # 2. Length Penalty Normalization Function
    def compute_normalized_score(hypothesis):
        seq, raw_score = hypothesis
        seq_len = len(seq)
        # Length penalty formula: ((5 + len)^alpha) / ((5 + 1)^alpha)
        lp = ((5.0 + seq_len) ** alpha) / ((5.0 + 1.0) ** alpha)
        return raw_score / lp

    # Select candidate with highest normalized score
    best_seq = max(completed_beams, key=compute_normalized_score)[0]

    clean_ids = [t for t in best_seq if t not in (en_tokenizer.sos_id, en_tokenizer.eos_id, pad_id)]
    return en_tokenizer.decode(clean_ids)
