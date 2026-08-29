import torch
import sacrebleu
import matplotlib.pyplot as plt
from nltk.translate.nist_score import corpus_nist

from generation import translate_sentence_beam_search


def evaluate_translation_metrics(
    model, dataloader, src_tokenizer, tgt_tokenizer, device,
    max_len=50, use_beam_search=True, beam_size=4, alpha=0.6
):
    """
    Evaluates translation quality using SacreBLEU, TER, and chrF scores.

    use_beam_search=True (default) decodes with beam search, sentence by sentence —
    greedy decoding is prone to degenerate repetition loops ("it. it. it up. too.
    too.") that beam search largely avoids, giving a more representative BLEU/chrF/
    TER score. This is significantly slower than greedy since it can't batch across
    sentences the same way. If this is used as a per-epoch hook during a long
    training run (see trainer.fit()'s translation_eval_fn), consider passing
    use_beam_search=False there for fast, cheap progress checks, and reserve
    use_beam_search=True for a final, one-off quality evaluation.
    """
    model.eval()
    hypotheses = []
    references = []

    if use_beam_search:
        with torch.no_grad():
            for (inp, _), (tar, _) in dataloader:
                batch_sz = inp.size(0)
                for b in range(batch_sz):
                    src_text = src_tokenizer.decode(inp[b].tolist())
                    hyp = translate_sentence_beam_search(
                        model, src_text, src_tokenizer, tgt_tokenizer, device,
                        beam_size=beam_size, alpha=alpha, max_len=max_len
                    )
                    hypotheses.append(hyp)
                    references.append(tgt_tokenizer.decode(tar[b].tolist()))
    else:
        with torch.no_grad():
            for (inp, mask_inp), (tar, _) in dataloader:
                inp, mask_inp = inp.to(device), mask_inp.to(device)
                batch_sz = inp.size(0)

                # Autoregressive sequence generation
                ys = torch.full((batch_sz, 1), tgt_tokenizer.sos_id, dtype=torch.long, device=device)
                for _ in range(max_len):
                    dec_pad = (ys == 0).unsqueeze(1).unsqueeze(2).float()

                    out = model(inp, ys, mask_inp, dec_pad)
                    next_word = torch.argmax(out[:, -1, :], dim=-1, keepdim=True)
                    ys = torch.cat([ys, next_word], dim=1)

                    # Stop early if all sequences output EOS token
                    if (ys == tgt_tokenizer.eos_id).any(dim=1).all():
                        break

                # Decode token IDs to text strings
                for b in range(batch_sz):
                    gen_ids = ys[b].tolist()
                    ref_ids = tar[b].tolist()

                    # Strip dynamic padding/SOS/EOS markers before computing metrics
                    hypotheses.append(tgt_tokenizer.decode(gen_ids))
                    references.append(tgt_tokenizer.decode(ref_ids))

    # Calculate standard translation benchmark metrics via SacreBLEU
    bleu_score = sacrebleu.corpus_bleu(hypotheses, [references]).score
    chrf_score = sacrebleu.corpus_chrf(hypotheses, [references]).score
    ter_score = sacrebleu.corpus_ter(hypotheses, [references]).score

    # NIST weights n-gram matches by informativeness (rare/content words count more
    # than common ones) — sacrebleu doesn't implement it, so this uses nltk instead.
    # Needs whitespace-tokenized input, unlike sacrebleu's raw-string metrics above.
    tokenized_hyps = [h.split() for h in hypotheses]
    tokenized_refs = [[r.split()] for r in references]
    try:
        nist_score = corpus_nist(tokenized_refs, tokenized_hyps)
    except ZeroDivisionError:
        # nltk's NIST divides by zero when hypotheses are too short/degenerate
        # to have any n-grams at some order — realistically only a risk very
        # early in training (e.g. curriculum stage 1, epoch 1) before the
        # model produces anything substantive. Fall back rather than crash a
        # multi-hour run over one progress-check metric.
        print("[evaluate_translation_metrics] NIST hit a ZeroDivisionError (likely still near-degenerate output) — reporting 0.0 for this check.")
        nist_score = 0.0

    return {
        "bleu": bleu_score,
        "chrf": chrf_score,
        "ter": ter_score,
        "nist": nist_score,
        "sample_hyp": hypotheses[:3],
        "sample_ref": references[:3]
    }


def evaluate_on_official_iwslt17(
    model, src_tokenizer, tgt_tokenizer, device,
    max_len=50, beam_size=4, alpha=0.6
):
    """
    Evaluates the model on the actual official IWSLT17 de-en test set (tst2017,
    1,138 sentence pairs), fetched via sacrebleu's built-in dataset registry.

    This is distinct from — and much smaller than — the HuggingFace
    `IWSLT/iwslt2017` loader's own "test" split used elsewhere in this project,
    which is actually tst2010-tst2015 concatenated (8,079 pairs), not tst2017.
    Scores here are directly comparable to published IWSLT17 de-en benchmarks;
    scores from the other test split are not.
    """
    model.eval()

    src_path = sacrebleu.DATASETS['iwslt17'].get_source_file('de-en')
    ref_path = sacrebleu.DATASETS['iwslt17'].get_reference_files('de-en')[0]

    with open(src_path, encoding='utf-8') as f:
        src_sentences = [line.strip() for line in f]
    with open(ref_path, encoding='utf-8') as f:
        references = [line.strip() for line in f]

    hypotheses = []
    with torch.no_grad():
        for src in src_sentences:
            hyp = translate_sentence_beam_search(
                model, src, src_tokenizer, tgt_tokenizer, device,
                beam_size=beam_size, alpha=alpha, max_len=max_len
            )
            hypotheses.append(hyp)

    bleu_score = sacrebleu.corpus_bleu(hypotheses, [references]).score
    chrf_score = sacrebleu.corpus_chrf(hypotheses, [references]).score
    ter_score = sacrebleu.corpus_ter(hypotheses, [references]).score

    # See evaluate_translation_metrics above for why NIST goes through nltk instead.
    tokenized_hyps = [h.split() for h in hypotheses]
    tokenized_refs = [[r.split()] for r in references]
    nist_score = corpus_nist(tokenized_refs, tokenized_hyps)

    return {
        "bleu": bleu_score,
        "chrf": chrf_score,
        "ter": ter_score,
        "nist": nist_score,
        "sample_hyp": hypotheses[:3],
        "sample_ref": references[:3]
    }


def plot_bleu_vs_length(model, dataloader, tgt_tokenizer, device, max_len=50, num_bins=5):
    """
    Evaluates generated translations and plots BLEU scores binned by source sequence length.
    """
    model.eval()
    results = []

    with torch.no_grad():
        for (inp, mask_inp), (tar, _) in dataloader:
            inp, mask_inp = inp.to(device), mask_inp.to(device)
            batch_sz = inp.size(0)

            ys = torch.full((batch_sz, 1), tgt_tokenizer.sos_id, dtype=torch.long, device=device)
            for _ in range(max_len):
                dec_pad = (ys == 0).unsqueeze(1).unsqueeze(2).float()

                out = model(inp, ys, mask_inp, dec_pad)
                next_word = torch.argmax(out[:, -1, :], dim=-1, keepdim=True)
                ys = torch.cat([ys, next_word], dim=1)

                if (ys == tgt_tokenizer.eos_id).any(dim=1).all():
                    break

            for b in range(batch_sz):
                # Measure true source length without padding
                src_len = (inp[b] != 0).sum().item()

                hyp_str = tgt_tokenizer.decode(ys[b].tolist())
                ref_str = tgt_tokenizer.decode(tar[b].tolist())

                results.append({"src_len": src_len, "hyp": hyp_str, "ref": ref_str})

    # Sort results by sequence length and group into equal-sized bins
    results.sort(key=lambda x: x["src_len"])
    bin_size = len(results) // num_bins

    bin_centers = []
    bin_bleus = []

    for i in range(num_bins):
        bin_data = results[i * bin_size: (i + 1) * bin_size] if i < num_bins - 1 else results[i * bin_size:]

        avg_len = sum(d["src_len"] for d in bin_data) / len(bin_data)
        hyps = [d["hyp"] for d in bin_data]
        refs = [[d["ref"] for d in bin_data]]

        score = sacrebleu.corpus_bleu(hyps, refs).score

        bin_centers.append(avg_len)
        bin_bleus.append(score)

    # Plot BLEU vs Length
    plt.figure(figsize=(8, 5))
    plt.plot(bin_centers, bin_bleus, marker='o', linewidth=2, color='b')
    plt.xlabel("Average Source Sequence Length (Tokens)")
    plt.ylabel("SacreBLEU Score")
    plt.title("BLEU Score Degradation Across Sequence Lengths")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
