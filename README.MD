## What is this project

During my masters I was very fond of the transformer model I learned of during my Deep Learning course. My thesis ended up being about using an LLM (GPT-4 based ChatGPT) to learn about Machine Learning concepts, more specifically, Transformers. 
Now, I wanted to improve my model, a machine translation transformer, as it was severly lacking in multiple aspects. One of the most important things to focus on was using benchmark datasets with comparable results, which I should have already used previously. 
Additionally, this project serves as a platform for me to learn agentic coding.
This project consists of two stages, which are similar models for the IWSLT17 de-en dataset and IWSLT14 de-en dataset. Both contain german sentences from TED talks translated to english. The reason for the '17 dataset being first is that I was comparing a bilingual model trained on small data to multilingual models on small data. 
Even though there was some evidence of multilingual models outperforming bilingual models on the few small datasets, which employed both methods, I generally do not wish to generalize based on such a small sample, with it also being on a different language combination dataset. 
That being said, I have completed the '17 model, which reached the following results:


1: Hugging Face provided test set (consisting of test sets 2010-2015)

- SacreBLEU Score: 24.16
- chrF Score:      47.07
- TER Score:       60.08
- NIST Score:      6.4209

Generated Sample Comparison 
-[1] Predicted: A couple of years ago, at TED, Peter Skillman asked a design competition called "The Marshallow House challenge."
    Reference: Several years ago here at TED, Peter Skillman introduced a design challenge called the marshmallow challenge.

-[2] Predicted: The idea is pretty simple. Verteams have to understand the biggest impossible structure with 20 spagetti, about a black band, building about a marsh
    Reference: And the idea's pretty simple: Teams of four have to build the tallest free-standing structure out of 20 sticks of spaghetti, one yard of tape, one yard of string and a marshmallow.

-[3] Predicted: Marshallow has to be up there.
    Reference: The marshmallow has to be on top.

2: Official test set IWSLT17 de-en

- SacreBLEU Score: 18.30
- chrF Score:      41.69
- TER Score:       66.19
- NIST Score:      4.6501

Generated Sample Comparison
-[1] Predicted: This is James Risse.
    Reference: So this is James Risen.

-[2] Predicted: You may know him because he won the New York Times Reporters the Pulzer Price.
    Reference: You may know him as the Pulitzer Prize-winning reporter for The New York Times.

-[3] Predicted: Long before someone who had heard of Edward Snowden, Risseen a book that he published in the spectacular, that the NSA is illegal telephone from Americans.
    Reference: Long before anybody knew Edward Snowden's name, Risen wrote a book in which he famously exposed that the NSA was illegally wiretapping the phone calls of Americans.



Criticisms: 

No directly comparable benchmarks. Longer sequences can probably be improved by using relative positional encoding as well as seq-length curriculum. Will be applied for the '14 set.
Weight decay flattened encoder attention weights to 0, which was corrected by the finetuning, will be taken into account on the '14 set.
The lower evaluation for the '17 set seems to point at a more difficult test set. To not waste time comparing apples to oranges, I move on to phase 2, which is the directly comparable IWSLT14 de-en, with the planned improvements.

## How to run it

This project assumes an NVIDIA GPU with CUDA — `config.yaml` defaults to `cuda` and the codebase is written around it throughout (no CPU fallback).

1. Clone the repo and create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate       # Windows
   source venv/bin/activate    # Linux/macOS
   ```

2. Install PyTorch from PyTorch's own CUDA build index first — it isn't hosted on plain PyPI:
   ```
   pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu118
   ```
   See [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) to pick the right CUDA build if your GPU/driver differs.

3. Install the rest of the dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Open `experiment.ipynb` (VS Code or Jupyter) and run the cells top to bottom.
   - The first run downloads the IWSLT17 de-en dataset via Hugging Face `datasets`, trains SentencePiece tokenizers, and tokenizes/caches everything to `data/`, `models/`, and `cache/`. These are all gitignored and get regenerated locally on first run. Every run after that reuses the cache automatically — `RESET_DATASET` is `False` by default, and only needs setting to `True` if you want to force a full rebuild (e.g. after changing vocab size).
   - Hyperparameter tuning (`RUN_TUNING`) is off by default, since the tuned architecture is already saved in `best_params_10k_vocab.json` and gets applied automatically. Set it to `True` to re-run the Optuna search from scratch.
   - Experiment tracking uses Weights & Biases and expects you to be logged in (`wandb login`). Set `wandb_mode: "offline"` in `config.yaml` to skip network calls entirely if you'd rather not.
   - A full training run takes several hours on a single consumer GPU. The notebook checkpoints after every epoch that improves validation loss, and is structured to resume across sessions via the `RESUME_FROM_CHECKPOINT` cells rather than requiring one uninterrupted run.

## Process of creating it

**Data pipeline.** IWSLT17 de-en TED talk transcripts are pulled via Hugging Face `datasets`, split into train/validation/test, cleaned, and tokenized with a SentencePiece BPE tokenizer; results are cached to disk so reruns skip re-downloading and re-tokenizing. Vocabulary size was reduced from an initial 32k down to 10k, matching typical IWSLT-scale convention and cutting embedding/output overfitting capacity. During training, subword segmentation is sampled on the fly (BPE dropout) rather than fixed, so the same sentence can tokenize differently across steps, which improves robustness to segmentation choice. Flash Attention is explicitly disabled on this Windows/CUDA setup to avoid a build-compatibility warning (safe to re-enable on Linux).

**Hyperparameter tuning.** An Optuna search sweeps `d_model`, number of heads, encoder/decoder layer counts, dropout, and weight decay, selected by validation loss. Tuning runs repeatedly stalled early on - traced to Windows silently spilling VRAM overflow into system RAM instead of raising an out-of-memory error, worsened by memory fragmentation across trials - fixed with explicit per-trial cleanup and an NVIDIA Control Panel setting. Tuning trials were also rebuilt to mirror the final run's exact optimizer/schedule (a fixed Noam-style warmup) after discovering the original tuning setup used a completely different, freely-tuned learning rate, meaning whatever hyperparameters it found couldn't actually transfer to the final training configuration.

**Correctness bugs found and fixed along the way:**
- An inverted attention-mask convention caused NaN losses early on.
- The tuned `weight_decay` value from Optuna was silently never applied to the final training run - it trained with a much larger default instead - which, combined with plain Adam's coupled (non-decoupled) weight decay, caused a complete training failure on an architecture that had tuned beautifully.
- A dataset-cache reset utility only scanned the top-level directory for tokenizer files, never the subfolder they actually lived in, so it silently failed to force re-tokenization when the vocab size changed.
- Weights & Biases logging was fully wired up but never actually connected to the trainer, so an entire run's metrics silently went nowhere.
- **The most interesting one:** weight decay applied uniformly to every parameter - including projections that receive a genuinely weak gradient from the loss — caused the encoder's self-attention query/key weights to decay all the way to zero over the course of training. Invisible in the loss curve, but obvious the moment attention weights were actually visualized: the encoder had silently collapsed into uniformly averaging its input instead of attending to anything at all. Because query and key were both exactly zero, the gradient at that point was also exactly zero - a genuine fixed point that no amount of further training could escape on its own. Fixed with a short encoder-only fine-tune: re-initializing the collapsed weights to break the fixed point, and switching to AdamW with weight decay excluded from 1D parameters (biases, LayerNorm gains) so it can't happen again.

**Performance work.** A `.detach().cpu()` call inside every decoder layer's forward pass - triggered whenever attention weights were requested - forced a GPU→CPU sync on every layer, and was the single biggest early slowdown. One mysterious ~70% epoch-time regression that survived several rounds of debugging turned out to be GPU contention from watching videos on the side, not a code issue at all - a good reminder to rule out external causes before digging deeper. A step-timing benchmark was, for a while, running real backward passes and optimizer steps directly on the model about to be trained, nudging its weights before training even started; fixed by benchmarking a disposable copy instead.

**Evaluation methodology.** Greedy decoding was replaced with beam search for final quality reporting, after greedy decoding was found to produce degenerate repetition loops that understated real translation quality. Final benchmark reporting was switched from the validation set (also used for checkpoint selection, and so mildly optimistically biased) to a genuinely held-out test set. An explicit "reload best checkpoint" step was added before evaluation, since the trainer otherwise leaves the model holding the last epoch's weights rather than the best one.

**Infrastructure.** Checkpointing captures model, optimizer, scheduler, and step state together, enabling training to be resumed incrementally across multiple sessions rather than requiring one uninterrupted run. An early, large `utils.py` was split into focused, single-purpose modules, and notebook sections were made flag-gated (toggle a boolean to skip or run a given block) for easier iteration.

The overall arc - from chasing GPU slowdowns, through a model that flatly refused to learn at all, to a subtler bug that only surfaced by visualizing attention directly - is, on its own, as much a demonstration of debugging process as the final BLEU numbers are of translation quality.

## Future considerations

- Cloud usage for tuning, training and testing
- Vocab size changes
