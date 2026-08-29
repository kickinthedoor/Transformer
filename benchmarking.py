import gc
import copy
import time
import torch


def benchmark_dataloader(train_loader, num_batches=100):
    """Times how long it takes to pull `num_batches` batches from the loader,
    isolating data-pipeline speed from GPU compute."""
    start = time.time()
    for i, _ in enumerate(train_loader):
        if i == num_batches:
            break
    elapsed = time.time() - start
    print(f"Time taken for {num_batches} DataLoader batches: {elapsed:.2f} seconds")
    return elapsed


def benchmark_step_time(model, criterion, train_loader, device, warmup_steps=5, timed_steps=20, bench_lr=1e-4):
    """Times pure GPU compute per training step (forward + backward + optimizer step).

    Benchmarks a disposable deep copy of `model`, never the model you actually
    intend to train — loss.backward() and optimizer.step() run for real here,
    `timed_steps` times, which would otherwise leave your real model's weights
    measurably shifted before real training even begins. The copy gets its own
    fresh optimizer rather than a deep-copied one: an optimizer holds
    references to specific parameter tensors, so deep-copying it alongside the
    model would leave it bound to disconnected tensors the copy's forward pass
    never actually touches.
    """
    bench_model = copy.deepcopy(model).to(device)
    bench_model.train()
    bench_optimizer = torch.optim.AdamW(bench_model.parameters(), lr=bench_lr)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    iterator = iter(train_loader)

    def _step():
        (inp, mask_inp), (tar, mask_tar) = next(iterator)
        inp, tar = inp.to(device, non_blocking=True), tar.to(device, non_blocking=True)
        mask_inp = mask_inp.to(device, non_blocking=True)
        mask_tar_inp = mask_tar[:, :, :, :-1] if mask_tar.dim() == 4 else mask_tar[:, :-1]
        mask_tar_inp = mask_tar_inp.to(device, non_blocking=True)

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            preds = bench_model(inp, tar[:, :-1], enc_mask=mask_inp, dec_mask=mask_tar_inp)
            if isinstance(preds, tuple):
                preds = preds[0]
            loss = criterion(preds.reshape(-1, preds.size(-1)), tar[:, 1:].reshape(-1))
        return loss

    print("Warming up GPU...")
    for _ in range(warmup_steps):
        _step()

    torch.cuda.synchronize()
    start_event.record()

    print(f"Benchmarking {timed_steps} steps...")
    for _ in range(timed_steps):
        loss = _step()
        loss.backward()
        bench_optimizer.step()
        bench_optimizer.zero_grad(set_to_none=True)

    end_event.record()
    torch.cuda.synchronize()

    ms_per_step = start_event.elapsed_time(end_event) / timed_steps
    print(f"\n==========================================")
    print(f"Pure GPU compute time per batch step: {ms_per_step:.2f} ms")
    print(f"==========================================")

    del bench_model, bench_optimizer
    gc.collect()
    torch.cuda.empty_cache()

    return ms_per_step
