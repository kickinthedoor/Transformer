import os
import time
import math
import torch
import torch.nn as nn


def calculate_perplexity(loss_val: float) -> float:
    """Calculates model perplexity from label-smoothed loss."""
    try:
        return math.exp(loss_val)
    except OverflowError:
        return float("inf")


class Trainer:
    def __init__(self, model, optimizer, scheduler, scaler, device, config, logger=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        # bfloat16 does NOT require GradScaler. Set scaler to None when using bfloat16
        self.scaler = scaler if config['training'].get('precision') == 'fp16' else None
        self.device = device
        self.config = config
        self.logger = logger
        
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=self.config['training'].get('label_smoothing_eps', 0.1),
            ignore_index=self.config['data'].get('pad_id', 0)
        )
        
        self.accum_steps = config['training'].get('gradient_accumulation_steps', 1)
        self.log_every_n_steps = config['training'].get('log_every_n_steps', 200)
        self.save_dir = config['paths'].get('checkpoint_dir', './checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)

        self.global_step = 0
        # Separate, never-decreasing counter for wandb's x-axis. global_step reflects
        # the actual best checkpoint (and can rewind on load_checkpoint if training
        # continued past the best epoch before reloading) — using it directly as the
        # wandb step caused "step X is less than the current step Y" warnings and
        # silently dropped log data across resumed sessions.
        self.wandb_step = 0
        self.start_epoch = 0
        self.best_val_loss = float('inf')

    def train_epoch(self, dataloader, epoch: int):
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad(set_to_none=True)
        device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'

        for step, ((inp, mask_inp), (tar, mask_tar)) in enumerate(dataloader):
            inp = inp.to(self.device, non_blocking=True)
            tar = tar.to(self.device, non_blocking=True)
            mask_inp = mask_inp.to(self.device, non_blocking=True)
            mask_tar = mask_tar.to(self.device, non_blocking=True)

            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]
            mask_tar_inp = mask_tar[..., :-1] if mask_tar.size(-1) > tar_inp.size(1) else mask_tar

            # AMP Forward Pass (bfloat16 on Ada Lovelace Tensor Cores)
            with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
                preds = self.model(inp, tar_inp, enc_mask=mask_inp, dec_mask=mask_tar_inp)
                if isinstance(preds, tuple):
                    preds = preds[0]

                loss = self.criterion(
                    preds.reshape(-1, preds.size(-1)), 
                    tar_real.reshape(-1)
                )
                loss = loss / self.accum_steps

            # Backward Pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item() * self.accum_steps

            # Optimizer Step on Accumulation Boundary
            if (step + 1) % self.accum_steps == 0 or (step + 1) == len(dataloader):
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config['training']['grad_clip_norm']
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config['training']['grad_clip_norm']
                    )
                    self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()

                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                self.wandb_step += 1

                if self.logger and self.global_step % self.log_every_n_steps == 0:
                    self.logger.log_metrics({
                        "train/step_loss": loss.item() * self.accum_steps,
                        "train/grad_norm": grad_norm.item(),
                        "train/lr": self.optimizer.param_groups[0]['lr'],
                        "epoch": epoch
                    }, step=self.wandb_step)

        num_batches = len(dataloader)
        avg_loss = total_loss / num_batches
        perplexity = calculate_perplexity(avg_loss)

        if self.logger:
            self.logger.log_metrics({
                "train/epoch_loss": avg_loss,
                "train/epoch_perplexity": perplexity,
                "epoch": epoch
            }, step=self.wandb_step)

        return avg_loss, perplexity

    @torch.no_grad()
    def evaluate(self, dataloader, epoch: int = None):
        self.model.eval()
        total_loss = 0.0
        device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'

        for (inp, mask_inp), (tar, mask_tar) in dataloader:
            inp = inp.to(self.device, non_blocking=True)
            tar = tar.to(self.device, non_blocking=True)
            mask_inp = mask_inp.to(self.device, non_blocking=True)
            mask_tar = mask_tar.to(self.device, non_blocking=True)

            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]
            mask_tar_inp = mask_tar[..., :-1] if mask_tar.size(-1) > tar_inp.size(1) else mask_tar

            with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
                preds = self.model(inp, tar_inp, enc_mask=mask_inp, dec_mask=mask_tar_inp)
                if isinstance(preds, tuple):
                    preds = preds[0]

                loss = self.criterion(
                    preds.reshape(-1, preds.size(-1)), 
                    tar_real.reshape(-1)
                )

            total_loss += loss.item()

        num_batches = len(dataloader)
        avg_loss = total_loss / num_batches
        perplexity = calculate_perplexity(avg_loss)

        if self.logger and epoch is not None:
            self.logger.log_metrics({
                "val/loss": avg_loss,
                "val/perplexity": perplexity,
                "epoch": epoch
            }, step=self.wandb_step)

        return avg_loss, perplexity

    def save_checkpoint(self, path, epoch):
        """Saves everything needed to resume training exactly where it left off:
        model + optimizer + scheduler state, and the trainer's step/epoch counters."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "global_step": self.global_step,
            "wandb_step": self.wandb_step,
            "epoch": epoch,
            "best_val_loss": self.best_val_loss,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """Restores model/optimizer/scheduler state and counters from a checkpoint
        saved by save_checkpoint(), so a subsequent fit() call continues training
        (correct LR position, optimizer momentum, epoch numbering) instead of
        restarting from scratch. The model must already be built with the exact
        same architecture used when the checkpoint was saved."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        # Never rewind — a checkpoint may be from an earlier (better) epoch than
        # the furthest this session has already logged to wandb.
        self.wandb_step = max(self.wandb_step, checkpoint.get("wandb_step", 0))
        self.start_epoch = checkpoint.get("epoch", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        print(
            f"Resumed from '{path}': epoch {self.start_epoch}, "
            f"global_step {self.global_step}, best_val_loss {self.best_val_loss:.4f}"
        )

    def fit(self, train_loader, val_loader, epochs: int, translation_eval_fn=None):
        """Runs `epochs` more epochs, continuing from self.start_epoch (0 unless
        load_checkpoint() was called first)."""
        first_epoch = self.start_epoch + 1
        last_epoch = self.start_epoch + epochs
        print(f"Starting training on device '{self.device}' for epochs {first_epoch}-{last_epoch}...")
        eval_every = self.config['training'].get('eval_every_epochs', 1)

        for epoch in range(first_epoch, last_epoch + 1):
            epoch_start = time.time()
            train_loss, train_ppl = self.train_epoch(train_loader, epoch=epoch)
            val_loss, val_ppl = self.evaluate(val_loader, epoch=epoch)
            epoch_duration = time.time() - epoch_start

            print(
                f"Epoch [{epoch}/{last_epoch}] | "
                f"Train Loss: {train_loss:.4f} (PPL: {train_ppl:.2f}) | "
                f"Val Loss: {val_loss:.4f} (PPL: {val_ppl:.2f}) | "
                f"Duration: {epoch_duration:.1f}s"
            )

            if self.logger:
                self.logger.log_metrics({
                    "epoch_duration_sec": epoch_duration,
                    "train_val_gap": train_loss - val_loss,
                    "epoch": epoch
                }, step=self.wandb_step)

            if translation_eval_fn is not None and epoch % eval_every == 0:
                metrics = translation_eval_fn()
                print(
                    f"  BLEU: {metrics['bleu']:.2f} | chrF: {metrics['chrf']:.2f} | "
                    f"TER: {metrics['ter']:.2f} | NIST: {metrics['nist']:.4f}"
                )
                if self.logger:
                    self.logger.log_metrics({
                        "val/bleu": metrics['bleu'],
                        "val/chrf": metrics['chrf'],
                        "val/ter": metrics['ter'],
                        "val/nist": metrics['nist'],
                        "epoch": epoch
                    }, step=self.wandb_step)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                checkpoint_path = os.path.join(self.save_dir, "best_model.pth")
                self.save_checkpoint(checkpoint_path, epoch)
                print(f"  Saved new best checkpoint to '{checkpoint_path}' (Val Loss: {val_loss:.4f})")
                if self.logger:
                    self.logger.log_summary({"best_val_loss": self.best_val_loss, "best_epoch": epoch})

        self.start_epoch = last_epoch