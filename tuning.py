import gc
import json
import time
import copy
import optuna
import torch
from model import Transformer
from trainer import Trainer
from scheduling import CustomWarmupSchedule


def build_objective(train_loader, val_loader, config, device, max_trial_minutes=40, max_epoch_minutes=10):
    """Factory function that returns an Optuna objective function with fixed data and device.

    max_trial_minutes: a trial is pruned if its cumulative time exceeds this budget,
    checked after each epoch. Bounds worst-case wall-clock time for any single trial
    (e.g. a large d_model/max-layers combo sampled early, before the pruner's
    n_startup_trials have accumulated enough history to prune on quality alone).

    max_epoch_minutes: a trial is pruned immediately if any single epoch alone takes
    longer than this — catches a pathologically slow epoch faster than waiting for
    the cumulative max_trial_minutes budget to be exceeded.
    """
    
    def objective(trial: optuna.Trial):
        # 1. Sample Hyperparameters
        # NOTE: lr is intentionally not tuned here. The final training run uses
        # CustomWarmupSchedule (Noam-style), which derives its own peak LR from
        # d_model/warmup_steps and ignores any freely-chosen lr — so trials mirror
        # that same schedule below rather than searching over a value that
        # wouldn't transfer to the final run anyway.
        d_model = trial.suggest_categorical("d_model", [256, 384,512])
        nhead = trial.suggest_categorical("nhead", [2, 4, 8])
        num_encoder_layers = trial.suggest_int("num_encoder_layers", 2, 6)
        num_decoder_layers = trial.suggest_int("num_decoder_layers", 2, 6)
        dropout = trial.suggest_float("dropout", 0.1, 0.3)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 5e-2, log=True)

        # Ensure d_model is divisible by nhead
        if d_model % nhead != 0:
            raise optuna.TrialPruned()

        # Use deepcopy to prevent mutating global config dictionary across trials
        trial_config = copy.deepcopy(config)
        trial_config['model'].update({
            'd_model': d_model,
            'nhead': nhead,
            'num_encoder_layers': num_encoder_layers,
            'num_decoder_layers': num_decoder_layers,
            'dropout': dropout,
        })

        # 2. Instantiate Model and Trainer
        model = optimizer = scheduler = trainer = None
        try:
            model = Transformer(
                num_enc_layers=num_encoder_layers,
                num_dec_layers=num_decoder_layers,
                d_model=d_model,
                num_heads=nhead,
                dff=4 * d_model,  # scale with d_model instead of a fixed value, matching standard Transformer ratio
                src_vocab=trial_config['data']['vocab_size'],
                tgt_vocab=trial_config['data']['vocab_size'],
                max_pe_src=trial_config['model']['max_pe_source'],
                max_pe_tgt=trial_config['model']['max_pe_target'],
                rate=dropout
            ).to(device)

            # Mirror the final run's optimizer/schedule exactly (Adam lr=1.0, scaled
            # by CustomWarmupSchedule) so tuning results transfer to the real thing.
            optimizer = torch.optim.Adam(
                model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9,
                weight_decay=weight_decay
            )
            scheduler = CustomWarmupSchedule(
                optimizer=optimizer,
                d_model=d_model,
                warmup_steps=trial_config['training'].get('warmup_steps', 4000)
            )

            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=None,
                device=device,
                config=trial_config
            )

            # 3. Train and Evaluate Loop
            epochs = trial_config['training'].get('tune_epochs', 5)
            trial_start = time.time()
            max_trial_seconds = max_trial_minutes * 60
            max_epoch_seconds = max_epoch_minutes * 60

            for epoch in range(1, epochs + 1):
                epoch_start = time.time()
                _ = trainer.train_epoch(train_loader, epoch=epoch)
                val_loss, val_ppl = trainer.evaluate(val_loader, epoch=epoch)
                epoch_seconds = time.time() - epoch_start
                elapsed_min = (time.time() - trial_start) / 60

                print(
                    f"Trial {trial.number} [d_model={d_model}, nhead={nhead}, "
                    f"enc={num_encoder_layers}, dec={num_decoder_layers}, "
                    f"dropout={dropout:.3f}, weight_decay={weight_decay:.5f}] "
                    f"epoch {epoch}/{epochs}: val_loss={val_loss:.4f} "
                    f"(epoch took {epoch_seconds:.1f}s, "
                    f"{elapsed_min:.1f}/{max_trial_minutes} min elapsed)"
                )

                # Report metric for intermediate pruning
                trial.report(val_loss, step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                if epoch_seconds > max_epoch_seconds:
                    print(
                        f"Trial {trial.number} epoch {epoch} took "
                        f"{epoch_seconds / 60:.1f} min, over the {max_epoch_minutes} min "
                        f"per-epoch cap — pruning."
                    )
                    raise optuna.TrialPruned()

                if time.time() - trial_start > max_trial_seconds:
                    print(
                        f"Trial {trial.number} exceeded {max_trial_minutes} min "
                        f"budget after epoch {epoch} — pruning."
                    )
                    raise optuna.TrialPruned()

            return val_loss

        # Gracefully handle CUDA Out-of-Memory without crashing the study
        except torch.cuda.OutOfMemoryError:
            raise optuna.TrialPruned()

        # Free this trial's model/optimizer before the next trial allocates a
        # differently-sized one, to avoid CUDA cache fragmentation across trials
        finally:
            del model, optimizer, scheduler, trainer
            gc.collect()
            torch.cuda.empty_cache()

    return objective


def save_best_params(study, best_params_path="best_params.json"):
    """
    Saves the winning trial's params to disk. Callable standalone at any time,
    e.g. after reloading a study with optuna.load_study(study_name=..., storage=...)
    following a killed/interrupted run — completed trials persist in the storage
    DB regardless of how the process ended, so this doesn't require optimize()
    to have finished.
    """
    best_params_record = {
        "study_name": study.study_name,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_number": study.best_trial.number,
    }
    with open(best_params_path, "w") as f:
        json.dump(best_params_record, f, indent=2)

    print(
        f"Best Parameters Found (trial {best_params_record['best_trial_number']}, "
        f"val_loss={best_params_record['best_value']:.4f}):"
    )
    print(json.dumps(best_params_record['best_params'], indent=2))
    print(f"Saved to {best_params_path}")

    return best_params_record


def run_hyperparameter_search(
    train_loader,
    val_loader,
    config,
    device,
    n_trials=30,
    study_name="transformer_tuning",
    storage="sqlite:///optuna_study.db",
    best_params_path="best_params.json",
    max_trial_minutes=50,
    max_epoch_minutes=10,
    n_startup_trials=3,
):
    """Runs an Optuna search and saves the winning params to `best_params_path`.

    If this gets interrupted (kernel restart, KeyboardInterrupt, crash), completed
    trials are still saved in `storage` — reload the study afterward and call
    save_best_params(study) manually rather than re-running this from scratch.

    n_startup_trials: the pruner can't prune any trial until this many trials have
    completed (Optuna default is 5). Lowered by default here since a small trial
    budget can't afford to guarantee 5 full-length trials regardless of quality.
    """
    objective_fn = build_objective(
        train_loader, val_loader, config, device,
        max_trial_minutes=max_trial_minutes,
        max_epoch_minutes=max_epoch_minutes,
    )

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=n_startup_trials, n_warmup_steps=1),
        storage=storage,
        load_if_exists=True,
    )

    study.optimize(objective_fn, n_trials=n_trials)
    save_best_params(study, best_params_path=best_params_path)

    return study