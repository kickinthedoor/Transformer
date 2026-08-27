import wandb

class ExperimentLogger:
    def __init__(self, config: dict, group: str = None, run_name: str = None, override_config: dict = None):
        """
        Args:
            config: Full dictionary config.
            group: Group name in W&B (e.g., 'optuna_search' or 'final_training').
            run_name: Name for this specific run (e.g., 'trial_1' or 'baseline_model').
            override_config: Dict of Optuna trial params to overlay onto config.
        """
        self.config = config.copy() if config else {}
        
        # Merge Optuna trial params into logged config if provided
        if override_config:
            self.config["optuna_params"] = override_config

        # Extract values or fall back to defaults
        project = self.config.get('experiment', {}).get('wandb_project', 'transformer-project')
        default_name = self.config.get('experiment', {}).get('name', 'experiment')
        mode = self.config.get('experiment', {}).get('wandb_mode', 'online')

        # Initialize W&B run
        self.run = wandb.init(
            project=project,
            name=run_name if run_name else default_name,
            group=group,
            config=self.config,
            mode=mode,  # "offline" removes network I/O from the training loop; sync later with `wandb sync`
            reinit=True,  # Allows multiple sequential runs in Optuna loops
            #settings=wandb.Settings(x_disable_stats=True)  # Disable background system-metrics (GPU/CPU) polling thread
        )

    def _flatten_dict(self, d, parent_key='', sep='.'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def log_metrics(self, metrics: dict, step: int):
        """Logs metrics dictionary (e.g., train_loss, val_loss) to W&B."""
        self.run.log(metrics, step=step)

    def log_summary(self, metrics: dict):
        """Sets one-off best/final values in the run summary (not a time series)."""
        for k, v in metrics.items():
            self.run.summary[k] = v

    def log_figure(self, fig_name: str, figure, step: int = None):
        """Logs matplotlib/plotly figures or images."""
        if step is not None:
            self.run.log({fig_name: wandb.Image(figure)}, step=step)
        else:
            self.run.log({fig_name: wandb.Image(figure)})

    def finish(self):
        """Gracefully close the current run."""
        self.run.finish()