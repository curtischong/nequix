from nequix.config import TrainerConfig


def _format_percentage(fraction: float) -> str:
    percentage = 100.0 * fraction
    if percentage.is_integer():
        return str(int(percentage))
    return f"{percentage:g}".replace(".", "p")


def wandb_run_name(config: TrainerConfig) -> str:
    """Build the data-schedule-prefixed run name used by the training configs."""
    if config.wandb_run_name:
        return config.wandb_run_name

    run_name = config.run_name or config.name
    dataset_name = config.dataset_name
    if not dataset_name:
        return run_name

    train_fraction = float(config.train_frac)
    fraction_suffix = "" if train_fraction == 1.0 else _format_percentage(train_fraction)
    return f"{dataset_name}{fraction_suffix}_{config.n_epochs}ep_{run_name}"
