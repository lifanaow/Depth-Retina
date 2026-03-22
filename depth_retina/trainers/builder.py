from importlib import import_module


def get_trainer(config):
    assert "trainer" in config and config.trainer is not None and config.trainer != '', "Trainer not specified. Config: {0}".format(
        config)
    try:
        Trainer = getattr(import_module(
            f"trainers.{config.trainer}_trainer"), 'Trainer')
    except ModuleNotFoundError as e:
        raise ValueError(f"Trainer {config.trainer}_trainer not found.") from e
    return Trainer
