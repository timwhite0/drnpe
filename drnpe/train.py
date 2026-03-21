import sys
from pathlib import Path

import hydra
import lightning
from hydra.utils import instantiate
from omegaconf import DictConfig

# The directory containing this script
_SCRIPT_DIR = Path(__file__).resolve().parent


@hydra.main(version_base=None, config_path="conf")
def main(cfg: DictConfig):
    lightning.seed_everything(cfg.seed)

    datamodule = instantiate(cfg.datamodule)

    trainer = instantiate(cfg.trainer)

    encoder = instantiate(cfg.encoder)

    trainer.fit(model=encoder, datamodule=datamodule)
    trainer.test(model=encoder, datamodule=datamodule)


if __name__ == "__main__":
    # Add the experiment directory (parent of --config-path) to sys.path
    # so Hydra can find the experiment's data module.
    # Hydra resolves --config-path relative to the calling module's directory,
    # so we do the same here.
    for i, arg in enumerate(sys.argv):
        if arg.startswith("--config-path="):
            raw_path = arg.split("=", 1)[1]
            config_path = (_SCRIPT_DIR / raw_path).resolve()
            experiment_dir = str(config_path.parent)
            if experiment_dir not in sys.path:
                sys.path.insert(0, experiment_dir)
            break
        elif arg == "--config-path" and i + 1 < len(sys.argv):
            raw_path = sys.argv[i + 1]
            config_path = (_SCRIPT_DIR / raw_path).resolve()
            experiment_dir = str(config_path.parent)
            if experiment_dir not in sys.path:
                sys.path.insert(0, experiment_dir)
            break

    main()
