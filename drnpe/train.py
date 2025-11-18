import hydra
import lightning
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="conf")
def main(cfg: DictConfig):
    lightning.seed_everything(cfg.seed)

    datamodule = instantiate(cfg.datamodule)

    trainer = instantiate(cfg.trainer)

    encoder = instantiate(cfg.encoder)

    trainer.fit(model=encoder, datamodule=datamodule)
    trainer.test(model=encoder, datamodule=datamodule)


if __name__ == "__main__":
    main()
