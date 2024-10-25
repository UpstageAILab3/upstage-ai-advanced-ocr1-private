import os
import sys
import lightning.pytorch as pl
import hydra
from lightning.pytorch.callbacks import (  # noqa
    LearningRateMonitor,
    ModelCheckpoint,
)

# sys.path.append(os.getcwd())
# print(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ocr.lightning_modules import get_pl_modules_by_cfg  # noqa: E402

CONFIG_DIR = os.environ.get('OP_CONFIG_DIR') or '../configs'


@hydra.main(config_path=CONFIG_DIR, config_name='train', version_base='1.2')
def train(config):
    """
    Train a OCR model using the provided configuration.

    Args:
        `config` (dict): A dictionary containing configuration settings for training.
    """
    pl.seed_everything(config.get("seed", 42), workers=True)

    model_module, data_module = get_pl_modules_by_cfg(config)

    lr = config.models.optimizer.lr
    weight_decay = config.models.optimizer.weight_decay
    step_size = config.models.scheduler.step_size

    if config.get("wandb"):
        from lightning.pytorch.loggers import WandbLogger as Logger  # noqa: E402
        # logger = Logger(config.exp_name ,
        #                 project=config.project_name,
        #                 config=dict(config),
        #                 reinit=True  # 실험마다 새로운 run을 생성
        #                 )
        logger = Logger(config.exp_name + f"/{lr}_{weight_decay}_{step_size}",
                        project=config.project_name,
                        config=dict(config),
                        reinit=True  # 실험마다 새로운 run을 생성
                        )
    else:
        from lightning.pytorch.loggers.tensorboard import TensorBoardLogger  # noqa: E402
        logger = TensorBoardLogger(
            save_dir=config.log_dir,
            name=config.exp_name,
            version=config.exp_version,
            default_hp_metric=False,
        )

    # lr = config.models.optimizer.lr
    # weight_decay = config.models.optimizer.weight_decay
    # step_size = config.models.scheduler.step_size

    checkpoint_dir =  f"{config.checkpoint_dir}/{lr}_{weight_decay}_{step_size}/"
    # print(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = config.checkpoint_dir
    
    checkpoint_path = checkpoint_dir


    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(dirpath=checkpoint_path,
                        save_top_k=3, monitor='val/loss', mode='min'),
    ]

    trainer = pl.Trainer(
        **config.trainer,
        logger=logger,
        callbacks=callbacks
    )

    trainer.fit(
        model_module,
        data_module,
        ckpt_path=config.get("resume", None),
    )
    trainer.test(
        model_module,
        data_module,
    )
    

if __name__ == "__main__":
    train()
