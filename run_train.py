import torch
import logging
import sys
from transformers import ElectraTokenizer

import pytorch_lightning as pl
from abusing.config import TrainConfig
from abusing.dataset import AbusingDataset
from abusing.utils import load_data
from abusing.module import AbusingClassifier
from torch.utils.data import DataLoader


def main():

    # Config
    config = TrainConfig()
    # Fixing Seed
    pl.seed_everything(config.seed)
    # Logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Data Loading...
    raw_train_instances = load_data(config.train_file_path)
    raw_dev_instances = load_data(config.dev_file_path)
    logger.info(f"훈련용 예시 개수:{len(raw_train_instances)}\t 검증용 예시 개수:{len(raw_dev_instances)}")

    tokenizer = ElectraTokenizer.from_pretrained(config.pretrained_model_name, do_lower_case=False)

    train_dataset = AbusingDataset(raw_train_instances, tokenizer)
    valid_dataset = AbusingDataset(raw_dev_instances, tokenizer)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=AbusingDataset.collate_fn,
    )
    val_dataloader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=AbusingDataset.collate_fn,
    )

    # Lightning
    lightning_module = AbusingClassifier(config,logger)
    trainer = pl.Trainer(
        gpus=config.gpus,
        max_epochs=config.num_epochs,
        deterministic=True,
        weights_save_path=config.save_model_file_prefix,
        gradient_clip_val=1.0,
    )
    trainer.fit(lightning_module, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
