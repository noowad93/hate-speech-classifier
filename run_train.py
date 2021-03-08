import torch
import logging
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
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Fixing Seed
    pl.seed_everything(config.seed)

    # Data Loading...
    raw_train_instances = load_data(config.train_file_path)
    raw_dev_instances = load_data(config.dev_file_path)

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
    lightning_module = AbusingClassifier(config, train_dataset, valid_dataset, config.learning_rate)
    trainer = pl.Trainer(gpus=config.gpus, max_epochs=config.num_epochs, deterministic=True)
    trainer.fit(lightning_module, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
