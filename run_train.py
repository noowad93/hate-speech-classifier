import torch
import logging
from transformers import ElectraTokenizer

import pytorch_lightning as pl
from abusing.config import TrainConfig
from abusing.dataset import AbusingDataset
from abusing.utils import load_data
from abusing.module import AbusingClassifier


def main():
    # Logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("lightning")

    # Config
    config = TrainConfig()
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Fixing Seed
    pl.seed_everything(0)

    # Data Loading...
    raw_train_instances = load_data(config.train_file_path)
    raw_dev_instances = load_data(config.dev_file_path)
    logger.info(f"학습 데이터 개수: {len(raw_train_instances)}\t개발 데이터 개수: {len(raw_dev_instances)}")

    tokenizer = ElectraTokenizer.from_pretrained(config.pretrained_model_name, do_lower_case=False)

    train_dataset = AbusingDataset(raw_train_instances, tokenizer, config.max_sequence_length)
    valid_dataset = AbusingDataset(raw_dev_instances, tokenizer, config.max_sequence_length)

    lightning_module = AbusingClassifier(config, train_dataset, valid_dataset, config.learning_rate)
    trainer = pl.Trainer(gpus=config.gpus, max_epochs=config.num_epochs, deterministic=True)
    trainer.fit(lightning_module)


if __name__ == "__main__":
    main()
