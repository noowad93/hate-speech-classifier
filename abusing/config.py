import os
import time
from typing import NamedTuple


class TrainConfig(NamedTuple):
    """
    Training Hyperparameters
    """

    #: random seed
    seed: int = 42
    #: epoch 도는 횟수
    num_epochs: int = 3
    #: 훈련 시의 batch size
    batch_size: int = 64
    #: learning rate
    learning_rate: float = 5e-5
    #: max sequence length
    max_sequence_length: int = 128
    #: warm up
    warmup_ratio: float = 0.0

    """
    Data Hyperparameters
    """
    timestamp: str = time.strftime("%m-%d-%Hh%Mm%Ss", time.localtime(time.time()))
    #: training data 파일 경로
    train_file_path: str = "./korean-hate-speech/labeled/train.tsv"
    #: dev data 파일 경로
    dev_file_path: str = "./korean-hate-speech/labeled/dev.tsv"
    pretrained_model_name: str = "monologg/koelectra-base-v3-discriminator"
    #: 모델이 저장될 경로
    save_model_file_prefix: str = "./checkpoints/model"
    train_log_interval: int = 10
    valid_log_interval: int = 200
