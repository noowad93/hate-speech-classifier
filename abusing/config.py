import time
from typing import NamedTuple


class TrainConfig(NamedTuple):
    """
    Training Hyperparameters
    """

    #: random seed
    seed: int = 42
    #: 사용할 gpu 갯수
    gpus: int = 1
    #: epoch 도는 횟수
    num_epochs: int = 5
    #: 훈련 시의 batch size
    batch_size: int = 64
    #: learning rate
    learning_rate: float = 5e-5
    #: warm up
    warmup_ratio: float = 0.0
    #: num workers
    num_workers: int = 20

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
