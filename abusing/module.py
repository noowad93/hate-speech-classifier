import logging
from typing import List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch.optim import AdamW
from torch.utils.data import Dataset
from transformers import ElectraModel

from abusing.config import TrainConfig

logger = logging.getLogger("lightning")


class AbusingClassifier(pl.LightningModule):
    def __init__(
        self,
        config: TrainConfig,
        train_dataset: Dataset,
        valid_dataset: Dataset,
        learning_rate: float = 5e-5,
    ):
        super().__init__()
        self.config = config

        self.electra: ElectraModel = ElectraModel.from_pretrained(config.pretrained_model_name)
        self.dense = nn.Linear(self.electra.config.hidden_size, self.electra.config.hidden_size)
        self.dropout = nn.Dropout(self.electra.config.hidden_dropout_prob)

        self.bias_classifier = nn.Linear(self.electra.config.hidden_size, 3)
        self.hate_classifier = nn.Linear(self.electra.config.hidden_size, 3)

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

    def forward(self, *electra_inputs: torch.Tensor) -> torch.Tensor:
        electra_outputs = self.electra.forward(*electra_inputs)
        sequence_outputs = electra_outputs[0]
        pooled_output = self.dense.forward(sequence_outputs[:, 0, :])
        pooled_output = self.dropout(pooled_output)
        pooled_output = F.gelu(pooled_output)  # Electra authors used gelu here
        pooled_output = self.dropout(pooled_output)

        bias_logits = self.bias_classifier.forward(pooled_output)
        hate_logits = self.hate_classifier.forward(pooled_output)
        return bias_logits, hate_logits

    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int):
        *features, bias_labels, hate_labels = batch
        bias_logits, hate_logits = self.forward(*features)
        bias_loss = self.criterion(bias_logits, bias_labels)
        hate_loss = self.criterion(hate_logits, hate_labels)
        train_loss = bias_loss + hate_loss
        return train_loss

    def validation_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int):
        *features, bias_labels, hate_labels = batch
        bias_logits, hate_logits = self.forward(*features)
        return bias_logits, bias_labels, hate_logits, hate_labels

    def validation_epoch_end(self, outputs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
        # called at the end of the validation epoch
        # outputs is an array with what you returned in validation_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}]
        predicted_bias_labels, predicted_hate_labels = [], []
        target_bias_labels, target_hate_labels = [], []
        for output in outputs:
            predicted_bias_labels.extend(torch.argmax(output[0], dim=-1).tolist())
            predicted_hate_labels.extend(torch.argmax(output[2], dim=-1).tolist())
            target_bias_labels.extend(output[1].view(-1).tolist())
            target_hate_labels.extend(output[3].view(-1).tolist())
        bias_report = classification_report(target_bias_labels, predicted_bias_labels)
        hate_report = classification_report(target_hate_labels, predicted_hate_labels)
        logging.info(bias_report)
        logging.info(hate_report)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.learning_rate)
