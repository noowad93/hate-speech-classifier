from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import ElectraTokenizer

InstanceType = Tuple[torch.Tensor, ...]


class AbusingDataset(Dataset):
    def __init__(
        self,
        instances: List[Tuple[str, int, int]],
        tokenizer: ElectraTokenizer,
        max_seq_len: int,
    ):
        self.instances = instances
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __getitem__(self, index: int) -> InstanceType:
        features = self.tokenizer.encode_plus(
            text=self.instances[index][0], add_special_tokens=True, max_length=self.max_seq_len
        )
        bias_label = torch.tensor(self.instances[index][1])
        hate_label = torch.tensor(self.instances[index][2])
        features = [torch.tensor(features[key]) for key in ["input_ids", "attention_mask", "token_type_ids"]]
        return tuple(features + [bias_label] + [hate_label])

    def __len__(self) -> int:
        return len(self.instances)

    @staticmethod
    def collate_fn(batch: List[InstanceType]):
        input_ids = pad_sequence([features[0] for features in batch], batch_first=True, padding_value=0)
        attention_mask = pad_sequence([features[1] for features in batch], batch_first=True, padding_value=0)
        token_type_ids = pad_sequence([features[2] for features in batch], batch_first=True, padding_value=0)
        bias_labels = torch.tensor([features[3] for features in batch])
        hate_labels = torch.tensor([features[4] for features in batch])
        return input_ids, attention_mask, token_type_ids, bias_labels, hate_labels
