from typing import List, Tuple

BIAS_LABEL_DICT = {"none": 0, "gender": 1, "others": 2}
HATE_LABEL_DICT = {"none": 0, "hate": 1, "offensive": 2}


def load_data(file_path: str) -> List[Tuple[str, int, int]]:
    raw_data = []
    with open(file_path, "r") as f:
        next(f)
        for line in f:
            comment, _, bias_label, hate_label = line.strip().split("\t")
            raw_data.append((comment, BIAS_LABEL_DICT[bias_label], HATE_LABEL_DICT[hate_label]))
    return raw_data
