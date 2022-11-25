import csv
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
from model.bert_intent_classifier import BertIntentClassifier
from model.intents import Intent


def read_training_data(path: str) -> List[Dict]:
    """
    Read training data from csv file with labels

    Returns
    -------
    List
        Dictionary containing chat text and list of intents

    """

    annotated_chats = []
    with open(
        path,
    ) as csvfile:

        reader = csv.reader(csvfile, delimiter="\t")
        for row in reader:
            chat_data = dict()
            chat_data["text"] = row[0]
            intents = [intent.strip() for intent in row[1].split("+")]
            if any([intent not in Intent._value2member_map_ for intent in intents]):
                continue
            chat_data["intents"] = [Intent(intent.strip()) for intent in row[1].split("+")]
            annotated_chats.append(chat_data)
    return annotated_chats


def load_data(train_data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    training_data = read_training_data(train_data_path)
    chat_texts = [d["text"] for d in training_data]
    x: np.ndarray = np.array(chat_texts)
    y: np.ndarray = np.array(
        [
            np.array(
                [label for label in d["intents"]],
                dtype=Intent,
            )
            for d in training_data
        ],
        dtype=np.ndarray,
    )
    return x, y


def train_cv():
    data_path = Path(__file__).absolute().parent.parent / "data/atis/train.tsv"
    x, y = load_data(data_path)
    print(x.shape)
    print("Data loaded...")
    BertIntentClassifier().cv_eval(x, y)
    print("Cross validation complete..report is logged while running.")

    # Do full training
    model = BertIntentClassifier().fit(x, y)
    torch.save(model, Path(__file__).absolute().parent.parent / "model/intent_model.bin")
    print("model saved")


if __name__ == "__main__":
    train_cv()
