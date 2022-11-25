import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Any, Optional

import numpy as np
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import compute_class_weight
from transformers import (
    DistilBertTokenizer,
    DistilBertModel,
    TrainingArguments,
    Trainer,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from scripts.dataset import device, IntentTrainerDataset
from model.intents import Intent

logger = logging.getLogger(__name__)


def sigmoid(_outputs: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-_outputs))


def softmax(_outputs: np.ndarray) -> np.ndarray:
    maxes = np.max(_outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(_outputs - maxes)
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)


class BertIntentClassifier(torch.nn.Module):
    """
    This is the basic DistilBert model with an added dropout layer, and a single layer
    classifier on top that outputs a vector equal in length to the number of labels
    in label_list.
    """

    def __init__(self, model_name: str = "distilbert-base-multilingual-cased"):
        """
        Init a model for identifying the intents of chat text (atis).

        Parameters
        ----------
        model_name : str
            base Module to load from huggingface. default is distillbert multilingual to support
            multiple languages in training data

        Raises
        ------
        ValueError
            Parameters not valid

        """
        super().__init__()

        self._model = DistilBertModel.from_pretrained(model_name, problem_type="")
        self._model.to(device())  # move the model to the GPU, if one is present
        self._tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self._dropout = torch.nn.Dropout(0.1)  # insert a dropout layer set to 10%
        # create a top layer that maps from DistilBert output [= 768 value vector] to a vector
        # equal in length to the number of unique labels/intents.
        # this is a simple single-layer classifier, aka a Perceptron
        self._classifier: Optional[torch.nn.Linear] = None  # load and initialize weights
        self._labels: np.ndarray = np.ndarray([], dtype=object)
        self._class_weights: torch.Tensor = torch.Tensor([])
        self._fine_tune(False)  # enable/disable tunning base model

    def _fine_tune(self, fine_tune: bool) -> None:
        for param in self._model.base_model.parameters():
            param.requires_grad = fine_tune

    @property
    def num_labels(self) -> int:
        return len(self._labels)

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    def _transform(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> List[float]:
        # Pass inputs to the base model, then extract outputs
        logits = self.forward(
            input_ids=input_ids.to(device()), attention_mask=attention_mask.to(device())
        ).logits

        scores = sigmoid(logits.detach().cpu().numpy())
        return [float(f) for f in scores[0].tolist()]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> SequenceClassifierOutput:
        # cf. DistilBertForSequenceClassification.forward for multi_label_classification
        # Pass inputs to the base model, then extract outputs

        if not self._classifier:
            raise RuntimeError("please run self.fit() first")

        bert_output = self._model(
            input_ids=input_ids.to(device()), attention_mask=attention_mask.to(device())
        )

        # For our custom layers
        sequence_output = self._dropout(bert_output[0])
        logits = self._classifier(
            sequence_output[:, 0, :].view(-1, self._model.config.dim)
        )  # calculate losses in logits

        loss = None

        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self._class_weights.to(device()))
            loss = loss_fct(
                logits.view(-1, self.num_labels),
                labels.float().view(-1, self.num_labels),
            )

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=bert_output.hidden_states,
            attentions=bert_output.attentions,
        )

    def label_map(self, lbl_int: int) -> Intent:
        return self._labels[lbl_int]

    def _text_token_vectors(self, txt: str) -> Dict[str, Any]:
        encoding = self._tokenizer(txt, truncation=True)
        input_ids = encoding.data["input_ids"]
        if len(input_ids) == 512:
            return encoding.data
        return {
            "input_ids": input_ids + ([0] * (512 - len(input_ids))),
            "attention_mask": ([1] * len(input_ids)) + ([0] * (512 - len(input_ids))),
        }

    def _label_vector(self, label: np.ndarray) -> List[int]:
        ret = [0] * len(self._labels)
        label_list = self._labels.tolist()
        labels = label.tolist()
        if any(label_ not in label_list for label_ in labels):
            raise ValueError(f"Label '{label}' is not in the set of allowed labels")
        for label_ in labels:
            ret[label_list.index(label_)] = 1
        return ret

    def predict(self, input_str: str) -> Dict[Intent, float]:
        """Predict on fitted model

        Args:
            input_str (str): input text

        Returns:
            prediction scores per intent (dict)
        """
        input_enc = self._text_token_vectors(input_str)
        scores = self._transform(
            input_ids=torch.tensor([input_enc["input_ids"]], device=device()),
            attention_mask=torch.tensor([input_enc["attention_mask"]], device=device()),
        )

        pred_scores = {self.label_map(idx): score for idx, score in enumerate(scores)}
        return pred_scores

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> "BertIntentClassifier":
        """Fit a intent model

        Args:
            x (np.ndarray): Array of text
            y (np.ndarray): Array of Intent Object

        Returns:
            BertIntentClassifier
        """

        self._fine_tune(True)

        y_flat = BertIntentClassifier.flatten_y(y)

        self._labels = np.unique(y_flat)
        self._class_weights = torch.tensor(
            compute_class_weight(class_weight="balanced", classes=self._labels, y=y_flat),
            dtype=torch.float,
        ).to(device())

        self._classifier = torch.nn.Linear(self._model.config.dim, self.num_labels)
        dataset = IntentTrainerDataset(
            [self._text_token_vectors(d) for d in x],
            [self._label_vector(d) for d in y],
            device(),
        )
        trainer = Trainer(
            model=self,
            args=self.training_args(),
            train_dataset=dataset,
        )
        logging.info("Training model")
        trainer.train()
        logging.info("Done training")
        self.eval()
        self._fine_tune(False)
        return self

    @classmethod
    def load(cls, path=Path(__file__).absolute().parent.parent / "model" / "intent_model.bin") \
            -> "BertIntentClassifier":
        model: BertIntentClassifier = torch.load(
            path,
            map_location=torch.device(device()),
        )
        model.predict("Hello world")
        return model

    @classmethod
    def training_args(cls) -> TrainingArguments:
        """
        This is here so we only have to edit one place when we change hyper parameters.

        Returns
        -------
        TrainingArguments
            the train args

        """
        return TrainingArguments(
            num_train_epochs=3,  # total number of training epochs
            output_dir=TemporaryDirectory().name,
            per_device_train_batch_size=8,  # batch size per device during training
            per_device_eval_batch_size=64,  # batch size for evaluation
            warmup_steps=500,  # number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # strength of weight decay
            logging_steps=10,
            dataloader_pin_memory=False,
            save_steps=10000000000,
        )

    @classmethod
    def flatten_y(cls, y: np.ndarray) -> np.ndarray:
        return np.array([label for y_ in y for label in y_], dtype=object)

    @classmethod
    def cv_eval(cls, x: np.ndarray, y: np.ndarray, folds: int = 3) -> np.ndarray:
        """Run Cross Validation and produce hold out predictions.

        Args:
            x (np.ndarray): array of text
            y (np.ndarray): array of Intent
            folds (int, optional): number of cv folds Defaults to 3 to save time.
            Ideally we should 10-fold cross validation

        Returns:
            np.ndarray: array cv holdout estimates of Intent Prediction
        """
        predictions: np.ndarray = np.ndarray((len(y),), dtype=object)

        y_all = np.unique(cls.flatten_y(y))
        for fold_idx, (train_idx, test_idx) in enumerate(
            KFold(n_splits=folds, shuffle=True, random_state=42).split(x, y)
        ):
            logger.info(f"cv-fold: {fold_idx}")
            if len(y_all) != len(np.unique(cls.flatten_y(y[train_idx]))):
                logger.warning("cv fold with incomplete label set")
            model = cls().fit(x[train_idx], y[train_idx])
            for te_idx_ in test_idx:
                predictions[te_idx_] = model.predict(x[te_idx_])

        prediction_values = np.array(
            [
                np.array([k for k, v in p.items() if v >= 0.5], dtype=object)
                for p in predictions[:]
            ],
            dtype=object,
        )
        label_encoder = MultiLabelBinarizer(classes=y_all)

        prediction_values = label_encoder.fit_transform(prediction_values)
        y = label_encoder.fit_transform(y)

        report = classification_report(y, prediction_values, zero_division=0.0, target_names=y_all)
        logging.info("full report")
        logging.info(report)
        report = classification_report(
            y,
            prediction_values,
            zero_division=0.0,
            target_names=y_all,
        )
        logging.info("classification report")
        logging.info(report)

        return predictions
