# -*- coding: utf-8 -*-

from typing import List

import attr

from model.bert_intent_classifier import BertIntentClassifier


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class IntentPrediction:
    label: str
    confidence: float


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class IntentRequest:
    text: str


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class IntentResponse:
    intents: List[IntentPrediction]


class IntentClassifier:
    def __init__(self, model_path: str):
        self._intent_model = BertIntentClassifier.load(model_path)

    def is_ready(self):
        if self._intent_model:
            return True
        return False

    def predict(self, request: IntentRequest, n_preds: int) -> IntentResponse:
        predictions = self._intent_model.predict(request.text)
        top_predictions = sorted(predictions.items(), key=lambda item: item[1], reverse=True)[
            0:n_preds
        ]
        intent_predictions = [
            IntentPrediction(label=pred[0].value, confidence=pred[1]) for pred in top_predictions
        ]
        return IntentResponse(intents=intent_predictions)


if __name__ == "__main__":
    pass
