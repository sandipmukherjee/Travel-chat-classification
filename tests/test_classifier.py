import pytest
from model.bert_intent_classifier import BertIntentClassifier
from model.intents import Intent
from scripts.dataset import device


@pytest.fixture
def bert_model():
    model = BertIntentClassifier.load()
    yield model.to(device())


def test_predict(bert_model):
    predictions = bert_model.predict("I want to book a flight")
    top_predictions = sorted(predictions.items(), key=lambda item: item[1], reverse=True)[
                      0:3
                      ]
    assert top_predictions[0][0] == Intent.FLIGHT
    prediction = bert_model.predict("I need to contact ground stuff")
    print(prediction)
    assert len(prediction.keys()) == len(bert_model._labels)