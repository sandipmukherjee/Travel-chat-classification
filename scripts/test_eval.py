from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer

from model.bert_intent_classifier import BertIntentClassifier
from scripts.train_intents import load_data


def main():
    data_path = Path(__file__).absolute().parent.parent / "data/atis/test.tsv"
    x, y = load_data(data_path)
    model = BertIntentClassifier.load("")
    predictions: np.ndarray = np.ndarray((len(y),), dtype=object)
    for idx, _x in enumerate(x):
        predictions[idx] = model.predict(x[idx])
    prediction_values = np.array(
        [
            np.array([k for k, v in p.items() if v >= 0.5], dtype=object)
            for p in predictions[:]
        ],
        dtype=object,
    )
    y_all = np.unique(BertIntentClassifier.flatten_y(y))
    label_encoder = MultiLabelBinarizer(classes=y_all)

    prediction_values = label_encoder.fit_transform(prediction_values)
    y = label_encoder.fit_transform(y)

    report = classification_report(y, prediction_values, zero_division=0.0, target_names=y_all)
    print("full report")
    print(report)
    report = classification_report(
        y,
        prediction_values,
        zero_division=0.0,
        target_names=y_all,
    )
    print("classification report")
    print(report)


if __name__ == '__main__':
    main()