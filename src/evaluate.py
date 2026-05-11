from sklearn.metrics import accuracy_score, classification_report
import json
from src.config import METRICS_PATH

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)

    metrics = {
        "accuracy": acc,
        "report": report
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics