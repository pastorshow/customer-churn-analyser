from src.train import train_model
from src.evaluate import evaluate_model

def run_pipeline():
    model, X_test, y_test = train_model()
    metrics = evaluate_model(model, X_test, y_test)

    print("Pipeline completed")
    print(metrics)

if __name__ == "__main__":
    run_pipeline()