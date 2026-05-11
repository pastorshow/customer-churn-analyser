DATA_PATH = "data/customer_churn_dataset.csv"
MODEL_PATH = "models/model.pkl"
METRICS_PATH = "artifacts/metrics.json"

TARGET = "Churn"

CATEGORICAL_COLS = ["Gender", "Subscription Type", "Contract Length"]
NUMERICAL_COLS = [
    "Age", "Tenure", "Usage Frequency",
    "Support Calls", "Payment Delay",
    "Total Spend", "Last Interaction"
]