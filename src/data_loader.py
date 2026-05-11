import pandas as pd
from src.config import DATA_PATH

def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop("CustomerID", axis=1)
    return df