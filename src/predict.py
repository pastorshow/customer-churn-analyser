import pickle
from src.config import MODEL_PATH

def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def predict(model, input_df):
    return model.predict(input_df)