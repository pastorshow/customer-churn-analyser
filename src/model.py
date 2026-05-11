{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "0acae743-62b4-4db6-913d-4d349a2b2a16",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle\n",
    "import pandas as pd\n",
    "\n",
    "MODEL_PATH = \"model.pkl\"\n",
    "\n",
    "def load_model():\n",
    "    with open(MODEL_PATH, \"rb\") as f:\n",
    "        model, encoders = pickle.load(f)\n",
    "    return model, encoders"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "a38b5ff2-b1e0-4f86-87ea-414286c95f4f",
   "metadata": {},
   "outputs": [],
   "source": [
    "def preprocess_input(data, encoders):\n",
    "    for col, le in encoders.items():\n",
    "        if col in data.columns:\n",
    "            data[col] = le.transform(data[col])\n",
    "    return data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "9da2c157-d4e4-4ca2-8acd-45935de93396",
   "metadata": {},
   "outputs": [],
   "source": [
    "def predict(model, encoders, data):\n",
    "    data = preprocess_input(data, encoders)\n",
    "    return model.predict(data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "54a97fd9-74a7-415c-af2b-3218d2b38b92",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
