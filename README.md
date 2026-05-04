# 🚀 Customer Churn Prediction Pipeline

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-learn](https://img.shields.io/badge/ML-Scikit--learn-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Production--Ready-success)

## 🎯 Project Summary

A production-ready machine learning system that predicts customer churn using structured customer data.
Built with a modular pipeline, this project demonstrates how to move from raw data to a deployable ML application.

👉 Designed for real-world use: reproducible, scalable, and deployable.

## 🔍 Why This Project Matters

Customer churn directly impacts revenue. This system helps:

* Identify at-risk customers early
* Enable targeted retention strategies
* Support data-driven decision making

## ⚙️ What This Project Does

✔ Loads and cleans customer data
✔ Applies automated preprocessing (no manual encoding issues)
✔ Trains a machine learning model
✔ Evaluates and stores performance metrics
✔ Serves predictions via an interactive web app

## 🧠 Tech Stack

* Python
* Pandas & NumPy
* Scikit-learn
* Streamlit

## 🏗️ Architecture (Simplified)

```id="g9a7gq"
Raw Data → Preprocessing Pipeline → Model Training → Evaluation → Saved Model → Streamlit App
```

## 📁 Project Structure

```id="9rm3p9"
├── app/              # Streamlit UI
├── src/              # Core ML logic
├── pipeline/         # Training pipeline
├── data/             # Dataset
├── models/           # Trained model
├── artifacts/        # Metrics
```

## ▶️ Quick Start

### 1. Install dependencies

```bash id="h9xjfw"
pip install -r requirements.txt
```

### 2. Train the model

```bash id="1l8zvv"
python pipeline/training_pipeline.py
```

### 3. Run the app

```bash id="r3zbxh"
streamlit run app/app.py
```

## 📊 Model Overview

* Model: Random Forest Classifier
* Handles both numerical and categorical features
* Uses pipeline-based preprocessing for consistency

## 💡 Key Strengths

* Clean, modular code structure
* End-to-end ML workflow (not just a notebook)
* Reproducible pipeline
* Ready for deployment

## 👤 Author

Shoyemi Jonathan Showunmi

## ⭐ If You Find This Useful

Give the repo a star or fork it to build on it.

That’s what actually separates “good repo” from “gets interviews.”
