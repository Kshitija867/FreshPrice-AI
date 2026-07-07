# FreshPrice AI – Intelligent Dynamic Pricing for Perishable Inventory

FreshPrice AI is an end-to-end Machine Learning Engineering project that dynamically optimizes the selling price of perishable products in a quick-commerce environment.

The platform combines demand forecasting, price optimization, explainable AI, and a real-time web application to help retailers maximize revenue while reducing inventory waste.

---

# Project Overview

Perishable inventory presents a difficult business challenge. Static pricing often results in either:

- Unsold products reaching expiry
- Lost revenue from underpriced high-demand products

FreshPrice AI solves this problem using a two-stage machine learning pipeline.

1. Forecast future demand using a time-series forecasting model.
2. Optimize the selling price using an XGBoost regression model based on operational conditions.
3. Explain every pricing decision using SHAP explainability.
4. Serve predictions through a FastAPI backend.
5. Visualize insights using an interactive Streamlit dashboard.

---

# System Architecture

```
                 External Factors
          (Weather • Holidays • Weekend)
                       │
                       ▼
             Demand Forecasting Model
                  (Meta Prophet)
                       │
             Forecasted Daily Demand
                       │
                       ▼
         Dynamic Pricing Engine (XGBoost)
                       │
      ┌────────────────┴────────────────┐
      │                                 │
      ▼                                 ▼
 Optimized Price                 SHAP Explainability
      │                                 │
      └──────────────┬──────────────────┘
                     ▼
               FastAPI REST API
                     │
                     ▼
          Streamlit Management Dashboard
```

---

# Features

- Dynamic pricing using machine learning
- Demand forecasting with Prophet
- Price optimization using XGBoost
- Explainable AI using SHAP
- FastAPI REST API for real-time inference
- Interactive Streamlit dashboard
- Synthetic data generation pipeline
- Modular project architecture
- Production-style folder structure

---

# Tech Stack

## Machine Learning

- Python
- XGBoost
- Prophet
- Scikit-learn

## Explainable AI

- SHAP

## Backend

- FastAPI
- Uvicorn

## Frontend

- Streamlit
- Matplotlib

## Data Processing

- Pandas
- NumPy

---

# Repository Structure

```
FreshPrice-AI
│
├── data/
│   └── generated_dataset.csv
│
├── src/
│   ├── api/
│   │   └── main.py
│   │
│   ├── core/
│   │   ├── generate_data.py
│   │   ├── train_demand_forecaster.py
│   │   ├── train_pricing_engine.py
│   │   └── explain_pricing.py
│   │
│   ├── frontend/
│   │   └── app.py
│   │
│   └── models/
│       ├── demand_forecaster.pkl
│       └── pricing_xgboost_model.pkl
│
├── requirements.txt
└── README.md
```

---

# Machine Learning Pipeline

## Stage 1 — Demand Forecasting

The forecasting model predicts future customer demand using:

- Weather conditions
- Weekends
- National holidays
- Historical demand trends

Model Used:

- Meta Prophet

Output:

- Forecasted Daily Demand

---

## Stage 2 — Dynamic Pricing

The pricing model receives operational features including:

- Cost price
- Base price
- Competitor price
- Inventory level
- Days until expiry
- Forecasted demand
- Weather
- Holidays
- Weekend information

Model Used:

- XGBoost Regressor

Output:

- Optimal Selling Price

---

## Explainable AI

FreshPrice AI integrates SHAP (SHapley Additive Explanations) to make every pricing decision transparent.

The dashboard visualizes:

- Most influential features
- Positive feature contributions
- Negative feature contributions
- Relative feature importance

This enables business users to understand why a particular selling price was recommended.

---

# Dashboard

The Streamlit dashboard allows users to:

- Select different products
- Simulate market conditions
- Adjust inventory levels
- Change competitor pricing
- Modify expiry duration
- View forecasted demand
- View optimized selling price
- Visualize SHAP explanations

---

# Installation

Clone the repository.

```bash
git clone https://github.com/Kshitija867/FreshPrice-AI.git
```

Navigate into the project.

```bash
cd FreshPrice-AI
```

Create a virtual environment.

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

# Training the Models

Generate the synthetic dataset.

```bash
python src/core/generate_data.py
```

Train the demand forecasting model.

```bash
python src/core/train_demand_forecaster.py
```

Train the pricing engine.

```bash
python src/core/train_pricing_engine.py
```

---

# Running the Project

## Start the FastAPI Server

```bash
uvicorn src.api.main:app --reload
```

FastAPI Documentation

```
http://127.0.0.1:8000/docs
```

---

## Start the Streamlit Dashboard

```bash
streamlit run src/frontend/app.py
```

Dashboard

```
http://localhost:8501
```

---

# Example Workflow

1. Select a product.
2. Adjust inventory level.
3. Change competitor price.
4. Set weather conditions.
5. Click **Calculate Optimal Dynamic Price**.
6. Receive:

- Forecasted demand
- Recommended selling price
- Markdown recommendation
- SHAP feature explanation

---

# Business Value

FreshPrice AI demonstrates how machine learning can improve retail operations by:

- Reducing perishable inventory waste
- Increasing pricing efficiency
- Responding to changing market conditions
- Improving revenue optimization
- Providing transparent AI-driven decisions

---

# Future Improvements

- Docker containerization
- PostgreSQL integration
- Redis caching
- Real-time weather API integration
- Live competitor price scraping
- Cloud deployment (AWS/Azure)
- CI/CD pipeline using GitHub Actions
- Kubernetes deployment
- MLflow experiment tracking

---

# Author

**Kshitija Santosh Bhagwat**

Bachelor of Engineering – Artificial Intelligence & Data Science

GitHub: https://github.com/Kshitija867

LinkedIn: https://www.linkedin.com/in/kshitija-bhagwat-a19567269

---

# License

This project is intended for educational and portfolio purposes.