import os
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="FreshPrice AI - Dynamic Pricing Engine",
    description="Production API wrapper optimizing quick-commerce perishable item pricing.",
    version="1.0.0"
)

# Global variables to act as our in-memory model registry
MODELS = {}

# Define the expected incoming request data structure using Pydantic
class MarketStateRequest(BaseModel):
    product_name: str
    cost_price: float
    base_price: float
    competitor_price: float
    inventory_level: int
    days_to_expiry: float
    is_weekend: int
    is_holiday: int
    weather_condition: int

@app.on_event("startup")
def load_models_into_memory():
    """Executes on API initialization to ensure zero-latency model retrieval."""
    print(" Loading model registry into memory...")
    
    # 1. Load the core XGBoost Pricing Engine
    pricing_path = "src/models/pricing_xgboost_model.pkl"
    if not os.path.exists(pricing_path):
        raise FileNotFoundError(f"Missing engine binary at {pricing_path}")
    with open(pricing_path, 'rb') as f:
        MODELS['pricing_engine'] = pickle.load(f)
        
    # 2. Load the Prophet Demand Forecasters
    products = ['gourmet_sushi_box', 'avocado_chicken_salad', 'organic_almond_milk', 'fresh_butter_croissant']
    for prod in products:
        prophet_path = f"src/models/demand_model_{prod}.pkl"
        if os.path.exists(prophet_path):
            with open(prophet_path, 'rb') as f:
                MODELS[f'demand_{prod}'] = pickle.load(f)
                
    print(" All models cached successfully. Server ready.")

@app.post("/predict-price")
def predict_optimal_price(request: MarketStateRequest):
    """
    Coordinates a two-stage ML inference process:
    1. Forecasts latent demand via Prophet using external shocks.
    2. Feeds demand into XGBoost to compute the optimal localized selling price.
    """
    # Format product key name to match our files
    prod_key = request.product_name.lower().replace(" ", "_")
    prophet_model = MODELS.get(f'demand_{prod_key}')
    pricing_model = MODELS.get('pricing_engine')
    
    if not prophet_model or not pricing_model:
        raise HTTPException(status_code=500, detail="Requested ML models not fully loaded.")
    
    try:
        # --- STAGE 1: DEMAND FORECASTING (Prophet) ---
        # Generate single-row evaluation frame for Prophet
        future_date = pd.DataFrame([{
            'ds': pd.Timestamp.now().normalize(), # Today's operational date
            'is_weekend': request.is_weekend,
            'is_holiday': request.is_holiday,
            'weather_condition': request.weather_condition
        }])
        
        forecast = prophet_model.predict(future_date)
        predicted_demand = max(0, int(forecast['yhat'].values[0]))
        
        # --- STAGE 2: PRICE OPTIMIZATION (XGBoost) ---
        # Structure payload matching our exact engineered features matrix
        input_data = {
            'cost_price': request.cost_price,
            'base_price': request.base_price,
            'competitor_price': request.competitor_price,
            'inventory_level': request.inventory_level,
            'days_to_expiry': request.days_to_expiry,
            'is_weekend': request.is_weekend,
            'is_holiday': request.is_holiday,
            'weather_condition': request.weather_condition,
            'stock_out_occured': 0,
            'stock_out_occurred': 0,
            'competitor_to_base_ratio': request.competitor_price / request.base_price,
            'inventory_to_demand_ratio': request.inventory_level / (predicted_demand + 1)
        }
        
        # Handle one-hot encoded variables explicitly
        all_products = ['Gourmet Sushi Box', 'Avocado Chicken Salad', 'Organic Almond Milk', 'Fresh Butter Croissant']
        for p in all_products:
            input_data[f"product_name_{p}"] = 1 if request.product_name == p else 0
            
        # Re-index to ensure alignment with XGBoost's exact native feature order
        df_inference = pd.DataFrame([input_data])
        expected_features = pricing_model.get_booster().feature_names
        X_inference = df_inference.reindex(columns=expected_features).astype(float)
        
        # Calculate final pricing decision
        optimized_price = float(pricing_model.predict(X_inference)[0])
        
        return {
            "status": "success",
            "product_name": request.product_name,
            "forecasted_demand": predicted_demand,
            "optimized_selling_price": round(optimized_price, 2),
            "suggested_markdown_applied": "Yes" if optimized_price < request.base_price else "No"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Engine failure: {str(e)}")

