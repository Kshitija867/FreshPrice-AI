import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import shap
from prophet import Prophet

st.set_page_config(page_title="FreshPrice AI - Live Demo", layout="wide")
st.markdown("<h1 style='text-align: center; color: #2E7D32;'>☘︎ FreshPrice AI</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #555;'>Production Dynamic Pricing Engine Live Demo</h3>", unsafe_allow_html=True)
st.markdown("---")

# ─── IN-MEMORY MODEL CACHING FOR CLOUD PERFORMANCE ───
@st.cache_resource
def load_cloud_models():
    models = {}
    with open("src/models/pricing_xgboost_model.pkl", 'rb') as f:
        models['pricing_engine'] = pickle.load(f)
    products = ['gourmet_sushi_box', 'avocado_chicken_salad', 'organic_almond_milk', 'fresh_butter_croissant']
    for prod in products:
        with open(f"src/models/demand_model_{prod}.pkl", 'rb') as f:
            models[f'demand_{prod}'] = pickle.load(f)
    return models

try:
    MODELS = load_cloud_models()
except Exception as e:
    st.error("Models not found. Please ensure you have run the training scripts and generated the .pkl files inside src/models/ before pushing!")
    st.stop()

# ─── SIDEBAR: MARKET SHOCKS ───
st.sidebar.header("Market Shocks")
weather_mapping = {"Sunny": 0, "Overcast": 1, "Heavy Rain": 2}
selected_weather = st.sidebar.selectbox("Weather", list(weather_mapping.keys()))
is_weekend = 1 if st.sidebar.toggle("Is Weekend?") else 0
is_holiday = 1 if st.sidebar.toggle("Is Holiday?") else 0

# ─── MAIN PANEL ───
st.header("Product Operations State")
col1, col2, col3 = st.columns(3)

with col1:
    product_name = st.selectbox("Product", ['Gourmet Sushi Box', 'Avocado Chicken Salad', 'Organic Almond Milk', 'Fresh Butter Croissant'])
    defaults = {'Gourmet Sushi Box': (12.0, 22.0), 'Avocado Chicken Salad': (5.0, 9.5), 'Organic Almond Milk': (2.5, 5.0), 'Fresh Butter Croissant': (1.0, 3.5)}
    cost_price = st.number_input("Cost Price ($)", value=defaults[product_name][0])

with col2:
    base_price = st.number_input("Base Price ($)", value=defaults[product_name][1])
    competitor_price = st.slider("Competitor Price ($)", min_value=float(cost_price*0.8), max_value=float(base_price*1.5), value=float(base_price*0.95))

with col3:
    inventory_level = st.slider("Inventory Level", 10, 200, 80)
    days_to_expiry = st.slider("Days to Expiry", 0.1, 5.0, 1.2)

if st.button("⚡ Calculate Optimal Dynamic Price", type="primary", use_container_width=True):
    prod_key = product_name.lower().replace(" ", "_")
    prophet_model = MODELS.get(f'demand_{prod_key}')
    pricing_model = MODELS.get('pricing_engine')
    
    # Stage 1: Local Demand Prediction
    future_date = pd.DataFrame([{
        'ds': pd.Timestamp.now().normalize(),
        'is_weekend': is_weekend,
        'is_holiday': is_holiday,
        'weather_condition': weather_mapping[selected_weather]
    }])
    forecast = prophet_model.predict(future_date)
    predicted_demand = max(0, int(forecast['yhat'].values[0]))
    
    # Stage 2: Pricing Logic
    input_data = {
        'cost_price': cost_price, 'base_price': base_price, 'competitor_price': competitor_price,
        'inventory_level': inventory_level, 'days_to_expiry': days_to_expiry,
        'is_weekend': is_weekend, 'is_holiday': is_holiday, 'weather_condition': weather_mapping[selected_weather],
        'stock_out_occurred': 0, 'competitor_to_base_ratio': competitor_price / base_price,
        'inventory_to_demand_ratio': inventory_level / (predicted_demand + 1)
    }
    for p in ['Gourmet Sushi Box', 'Avocado Chicken Salad', 'Organic Almond Milk', 'Fresh Butter Croissant']:
        input_data[f"product_name_{p}"] = 1 if product_name == p else 0
        
    X_shap = pd.DataFrame([input_data]).reindex(columns=pricing_model.get_booster().feature_names).astype(float)
    optimized_price = float(pricing_model.predict(X_shap)[0])
    
    # Post-Processing Optimization Rules
    if days_to_expiry <= 0.25:
        optimized_price = max(cost_price * 1.05, optimized_price * 0.60)
        
    st.success("Analysis Complete!")
    
    # Metrics Display
    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("Predicted Demand", f"{predicted_demand} units")
    rc2.metric("Optimized Price", f"${optimized_price:.2f}")
    rc3.metric("Markdown Applied?", "Yes" if optimized_price < base_price else "No")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 2.5))
    bars = ax.barh(['Cost', 'Optimized', 'Base', 'Competitor'], [cost_price, optimized_price, base_price, competitor_price], color=['#EF5350', '#2E7D32', '#42A5F5', '#FFA726'])
    ax.bar_label(bars, fmt='$%0.2f', padding=5)
    st.pyplot(fig)
    
    # SHAP Explainer
    st.markdown("---")
    st.markdown("### AI Decision Transparency (SHAP Explainer)")
    explainer = shap.TreeExplainer(pricing_model)
    shap_values = explainer(X_shap)
    
    fig_shap, ax_shap = plt.subplots(figsize=(10, 4))
    shap.plots.bar(shap_values[0], max_display=6, show=False)
    plt.title("Feature Impact Magnitude on Price Decision")
    plt.tight_layout()
    st.pyplot(fig_shap)