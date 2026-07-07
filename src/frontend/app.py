import streamlit as pd_st
import requests
import pandas as pd
import matplotlib.pyplot as plt

# Configure page layout
pd_st.set_page_config(
    page_title="FreshPrice AI - Management Console",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom title styling
pd_st.markdown("<h1 style='text-align: center; color: #2E7D32;'> ☘︎ FreshPrice AI</h1>", unsafe_allow_html=True)
pd_st.markdown("<h3 style='text-align: center; color: #555;'>Perishable Inventory & Dynamic Pricing Simulator</h3>", unsafe_allow_html=True)
pd_st.markdown("---")

# Backend API Configuration
API_URL = "http://127.0.0.1:8000/predict-price"

# ─── SIDEBAR: MARKET CONDITIONS ───
pd_st.sidebar.header("Market Environment Shocks")

weather_mapping = {"Sunny / Clear": 0, "Overcast / Cloudy": 1, "Heavy Rain / Delivery Surge": 2}
selected_weather = pd_st.sidebar.selectbox("Current Weather", list(weather_mapping.keys()))
weather_val = weather_mapping[selected_weather]

is_weekend_bool = pd_st.sidebar.toggle("Is Weekend?", value=False)
is_holiday_bool = pd_st.sidebar.toggle("Is National Holiday?", value=False)

is_weekend = 1 if is_weekend_bool else 0
is_holiday = 1 if is_holiday_bool else 0

# ─── MAIN PANEL: PRODUCT SELECTION & METRICS ───
pd_st.header("Product Operational State")

col1, col2, col3 = pd_st.columns(3)

with col1:
    product_name = pd_st.selectbox(
        "Select Product Portfolio Item",
        ['Gourmet Sushi Box', 'Avocado Chicken Salad', 'Organic Almond Milk', 'Fresh Butter Croissant']
    )
    
    # Auto-populate baseline data based on selections to keep it simple
    defaults = {
        'Gourmet Sushi Box': {'cost': 12.0, 'base': 22.0},
        'Avocado Chicken Salad': {'cost': 5.0, 'base': 9.5},
        'Organic Almond Milk': {'cost': 2.5, 'base': 5.0},
        'Fresh Butter Croissant': {'cost': 1.0, 'base': 3.5}
    }
    meta = defaults[product_name]
    
    cost_price = pd_st.number_input("Unit Cost Price ($)", value=meta['cost'], step=0.5)

with col2:
    base_price = pd_st.number_input("Standard Base Retail Price ($)", value=meta['base'], step=0.5)
    competitor_price = pd_st.slider("Immediate Competitor Price ($)", min_value=float(meta['cost']*0.8), max_value=float(meta['base']*1.5), value=float(meta['base']*0.98), step=0.1)

with col3:
    inventory_level = pd_st.slider("Current Dark Store Inventory (Units)", min_value=10, max_value=200, value=75, step=5)
    days_to_expiry = pd_st.slider("Days Remaining Until Expiry", min_value=0.1, max_value=5.0, value=1.5, step=0.1)

# ─── INFERENCE TRIGGER ───
pd_st.markdown("<br>", unsafe_allow_html=True)
if pd_st.button("Calculate Optimal Dynamic Price", type="primary", use_container_width=True):
    
    # Package the payload exactly matching our FastAPI Pydantic schema
    payload = {
        "product_name": product_name,
        "cost_price": cost_price,
        "base_price": base_price,
        "competitor_price": competitor_price,
        "inventory_level": inventory_level,
        "days_to_expiry": days_to_expiry,
        "is_weekend": is_weekend,
        "is_holiday": is_holiday,
        "weather_condition": weather_val
    }
    
    try:
        with pd_st.spinner("Querying Live Engine Microservice..."):
            response = requests.post(API_URL, json=payload)
            
        if response.status_code == 200:
            result = response.json()
            
            # Display Key Output Cards
            pd_st.success("Analysis Complete!")
            res_col1, res_col2, res_col3 = pd_st.columns(3)
            
            with res_col1:
                pd_st.metric(label="Forecasted Demand Volume", value=f"{result['forecasted_demand']} units")
            with res_col2:
                pd_st.metric(label="Optimized Selling Price", value=f"${result['optimized_selling_price']:.2f}")
            with res_col3:
                pd_st.metric(label="Operational Markdown Applied?", value=result['suggested_markdown_applied'])
                
            # Render a clean business logic sanity visualization
            pd_st.markdown("### Pricing Position vs Costs")
            fig, ax = plt.subplots(figsize=(10, 2))
            categories = ['Your Cost', 'Optimized Price', 'Base Price', 'Competitor Price']
            values = [cost_price, result['optimized_selling_price'], base_price, competitor_price]
            colors = ['#EF5350', '#2E7D32', '#42A5F5', '#FFA726']
            
            bars = ax.barh(categories, values, color=colors)
            ax.bar_label(bars, fmt='$%0.2f', padding=5)
            ax.set_xlim(0, max(values) * 1.2)
            plt.title("Financial Stack Comparison ($)")
            pd_st.pyplot(fig)
            
        else:
            pd_st.error(f"Backend Service Error: {response.json().get('detail', 'Unknown error occurred')}")
            
    except requests.exceptions.ConnectionError:
        pd_st.error("Critical Error: Could not connect to the FastAPI server. Make sure your Uvicorn service is running on http://127.0.0.1:8000!")