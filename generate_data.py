import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# set random seed for consistent data generation across runs
np.random.seed(42)

# Configuration: 2 Years of Daily Data
num_days = 730
start_date = datetime(2024, 1, 1)
date_list = [start_date + timedelta(days=x) for x in range(num_days)]

# Define the products, baseline costs, target retail prices, and maximum shelf life
products = {
    'Gourmet Sushi Box': {'cost': 12.0, 'base_price': 22.0, 'shelf_life_days':1},
    'Avocado Chicken Salad': {'cost':5.0, 'base_price': 9.5, 'shelf_life_days':2},
    'Organic Almond Milk': {'cost':2.5, 'base_price':5.0, 'shelf_life_days':5},
    'Fresh Butter Croissant':{'cost': 1.0, 'base_price': 3.5, 'shelf_life_days': 1}
}

data = []

print("Starting realistic market simulation data generation")

for current_date in date_list:
    # 1. Environmental features
    is_weekend = 1 if current_date.weekday() in [5, 6] else 0
    is_holiday = 1 if np.random.rand() < 0.04 else 0      # 4% chance of local holiday

    # weather Simulation (0: sunny/clear, 1: Overcast, 2: Heavy rain)
    # Rain is more likely on weekends to simulate volatile demand spikes
    weather_prob = [0.6, 0.3, 0.1] if not is_weekend else [0.5, 0.3, 0.2]
    weather = np.random.choice([0,1,2], p=weather_prob)

    for prod_name, meta in products.items():
        # 2. Simulate baseline laent demand (organic consumer interest)
        base_demand = np.random.randint(40, 100)

        # Apply external demand modifiers
        if is_weekend: base_demand = int(base_demand* 1.25)
        if is_holiday: base_demand = int(base_demand * 1.40)
        if weather == 2: base_demand = int(base_demand * 1.35) #Rain causes delivery surge

        # Inventory allocated for the day by management (with historical mismatch errors)
        inventory_allocated = int(base_demand * np.random.uniform(0.8, 1.15))

        # Determine average batch age (days remaining before expiring)
        avg_days_to_expiry = np.random.uniform(0.2, meta['shelf_life_days'])

        # 3. Competitor Pricing layer
        competitor_price = round(meta['base_price']* np.random.uniform(0.90, 1.10), 2)

        # 4. Historical pricing strategy (what the store actually charged historically)
        # Includes an erratic human markdown rule if an item is dangerously close to expiry
        expiry_ratio = avg_days_to_expiry / meta['shelf_life_days']
        expiry_markdown = 0.35 if expiry_ratio < 0.25 else 0.0      #discount

        historical_price = round(meta['base_price']* np.random.uniform(0.95, 1.05)*(1-expiry_markdown),2)

        # 5. Price elasticity & realized sales calculations
        # Demand drops exponentially if our price significantly outpaces competitors
        price_ratio = historical_price / competitor_price
        elasticity_factor = np.exp(-1.6 * (price_ratio -1))

        final_demand = int(base_demand * elasticity_factor)
        final_sales = min(final_demand, inventory_allocated)  # cannot sell past physica; inventory limit

        # Financial metrics tracking
        revenue = round(final_sales * historical_price, 2)
        cost_of_goods = round(final_sales * meta['cost'], 2)
        profit = round(revenue - cost_of_goods, 2)

        stock_out = 1 if final_demand > inventory_allocated else 0
        # If stock is old and sales were weak, remaining inventory is wasted
        wastage = max(0, inventory_allocated - final_sales) if expiry_ratio < 0.30 else 0

        data.append({
            'date': current_date.strftime('%Y-%m-%d'),
            'product_name': prod_name, 
            'cost_price': meta['cost'],
            'base_price': meta['base_price'],
            'competitor_price': competitor_price,
            'historical_price': historical_price,
            'inventory_level': inventory_allocated,
            'days_to_expiry': round(avg_days_to_expiry, 2),
            'is_weekend': is_weekend,
            'is_holiday': is_holiday,
            'weather_condition': weather,
            'items_sold': final_sales,
            'revenue': revenue,
            'profit': profit,
            'stock_out_occured': stock_out,
            'inventory_wastage': wastage
        })

# Convert to DataFrame and save to the local data / folder
df = pd.DataFrame(data)
os.makedirs('data', exist_ok=True)
df.to_csv('data/perishable_sales_data.csv', index=False)

print(f" Success! Saved data matrix with {len(df)} rows to 'data/perishable_sales_data.csv'.")




# Choose Product
#       ▼
# Generate Base Demand
#       ▼
# Adjust for Weekend/Holiday/Rain
#       ▼
# Allocate Inventory
#       ▼
# Generate Days to Expiry
#       ▼
# Generate Competitor Price
#       ▼
# Calculate Store Price
#       ▼
# Price affects Demand (Elasticity)
#       ▼
# Calculate Final Sales
#       ▼
# Revenue
#       ▼
# Cost
#       ▼
# Profit
#       ▼
# Check Stock Out
#       ▼
# Check Wastage
#       ▼
# Save Row into data