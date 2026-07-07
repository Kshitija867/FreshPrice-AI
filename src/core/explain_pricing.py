import os
import pickle
import pandas as pd
import shap
import matplotlib.pyplot as plt

class PricingExplainer:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.explainer = None
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Pricing model missing at {self.model_path}. Run Step 4 first.")
            
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
            
        # Initialize the SHAP TreeExplainer designed specifically for tree-based models like XGBoost
        self.explainer = shap.TreeExplainer(self.model)
        print(" SHAP TreeExplainer successfully initialized for the Pricing Engine.")

    def explain_single_transaction(self, sample_input: dict):
        """
        Takes a raw real-time operational state, formats it, dynamically aligns 
        it with the model's expected features, and computes SHAP values.
        """
        # Convert raw inputs to a single-row DataFrame
        df_input = pd.DataFrame([sample_input])
        
        # Feature Engineering
        df_input['competitor_to_base_ratio'] = df_input['competitor_price'] / df_input['base_price']
        df_input['inventory_to_demand_ratio'] = df_input['inventory_level'] / (df_input['expected_demand'] + 1)
        
        # If the model expects a stock_out feature that wasn't provided, pass a default safety value (0)
        if 'stock_out_occured' not in df_input.columns:
            df_input['stock_out_occured'] = 0
        if 'stock_out_occurred' not in df_input.columns:
            df_input['stock_out_occurred'] = 0
        
        # Reconstruct the dummy-encoded columns for product names
        all_products = ['Gourmet Sushi Box', 'Avocado Chicken Salad', 'Organic Almond Milk', 'Fresh Butter Croissant']
        for prod in all_products:
            col_name = f"product_name_{prod}"
            df_input[col_name] = 1 if sample_input['product_name'] == prod else 0
            
        # Drop the unencoded product_name string column
        df_input = df_input.drop(columns=['product_name'], errors='ignore')
        
        #  THE PRODUCTION FIX: Dynamically extract the exact feature order the model expects
        expected_features = self.model.get_booster().feature_names
        
        # Reindex the dataframe to match that exact list and order perfectly
        X_inference = df_input.reindex(columns=expected_features).astype(float)
        
        # 1. Compute predicted price
        predicted_price = self.model.predict(X_inference)[0]
        
        # 2. Compute SHAP Values
        shap_values = self.explainer(X_inference)
        
        print(f"\n [Prediction Engine Output]")
        print(f"   Target Product : {sample_input['product_name']}")
        print(f"   Calculated Optimal Selling Price: ${predicted_price:.2f}")
        print(f"\n [SHAP Explainer Contributions]")
        
        # Display feature contributions to the terminal
        feature_contributions = zip(expected_features, shap_values.values[0])
        for feat, val in sorted(feature_contributions, key=lambda x: abs(x[1]), reverse=True):
            if abs(val) > 0.01:
                direction = " Increased price by" if val > 0 else " Decreased price by"
                print(f"   - {feat:<32}: {direction} ${abs(val):.2f}")
                
        return predicted_price, shap_values

if __name__ == "__main__":
    # Simulate a critical operational scenario: An Avocado Chicken Salad 
    # that is expiring very soon (0.3 days left) during a massive delivery rainstorm (weather = 2)
    mock_market_state = {
        'product_name': 'Avocado Chicken Salad',
        'cost_price': 5.0,
        'base_price': 9.5,
        'competitor_price': 9.2,
        'inventory_level': 85,
        'expected_demand': 110, # Sourced from our Prophet forecast step
        'days_to_expiry': 0.3,   # Warning: Expiring quickly!
        'is_weekend': 0,
        'is_holiday': 0,
        'weather_condition': 2   # Heavy Rain
    }
    
    explainer_service = PricingExplainer(model_path="src/models/pricing_xgboost_model.pkl")
    explainer_service.explain_single_transaction(mock_market_state)