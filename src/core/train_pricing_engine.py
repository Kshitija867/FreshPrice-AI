import os
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class PricingEnginePipeline:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        self.model = None
        
    def load_and_engineer_features(self):
        """Loads dataset and prepares features for gradient boosting."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Missing dataset matrix at {self.data_path}")
            
        self.df = pd.read_csv(self.data_path)
        
        # Feature Engineering: Help the tree-based model capture business relationships easily
        self.df['competitor_to_base_ratio'] = self.df['competitor_price'] / self.df['base_price']
        self.df['inventory_to_demand_ratio'] = self.df['inventory_level'] / (self.df['items_sold'] + 1)
        
        # One-Hot Encode our categorical variable: product_name
        self.df = pd.get_dummies(self.df, columns=['product_name'], drop_first=False)
        
        print(f"Extracted feature matrix. Shape: {self.df.shape}")

    def train(self):
        """Splits data, optimization parameters, and fits an XGBoost Regressor."""
        # Define our model input features (X) and target variable (y)
        # We drop direct financial results and items_sold because those are lagging indicators
        feature_cols = [col for col in self.df.columns if col not in [
            'date', 'historical_price', 'revenue', 'profit', 'items_sold', 
            'stock_out_occurred', 'inventory_wastage'
        ]]
        
        X = self.df[feature_cols]
        y = self.df['historical_price']
        
        # Split into training and validation matrices (80/20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("Training foundational XGBoost Pricing Engine...")
        self.model = XGBRegressor(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model validity
        preds = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        print(f"Pricing Engine Evaluation Metrics:")
        print(f"   - Root Mean Squared Error (RMSE): ${rmse:.4f}")
        print(f"   - R-squared Score (Variance Explained): {r2*100:.2f}%")
        
        # Store features on the model object for easy pipeline alignment in inference steps
        # self.model.feature_names_in_ = list(X.columns)
        self.model_features = list(X.columns) # Safe custom attribute

    def save_engine(self, output_dir: str = "src/models"):
        """Saves the trained XGBoost model to the model registry."""
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "pricing_xgboost_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Core pricing engine model exported to '{model_path}'!")

if __name__ == "__main__":
    pipeline = PricingEnginePipeline(data_path="data/perishable_sales_data.csv")
    pipeline.load_and_engineer_features()
    pipeline.train()
    pipeline.save_engine()


