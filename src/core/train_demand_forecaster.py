import os
import pickle
import pandas as pd
from prophet import Prophet

class DemandForecasterPipeline:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        self.models = {}

    def load_and_preprocess(self):
        """Loads data and converts columns to Prophet standards."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset missing at {self.data_path}")
        
        self.df = pd.read_csv(self.data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        print(f" Loaded {len(self.df)} rows for demand forecasting.")

    def train_product_model(self, product_name: str):
        """Trains a dedicated Prophet model with external regressors for a specific product."""
        print(f"\n Training Demand Forecaster for: {product_name}...")
        
        # Filter for the target item
        prod_df = self.df[self.df['product_name'] == product_name].copy()
        
        # Prophet requires exactly 'ds' (datestamp) and 'y' (target value) columns
        prophet_df = prod_df[[
            'date', 'items_sold', 'is_weekend', 'is_holiday', 'weather_condition'
        ]].rename(columns={'date': 'ds', 'items_sold': 'y'})
        
        # Initialize Prophet with growth tracking and yearly/weekly patterns turned on
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        
        # Inject our external context variables as multivariate regressors
        model.add_regressor('is_weekend')
        model.add_regressor('is_holiday')
        model.add_regressor('weather_condition')
        
        # Fit the model
        model.fit(prophet_df)
        self.models[product_name] = model
        print(f" Trained successfully.")

    def save_models(self, output_dir: str = "src/models"):
        """Saves trained models out to disk as binary objects."""
        os.makedirs(output_dir, exist_ok=True)
        for prod_name, model in self.models.items():
            safe_name = prod_name.lower().replace(" ", "_")
            model_path = os.path.join(output_dir, f"demand_model_{safe_name}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        print(f"\n All model binaries exported to '{output_dir}/' directory!")

if __name__ == "__main__":
    # Initialize and execute the pipeline workflow
    pipeline = DemandForecasterPipeline(data_path="data/perishable_sales_data.csv")
    pipeline.load_and_preprocess()
    
    # Train a model for each unique product in our portfolio
    unique_products = [
        'Gourmet Sushi Box', 
        'Avocado Chicken Salad', 
        'Organic Almond Milk', 
        'Fresh Butter Croissant'
    ]
    
    for product in unique_products:
        pipeline.train_product_model(product)
        
    pipeline.save_models()