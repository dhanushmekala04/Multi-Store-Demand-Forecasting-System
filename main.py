from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
from datetime import datetime, timedelta
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(title="Sales Forecasting API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store loaded models
store_metadata = None
all_scalers = None
all_models = {}
df = None
forecast_df = None

LOOKBACK = 30
FORECAST_DAYS = 30

class ForecastRequest(BaseModel):
    store_id: int
    item_id: int

class ForecastResponse(BaseModel):
    store: int
    item: int
    dates: List[str]
    forecasted_sales: List[float]
    avg_sales: float
    total_sales: float
    min_sales: float
    max_sales: float

@app.on_event("startup")
async def load_models():
    """Load all models, scalers, and data on startup"""
    global store_metadata, all_scalers, all_models, df, forecast_df
    
    try:
        import os
        
        # Determine base path - check if models are in 'all_models' subfolder
        base_path = ''
        if os.path.exists('all_models'):
            base_path = 'all_models/'
            print(f"✓ Found 'all_models' directory")
        
        # Load metadata
        metadata_path = f'{base_path}store_metadata.pkl'
        with open(metadata_path, 'rb') as f:
            store_metadata = pickle.load(f)
        print(f"✓ Loaded metadata for {len(store_metadata)} stores")
        
        # Load scalers
        scalers_path = f'{base_path}scalers.pkl'
        with open(scalers_path, 'rb') as f:
            all_scalers = pickle.load(f)
        print(f"✓ Loaded scalers for {len(all_scalers)} stores")
        
        # Load models
        for store_id in store_metadata.keys():
            model_path = f'{base_path}model_store_{store_id}.keras'
            model = keras.models.load_model(model_path, compile=False)
            all_models[store_id] = model
            print(f"✓ Loaded {model_path}")
        
        # Load full data
        data_path = f'{base_path}full_data.csv'
        df = pd.read_csv(data_path)
        # Handle DD-MM-YYYY format (like "13-01-2013")
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', dayfirst=True)
        print(f"✓ Loaded full_data.csv - Shape: {df.shape}")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Generate all forecasts
        generate_all_forecasts()
        
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def generate_all_forecasts():
    """Generate forecasts for all store-item combinations"""
    global forecast_df
    
    all_forecasts = []
    forecast_start_date = df['date'].max() + timedelta(days=1)
    forecast_dates = pd.date_range(forecast_start_date, periods=FORECAST_DAYS, freq='D')
    
    for store_id in sorted(all_models.keys()):
        model = all_models[store_id]
        scalers = all_scalers[store_id]
        metadata = store_metadata[store_id]
        store_items_list = metadata['items']
        
        for item_idx, item in enumerate(store_items_list):
            # Get historical data
            mask = (df['store'] == store_id) & (df['item'] == item)
            subset = df[mask].sort_values('date')
            
            if len(subset) < LOOKBACK + 60:
                continue
            
            sales = subset['sales'].values
            
            # Extract features
            trend = pd.Series(sales).rolling(window=7, center=True).mean()
            trend = trend.bfill().ffill().values
            trend_value = trend[-1]
            trend_slope = (trend[-1] - trend[-LOOKBACK]) / LOOKBACK
            
            detrended = sales - trend
            seasonal = np.zeros_like(sales)
            for day in range(7):
                mask_day = np.arange(day, len(sales), 7)
                seasonal[mask_day] = np.mean(detrended[mask_day])
            
            seasonal_strength = np.std(seasonal[-LOOKBACK:])
            seasonal_recent_avg = np.mean(seasonal[-7:])
            
            # Prepare inputs
            last_sequence = sales[-LOOKBACK:]
            last_sequence_scaled = scalers['X'].transform(last_sequence.reshape(1, -1))
            
            trend_features = np.array([[trend_value, trend_slope]])
            trend_features_scaled = scalers['trend'].transform(trend_features)
            
            seasonal_features = np.array([[seasonal_strength, seasonal_recent_avg]])
            seasonal_features_scaled = scalers['seasonal'].transform(seasonal_features)
            
            # Predict
            pred_scaled = model.predict([
                last_sequence_scaled,
                np.array([item_idx]),
                trend_features_scaled,
                seasonal_features_scaled
            ], verbose=0)
            
            pred = scalers['y'].inverse_transform(pred_scaled)[0]
            pred_30_days = pred[:FORECAST_DAYS]
            
            # Add irregular component
            historical_std = np.std(sales[-LOOKBACK:])
            irregular = np.random.normal(0, historical_std * 0.1, FORECAST_DAYS)
            final_pred = pred_30_days + irregular
            final_pred = np.maximum(final_pred, 0)
            
            # Store forecasts
            for day_idx, forecast_date in enumerate(forecast_dates):
                all_forecasts.append({
                    'date': forecast_date.date(),
                    'store': store_id,
                    'item': item,
                    'forecasted_sales': final_pred[day_idx]
                })
    
    forecast_df = pd.DataFrame(all_forecasts)
    print(f"✓ Generated {len(forecast_df)} forecast records")

@app.get("/")
async def root():
    return {
        "message": "Sales Forecasting API",
        "version": "1.0",
        "endpoints": ["/forecast", "/stores", "/items", "/health"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": len(all_models),
        "forecasts_generated": len(forecast_df) if forecast_df is not None else 0
    }

@app.get("/stores")
async def get_stores():
    """Get list of available stores"""
    if store_metadata is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    # Convert numpy int64 to Python int
    stores = [int(store_id) for store_id in sorted(store_metadata.keys())]
    return {"stores": stores}

@app.get("/items/{store_id}")
async def get_items(store_id: int):
    """Get items for a specific store"""
    if store_metadata is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    if store_id not in store_metadata:
        raise HTTPException(status_code=404, detail=f"Store {store_id} not found")
    
    # Convert numpy types to Python native types
    items = [int(item) for item in store_metadata[store_id]['items']]
    return {"store": store_id, "items": items}

@app.post("/forecast", response_model=ForecastResponse)
async def get_forecast(request: ForecastRequest):
    """Get forecast for specific store and item"""
    if forecast_df is None:
        raise HTTPException(status_code=500, detail="Forecasts not generated")
    
    # Filter forecast data
    mask = (forecast_df['store'] == request.store_id) & (forecast_df['item'] == request.item_id)
    forecast_data = forecast_df[mask].sort_values('date')
    
    if len(forecast_data) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No forecast found for store {request.store_id}, item {request.item_id}"
        )
    
    # Calculate statistics and convert numpy types to Python native types
    sales_values = forecast_data['forecasted_sales'].values
    
    return ForecastResponse(
        store=int(request.store_id),
        item=int(request.item_id),
        dates=[str(d) for d in forecast_data['date'].values],
        forecasted_sales=[float(x) for x in sales_values.tolist()],
        avg_sales=float(np.mean(sales_values)),
        total_sales=float(np.sum(sales_values)),
        min_sales=float(np.min(sales_values)),
        max_sales=float(np.max(sales_values))
    )

@app.get("/historical/{store_id}/{item_id}")
async def get_historical(store_id: int, item_id: int, days: int = 90):
    """Get historical sales data"""
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    mask = (df['store'] == store_id) & (df['item'] == item_id)
    historical = df[mask].sort_values('date').tail(days)
    
    if len(historical) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No historical data for store {store_id}, item {item_id}"
        )
    
    # Convert pandas Timestamps to strings properly
    # Use pd.Timestamp to handle numpy.datetime64 objects
    dates_list = [pd.Timestamp(d).strftime('%Y-%m-%d') for d in historical['date'].values]
    
    return {
        "store": int(store_id),
        "item": int(item_id),
        "dates": dates_list,
        "sales": [float(s) for s in historical['sales'].tolist()]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)