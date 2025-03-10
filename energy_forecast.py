"""
Uses time-series forecasting models to predict power consumption patterns.
Integrates with weather data to improve prediction accuracy.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnergyForecaster:
    def __init__(self, model_type='lstm', forecast_horizon=24, use_weather=True):
        """
        Initialize the Energy Forecaster.
        
        Parameters:
        -----------
        model_type : str
            Type of forecasting model to use ('lstm' or 'prophet')
        forecast_horizon : int
            Number of hours to forecast ahead
        use_weather : bool
            Whether to include weather data in predictions
        """
        # Store configuration
        self.model_type = model_type.lower()
        self.forecast_horizon = forecast_horizon
        self.use_weather = use_weather
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Path configurations - adjust as needed
        self.data_path = "./data/energy_consumption.csv"
        self.weather_path = "./data/weather_data.csv"
        self.model_save_path = f"./models/energy_{model_type}_model.pkl"
        
        logger.info(f"Initialized {model_type} energy forecaster with {forecast_horizon}h horizon")
        
    def _load_data(self):
        """Load historical energy consumption data and optionally weather data."""
        try:
            # Load energy consumption data
            energy_df = pd.read_csv(self.data_path, parse_dates=['timestamp'])
            energy_df.set_index('timestamp', inplace=True)
            logger.info(f"Loaded energy data with shape: {energy_df.shape}")
            
            # Optionally load and merge weather data
            if self.use_weather:
                weather_df = pd.read_csv(self.weather_path, parse_dates=['timestamp'])
                weather_df.set_index('timestamp', inplace=True)
                
                # Merge datasets
                data = pd.merge(energy_df, weather_df, left_index=True, right_index=True, how='inner')
                logger.info(f"Merged with weather data. Combined shape: {data.shape}")
            else:
                data = energy_df
                
            return data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _prepare_lstm_data(self, data, sequence_length=24):
        """
        Prepare data for LSTM model by creating sequences.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Time series data
        sequence_length : int
            Number of time steps to include in each sequence
            
        Returns:
        --------
        X, y : Training data sequences and target values
        """
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[i:i+sequence_length])
            y.append(scaled_data[i+sequence_length, 0])  # Target is consumption value
            
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Split into training and testing sets (80/20)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        logger.info(f"Prepared LSTM data: {X_train.shape} training sequences, {X_test.shape} test sequences")
        return X_train, y_train, X_test, y_test
    
    def _build_lstm_model(self, input_shape):
        """Build and compile an LSTM model for time series forecasting."""
        logger.info("Building LSTM model...")
        
        # Create a Sequential model
        model = Sequential([
            # LSTM layers
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),  # Prevent overfitting
            LSTM(50),
            Dropout(0.2),
            
            # Output layer - single value prediction
            Dense(1)
        ])
        
        # Compile model
        model.compile(optimizer='adam', loss='mean_squared_error')
        logger.info("LSTM model built successfully")
        
        return model
    
    def _build_prophet_model(self, data):
        """Prepare and create a Prophet model."""
        # Prophet requires specific column names
        prophet_data = data.reset_index()
        prophet_data = prophet_data.rename(columns={'timestamp': 'ds', 'consumption': 'y'})
        
        # Initialize and fit Prophet model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            changepoint_prior_scale=0.05  # Flexibility of trend
        )
        
        # Add weather regressors if available
        if self.use_weather and 'temperature' in prophet_data.columns:
            model.add_regressor('temperature')
            logger.info("Added temperature as regressor in Prophet model")
            
        if self.use_weather and 'humidity' in prophet_data.columns:
            model.add_regressor('humidity') 
            logger.info("Added humidity as regressor in Prophet model")
            
        return model, prophet_data
    
    def train(self):
        """Train the forecasting model based on historical data."""
        # Load data
        data = self._load_data()
        
        # Train appropriate model type
        if self.model_type == 'lstm':
            X_train, y_train, X_test, y_test = self._prepare_lstm_data(data)
            
            # Build LSTM model
            model = self._build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
            
            # Train model with early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5, restore_best_weights=True)
            
            history = model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=1
            )
            
            # Evaluate model
            loss = model.evaluate(X_test, y_test, verbose=0)
            logger.info(f"LSTM model trained. Test loss: {loss:.4f}")
            
            # Save model and scaler
            model.save(self.model_save_path.replace('.pkl', '.h5'))
            self.model = model
            
        elif self.model_type == 'prophet':
            # Build and fit Prophet model
            model, prophet_data = self._build_prophet_model(data)
            model.fit(prophet_data)
            logger.info("Prophet model trained successfully")
            
            # Save model
            with open(self.model_save_path, 'wb') as f:
                joblib.dump(model, f)
            self.model = model
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
        logger.info(f"{self.model_type.upper()} model training completed")
    
    def predict(self, prediction_date=None):
        """
        Generate energy demand predictions.
        
        Parameters:
        -----------
        prediction_date : datetime or None
            Date to start predictions from. Uses current date if None.
            
        Returns:
        --------
        pandas.DataFrame with predictions
        """
        if self.model is None:
            try:
                self._load_model()
            except:
                logger.error("No trained model found. Please train the model first.")
                return None
                
        if prediction_date is None:
            prediction_date = datetime.now()
            
        logger.info(f"Generating predictions from {prediction_date}")
        
        # Make predictions based on model type
        if self.model_type == 'lstm':
            # Load recent data for making predictions
            recent_data = self._load_data()
            
            # Scale data
            scaled_data = self.scaler.transform(recent_data)
            
            # Create input sequence
            input_seq = scaled_data[-24:].reshape(1, 24, scaled_data.shape[1])
            
            # Generate predictions
            predictions = []
            for i in range(self.forecast_horizon):
                # Predict next value
                next_value = self.model.predict(input_seq, verbose=0)
                predictions.append(next_value[0, 0])
                
                # Update input sequence
                new_point = np.zeros((1, 1, scaled_data.shape[1]))
                new_point[0, 0, 0] = next_value[0, 0]
                
                # For simplicity, we'll assume other features stay constant
                if scaled_data.shape[1] > 1:
                    new_point[0, 0, 1:] = input_seq[0, -1, 1:]
                    
                input_seq = np.concatenate([input_seq[:, 1:, :], new_point], axis=1)
                
            # Convert predictions back to original scale
            original_scale_preds = self.scaler.inverse_transform(
                np.column_stack([predictions, np.zeros((len(predictions), scaled_data.shape[1]-1))])
            )[:, 0]
            
            # Create prediction DataFrame
            prediction_dates = [prediction_date + timedelta(hours=i) for i in range(self.forecast_horizon)]
            predictions_df = pd.DataFrame({
                'timestamp': prediction_dates,
                'predicted_consumption': original_scale_preds
            })
            
        elif self.model_type == 'prophet':
            # Create future dataframe
            future = self.model.make_future_dataframe(
                periods=self.forecast_horizon,
                freq='H',
                include_history=False
            )
            
            # Add weather regressors if using weather data
            # (In a real system, you'd need to get weather forecasts)
            if self.use_weather:
                # Dummy weather data - in a real system, you'd use forecasts
                future['temperature'] = 20  # Placeholder
                future['humidity'] = 60     # Placeholder
            
            # Make predictions
            forecast = self.model.predict(future)
            
            # Extract prediction data
            predictions_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            predictions_df = predictions_df.rename(columns={
                'ds': 'timestamp',
                'yhat': 'predicted_consumption',
                'yhat_lower': 'prediction_lower',
                'yhat_upper': 'prediction_upper'
            })
        
        logger.info(f"Generated {len(predictions_df)} predictions")
        return predictions_df
    
    def _load_model(self):
        """Load a previously trained model."""
        try:
            if self.model_type == 'lstm':
                self.model = tf.keras.models.load_model(self.model_save_path.replace('.pkl', '.h5'))
                logger.info("Loaded LSTM model from file")
            else:
                with open(self.model_save_path, 'rb') as f:
                    self.model = joblib.load(f)
                logger.info("Loaded Prophet model from file")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def visualize_forecast(self, predictions, historical_periods=72):
        """
        Visualize forecasted energy demand alongside historical data.
        
        Parameters:
        -----------
        predictions : pandas.DataFrame
            DataFrame with predictions
        historical_periods : int
            Number of historical periods to show
        """
        # Load recent historical data
        historical_data = self._load_data().tail(historical_periods)
        
        # Create plot
        plt.figure(figsize=(15, 7))
        
        # Plot historical data
        plt.plot(historical_data.index, historical_data['consumption'], 
                 label='Historical Consumption', color='blue', alpha=0.7)
        
        # Plot predictions
        plt.plot(predictions['timestamp'], predictions['predicted_consumption'], 
                 label='Forecasted Consumption', color='red', linestyle='--')
        
        # Add confidence interval for Prophet
        if self.model_type == 'prophet' and 'prediction_lower' in predictions.columns:
            plt.fill_between(
                predictions['timestamp'],
                predictions['prediction_lower'],
                predictions['prediction_upper'],
                color='red', alpha=0.2, label='95% Confidence Interval'
            )
        
        # Add labels and legends
        plt.title('Energy Consumption Forecast', fontsize=16)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Energy Consumption (kWh)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save and show plot
        plt.savefig('./output/energy_forecast.png')
        logger.info("Forecast visualization saved to ./output/energy_forecast.png")
        plt.show()
        
# Example usage function
def run_forecast_example():
    """Example function to demonstrate energy forecasting."""
    # Create forecaster with LSTM model
    forecaster = EnergyForecaster(model_type='lstm', forecast_horizon=48, use_weather=True)
    
    # Train or load model
    try:
        forecaster._load_model()
        print("Loaded existing model")
    except:
        print("Training new model...")
        forecaster.train()
    
    # Generate predictions
    predictions = forecaster.predict()
    
    # Visualize results
    forecaster.visualize_forecast(predictions)
    
    return predictions

# Uncomment to run example
# if __name__ == "__main__":
#     run_forecast_example()

"""
SUMMARY:
--------
This module implements a flexible energy demand forecasting system using 
either LSTM neural networks or Facebook Prophet for time series prediction.
Key features:
- LSTM model for capturing complex temporal patterns
- Prophet model for interpretable forecasting with built-in seasonality
- Weather data integration to improve prediction accuracy
- Data preprocessing utilities for both model types
- Visualization tools for forecast evaluation
- Proper error handling and logging

TODO:
-----
- Add support for ensemble methods combining multiple forecasting approaches
- Implement feature importance analysis for better model interpretability
- Add API endpoint for integration with other systems
- Optimize hyperparameters automatically based on historical performance
"""
