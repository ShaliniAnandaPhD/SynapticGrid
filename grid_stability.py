"""
Grid Stability Monitor
---------------------
Real-time monitoring system to analyze power fluctuations and detect
potential failures in decentralized energy grids. Uses statistical
methods and anomaly detection to identify unstable grid conditions.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import time
import threading
import logging
import warnings
import os
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("grid_stability.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("GridStabilityMonitor")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

@dataclass
class GridMetrics:
    """Data class to store grid stability metrics."""
    voltage_deviation: float = 0.0
    frequency_deviation: float = 0.0
    phase_imbalance: float = 0.0
    power_factor: float = 0.0
    thd: float = 0.0  # Total Harmonic Distortion
    instability_score: float = 0.0
    timestamp: datetime = None

class GridStabilityMonitor:
    """
    Monitor and analyze power grid stability metrics to detect
    potential failures and instabilities in the grid.
    """
    def __init__(self, data_path=None, model_path=None, alert_threshold=0.8):
        """
        Initialize Grid Stability Monitor.
        
        Parameters:
        -----------
        data_path : str or None
            Path to historical grid metrics data
        model_path : str or None
            Path to save/load anomaly detection model
        alert_threshold : float
            Threshold for triggering stability alerts (0-1)
        """
        self.data_path = data_path or "./data/grid_metrics.csv"
        self.model_path = model_path or "./models/grid_stability_model.pkl"
        self.alert_threshold = alert_threshold
        
        # Component stability thresholds (default values)
        self.thresholds = {
            'voltage_deviation': 0.05,  # 5% from nominal
            'frequency_deviation': 0.02,  # 2% from nominal (50/60Hz)
            'phase_imbalance': 0.1,  # 10% imbalance
            'power_factor': 0.9,  # Below 0.9 is concerning
            'thd': 0.08  # Total Harmonic Distortion > 8% is concerning
        }
        
        # Initialize state variables
        self.historical_data = None
        self.anomaly_model = None
        self.scaler = StandardScaler()
        self.is_monitoring = False
        self.monitoring_thread = None
        self.current_metrics = []  # Store recent metrics for analysis
        
        # Try to load existing model and historical data
        self._load_resources()
        
        logger.info("Grid Stability Monitor initialized")
        
    def _load_resources(self):
        """Load historical data and anomaly detection model if available."""
        # Load historical data if available
        try:
            if os.path.exists(self.data_path):
                self.historical_data = pd.read_csv(self.data_path, parse_dates=['timestamp'])
                logger.info(f"Loaded historical data: {len(self.historical_data)} records")
        except Exception as e:
            logger.warning(f"Could not load historical data: {str(e)}")
            self.historical_data = pd.DataFrame(columns=[
                'timestamp', 'voltage_deviation', 'frequency_deviation', 
                'phase_imbalance', 'power_factor', 'thd', 'instability_score'
            ])
        
        # Load anomaly detection model if available
        try:
            if os.path.exists(self.model_path):
                self.anomaly_model = joblib.load(self.model_path)
                logger.info("Loaded anomaly detection model")
        except Exception as e:
            logger.warning(f"Could not load anomaly model: {str(e)}")
    
    def train_anomaly_model(self, retrain=False):
        """
        Train anomaly detection model on historical data.
        
        Parameters:
        -----------
        retrain : bool
            Whether to retrain the model even if one already exists
        """
        # Check if we need to train a new model
        if self.anomaly_model is not None and not retrain:
            logger.info("Anomaly model already exists. Use retrain=True to force retraining.")
            return
            
        # Check if we have enough data
        if self.historical_data is None or len(self.historical_data) < 100:
            logger.warning("Not enough data to train anomaly model. Need at least 100 records.")
            return
            
        logger.info("Training anomaly detection model...")
        
        # Prepare features
        features = self.historical_data[['voltage_deviation', 'frequency_deviation', 
                                         'phase_imbalance', 'power_factor', 'thd']]
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Train Isolation Forest model for anomaly detection
        model = IsolationForest(
            n_estimators=100, 
            contamination=0.05,  # Expect approximately 5% anomalies
            random_state=42,
            n_jobs=-1  # Use all available cores
        )
        
        model.fit(scaled_features)
        self.anomaly_model = model
        
        # Save the model
        joblib.dump(model, self.model_path)
        joblib.dump(self.scaler, self.model_path.replace('.pkl', '_scaler.pkl'))
        
        logger.info("Anomaly detection model trained and saved")
    
    def analyze_stability(self, metrics):
        """
        Analyze grid metrics to determine stability and detect anomalies.
        
        Parameters:
        -----------
        metrics : GridMetrics or dict
            The grid metrics to analyze
            
        Returns:
        --------
        instability_score : float
            Overall instability score (0-1)
        is_anomaly : bool
            Whether the current state is anomalous
        component_alerts : dict
            Per-component alert status
        """
        # Convert to dict if GridMetrics object
        if isinstance(metrics, GridMetrics):
            metrics_dict = {
                'voltage_deviation': metrics.voltage_deviation,
                'frequency_deviation': metrics.frequency_deviation,
                'phase_imbalance': metrics.phase_imbalance,
                'power_factor': metrics.power_factor,
                'thd': metrics.thd
            }
        else:
            metrics_dict = metrics
            
        # Check component-level alerts
        component_alerts = {}
        for component, value in metrics_dict.items():
            if component in self.thresholds:
                # Different logic for power_factor (higher is better)
                if component == 'power_factor':
                    component_alerts[component] = value < self.thresholds[component]
                else:
                    component_alerts[component] = value > self.thresholds[component]
                    
        # Calculate instability score (weighted average of normalized metrics)
        weights = {
            'voltage_deviation': 0.3,
            'frequency_deviation': 0.3,
            'phase_imbalance': 0.15,
            'power_factor': 0.15,  # Invert since higher is better
            'thd': 0.1
        }
        
        # Normalize metrics between 0-1 based on thresholds
        normalized_metrics = {}
        for component, value in metrics_dict.items():
            if component in self.thresholds:
                if component == 'power_factor':
                    # Power factor: 1.0 is ideal, below threshold is bad
                    threshold = self.thresholds[component]
                    normalized_metrics[component] = max(0, 1 - (value / threshold))
                else:
                    # Other metrics: 0 is ideal, above threshold is bad
                    threshold = self.thresholds[component]
                    normalized_metrics[component] = min(1, value / (threshold * 2))
                    
        # Calculate weighted instability score
        instability_score = sum(normalized_metrics[c] * weights[c] for c in weights.keys())
        
        # Run anomaly detection if model exists
        is_anomaly = False
        if self.anomaly_model is not None:
            try:
                # Prepare features
                features = np.array([[
                    metrics_dict['voltage_deviation'],
                    metrics_dict['frequency_deviation'],
                    metrics_dict['phase_imbalance'],
                    metrics_dict['power_factor'],
                    metrics_dict['thd']
                ]])
                
                # Scale features
                scaled_features = self.scaler.transform(features)
