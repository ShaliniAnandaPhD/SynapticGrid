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
                
                # Predict anomaly (-1 for anomalies, 1 for normal)
                prediction = self.anomaly_model.predict(scaled_features)
                is_anomaly = prediction[0] == -1
                
                # Adjust instability score based on anomaly detection
                if is_anomaly:
                    anomaly_score = self.anomaly_model.decision_function(scaled_features)[0]
                    # Convert to 0-1 scale (lower values are more anomalous)
                    anomaly_factor = max(0, min(1, 0.5 - anomaly_score))
                    # Boost instability score based on anomaly
                    instability_score = max(instability_score, 0.7 + (0.3 * anomaly_factor))
            except Exception as e:
                logger.error(f"Error in anomaly detection: {str(e)}")
        
        return instability_score, is_anomaly, component_alerts
    
    def _simulate_metrics(self):
        """
        Simulate grid metrics for testing and demonstration.
        In a real system, this would be replaced with actual sensor data.
        
        Returns:
        --------
        GridMetrics object with simulated values
        """
        # Normal conditions most of the time
        if np.random.random() < 0.8:
            # Normal operating conditions with small variations
            metrics = GridMetrics(
                voltage_deviation=np.random.uniform(0.01, 0.04),
                frequency_deviation=np.random.uniform(0.001, 0.015),
                phase_imbalance=np.random.uniform(0.02, 0.08),
                power_factor=np.random.uniform(0.92, 0.98),
                thd=np.random.uniform(0.02, 0.06),
                timestamp=datetime.now()
            )
        else:
            # Occasionally introduce abnormal conditions
            abnormal_type = np.random.choice(['voltage', 'frequency', 'phase', 'combined'])
            
            if abnormal_type == 'voltage':
                # Voltage sag or swell
                metrics = GridMetrics(
                    voltage_deviation=np.random.uniform(0.06, 0.15),  # Above threshold
                    frequency_deviation=np.random.uniform(0.001, 0.015),
                    phase_imbalance=np.random.uniform(0.02, 0.08),
                    power_factor=np.random.uniform(0.92, 0.98),
                    thd=np.random.uniform(0.02, 0.06),
                    timestamp=datetime.now()
                )
            elif abnormal_type == 'frequency':
                # Frequency deviation
                metrics = GridMetrics(
                    voltage_deviation=np.random.uniform(0.01, 0.04),
                    frequency_deviation=np.random.uniform(0.025, 0.05),  # Above threshold
                    phase_imbalance=np.random.uniform(0.02, 0.08),
                    power_factor=np.random.uniform(0.92, 0.98),
                    thd=np.random.uniform(0.02, 0.06),
                    timestamp=datetime.now()
                )
            elif abnormal_type == 'phase':
                # Phase imbalance
                metrics = GridMetrics(
                    voltage_deviation=np.random.uniform(0.01, 0.04),
                    frequency_deviation=np.random.uniform(0.001, 0.015),
                    phase_imbalance=np.random.uniform(0.12, 0.2),  # Above threshold
                    power_factor=np.random.uniform(0.85, 0.89),  # Slightly below threshold
                    thd=np.random.uniform(0.02, 0.06),
                    timestamp=datetime.now()
                )
            else:  # combined
                # Multiple issues (serious problem)
                metrics = GridMetrics(
                    voltage_deviation=np.random.uniform(0.06, 0.15),
                    frequency_deviation=np.random.uniform(0.025, 0.05),
                    phase_imbalance=np.random.uniform(0.12, 0.2),
                    power_factor=np.random.uniform(0.85, 0.89),
                    thd=np.random.uniform(0.09, 0.15),  # Above threshold
                    timestamp=datetime.now()
                )
                
        return metrics
    
    def _monitoring_loop(self, interval=5):
        """
        Background monitoring loop that periodically checks grid stability.
        
        Parameters:
        -----------
        interval : int
            Seconds between stability checks
        """
        while self.is_monitoring:
            try:
                # Get metrics (simulated or from actual sensors)
                metrics = self._simulate_metrics()
                
                # Analyze stability
                instability_score, is_anomaly, component_alerts = self.analyze_stability(metrics)
                
                # Update metrics object with score
                metrics.instability_score = instability_score
                
                # Store in historical data
                if self.historical_data is not None:
                    new_data = pd.DataFrame([{
                        'timestamp': metrics.timestamp,
                        'voltage_deviation': metrics.voltage_deviation,
                        'frequency_deviation': metrics.frequency_deviation,
                        'phase_imbalance': metrics.phase_imbalance,
                        'power_factor': metrics.power_factor,
                        'thd': metrics.thd,
                        'instability_score': instability_score
                    }])
                    
                    self.historical_data = pd.concat([self.historical_data, new_data], ignore_index=True)
                
                # Add to current metrics cache (keep last 100)
                self.current_metrics.append(metrics)
                if len(self.current_metrics) > 100:
                    self.current_metrics.pop(0)
                
                # Check for alerts
                if instability_score > self.alert_threshold:
                    self._trigger_alert(metrics, is_anomaly, component_alerts)
                
                # Periodically save historical data
                if np.random.random() < 0.05:  # ~5% chance each loop
                    self.save_historical_data()
                    
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                
            # Wait for next check
            time.sleep(interval)
    
    def start_monitoring(self, interval=5):
        """
        Start the background monitoring process.
        
        Parameters:
        -----------
        interval : int
            Seconds between monitoring checks
        """
        if self.is_monitoring:
            logger.warning("Monitoring already running")
            return
            
        logger.info(f"Starting grid stability monitoring (interval={interval}s)")
        self.is_monitoring = True
        
        # Start monitoring in a background thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop the background monitoring process."""
        if not self.is_monitoring:
            logger.warning("Monitoring not running")
            return
            
        logger.info("Stopping grid stability monitoring")
        self.is_monitoring = False
        
        # Wait for thread to finish
        if self.monitoring_thread is not None:
            self.monitoring_thread.join(timeout=5)
            self.monitoring_thread = None
    
    def _trigger_alert(self, metrics, is_anomaly, component_alerts):
        """
        Trigger an alert when grid instability is detected.
        
        Parameters:
        -----------
        metrics : GridMetrics
            Current grid metrics
        is_anomaly : bool
            Whether anomaly detection flagged this as unusual
        component_alerts : dict
            Per-component alert status
        """
        alert_components = [c for c, alerted in component_alerts.items() if alerted]
        
        # Log the alert
        logger.warning(f"GRID STABILITY ALERT: Instability score {metrics.instability_score:.2f}")
        logger.warning(f"  Anomaly detected: {is_anomaly}")
        logger.warning(f"  Components affected: {', '.join(alert_components)}")
        logger.warning(f"  Metrics: V-dev={metrics.voltage_deviation:.3f}, "
                      f"F-dev={metrics.frequency_deviation:.3f}, "
                      f"Phase-imb={metrics.phase_imbalance:.3f}, "
                      f"PF={metrics.power_factor:.3f}, "
                      f"THD={metrics.thd:.3f}")
        
        # In a real system, this would:
        # 1. Send notifications (email, SMS, push)
        # 2. Trigger automated responses
        # 3. Log to a centralized alert system
        # 4. Potentially initiate emergency procedures for severe issues
    
    def get_current_stability(self):
        """
        Get the current grid stability status.
        
        Returns:
        --------
        dict with current stability metrics and alert status
        """
        if not self.current_metrics:
            return {'status': 'No data available', 'instability_score': 0}
            
        # Use the most recent metrics
        latest_metrics = self.current_metrics[-1]
        
        # Analyze stability
        instability_score, is_anomaly, component_alerts = self.analyze_stability(latest_metrics)
        
        # Determine status message
        if instability_score > self.alert_threshold:
            status = "CRITICAL" if instability_score > 0.9 else "WARNING"
        else:
            status = "Stable"
            
        # Count alerts by component
        alert_components = [c for c, alerted in component_alerts.items() if alerted]
        
        return {
            'status': status,
            'timestamp': latest_metrics.timestamp,
            'instability_score': instability_score,
            'is_anomaly': is_anomaly,
            'alert_components': alert_components,
            'metrics': {
                'voltage_deviation': latest_metrics.voltage_deviation,
                'frequency_deviation': latest_metrics.frequency_deviation,
                'phase_imbalance': latest_metrics.phase_imbalance,
                'power_factor': latest_metrics.power_factor,
                'thd': latest_metrics.thd
            }
        }
    
    def save_historical_data(self):
        """Save the historical metrics data to CSV."""
        if self.historical_data is not None and not self.historical_data.empty:
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
                
                # Save to CSV
                self.historical_data.to_csv(self.data_path, index=False)
                logger.info(f"Saved {len(self.historical_data)} records to {self.data_path}")
            except Exception as e:
                logger.error(f"Error saving historical data: {str(e)}")
    
    def visualize_stability_trend(self, hours=24):
        """
        Create a visualization of recent grid stability trends.
        
        Parameters:
        -----------
        hours : int
            Number of hours to include in the visualization
        """
        if self.historical_data is None or self.historical_data.empty:
            logger.warning("No historical data available for visualization")
            return
            
        # Filter to recent data
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        recent_data = self.historical_data[
            (self.historical_data['timestamp'] >= start_time) & 
            (self.historical_data['timestamp'] <= end_time)
        ]
        
        if recent_data.empty:
            logger.warning(f"No data available for the last {hours} hours")
            return
            
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [1, 2]})
        
        # Plot instability score
        ax1.plot(recent_data['timestamp'], recent_data['instability_score'], 
                 color='red', linewidth=2)
        ax1.axhline(y=self.alert_threshold, color='orange', linestyle='--', 
                   label=f'Alert Threshold ({self.alert_threshold})')
        ax1.set_ylabel('Instability Score')
        ax1.set_title('Grid Stability Trends')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot individual metrics
        metrics_to_plot = ['voltage_deviation', 'frequency_deviation', 
                          'phase_imbalance', 'power_factor', 'thd']
        colors = ['blue', 'green', 'purple', 'brown', 'teal']
        
        for metric, color in zip(metrics_to_plot, colors):
            ax2.plot(recent_data['timestamp'], recent_data[metric], 
                    label=metric.replace('_', ' ').title(), color=color)
                    
            # Add threshold lines
            if metric in self.thresholds:
                threshold = self.thresholds[metric]
                ax2.axhline(y=threshold, color=color, linestyle=':', alpha=0.7)
        
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Metric Values')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis
        fig.autofmt_xdate()
        
        # Save and show plot
        os.makedirs('./output', exist_ok=True)
        plt.savefig('./output/grid_stability_trend.png')
        logger.info("Grid stability visualization saved to ./output/grid_stability_trend.png")
        plt.tight_layout()
        plt.show()
    
    def set_alert_threshold(self, threshold):
        """
        Update the alert threshold.
        
        Parameters:
        -----------
        threshold : float
            New threshold for triggering stability alerts (0-1)
        """
        if 0 <= threshold <= 1:
            self.alert_threshold = threshold
            logger.info(f"Alert threshold updated to {threshold}")
        else:
            logger.error("Threshold must be between 0 and 1")
            
    def set_component_threshold(self, component, value):
        """
        Update the threshold for a specific component.
        
        Parameters:
        -----------
        component : str
            Component name (e.g., 'voltage_deviation')
        value : float
            New threshold value
        """
        if component in self.thresholds:
            self.thresholds[component] = value
            logger.info(f"Updated {component} threshold to {value}")
        else:
            logger.error(f"Unknown component: {component}")

# Example usage function
def run_stability_monitor_example():
    """Example function to demonstrate grid stability monitoring."""
    # Create monitor
    monitor = GridStabilityMonitor()
    
    # Train anomaly model if needed
    try:
        monitor.train_anomaly_model()
    except:
        print("Not enough data to train model - will use threshold-based detection only")
    
    # Start monitoring
    print("Starting grid stability monitoring... (will run for 60 seconds)")
    monitor.start_monitoring(interval=2)
    
    # Run for some time
    try:
        for _ in range(30):
            # Get and print current status every 2 seconds
            time.sleep(2)
            status = monitor.get_current_stability()
            print(f"\rGrid Status: {status['status']} | "
                  f"Score: {status['instability_score']:.2f}", end="")
    except KeyboardInterrupt:
        print("\nMonitoring interrupted")
    finally:
        # Stop monitoring
        monitor.stop_monitoring()
        
    # Save historical data
    monitor.save_historical_data()
    
    # Visualize results
    print("\nGenerating visualization...")
    monitor.visualize_stability_trend(hours=1)
    
    return monitor

# Uncomment to run example
# if __name__ == "__main__":
#     run_stability_monitor_example()

"""
SUMMARY:
--------
This module implements a comprehensive grid stability monitoring system
capable of detecting potential failures in decentralized energy grids.
Key features:
- Real-time monitoring of critical grid stability metrics
- Anomaly detection using Isolation Forest algorithm
- Component-level threshold monitoring
- Weighted instability scoring system
- Alerting for potential grid stability issues
- Visualization of stability trends over time
- Simulated metrics for testing and demonstration
- Persistent storage of historical data for analysis

TODO:
-----
- Implement connection to actual grid sensors/meters
- Add predictive capabilities to forecast potential failures
- Integrate with notification systems (email, SMS, app alerts)
- Create web dashboard for real-time monitoring
- Implement automated response actions for critical stability issues
- Add support for geographical distribution of stability metrics
"""
