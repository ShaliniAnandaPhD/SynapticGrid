"""
A reinforcement learning-based system for dynamically redistributing 
energy loads across different nodes in a smart grid to optimize 
efficiency, cost, and stability.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import logging
import threading
import json
import os
from datetime import datetime, timedelta
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import joblib
import gymnasium as gym
from gymnasium import spaces
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("load_balancer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PowerLoadBalancer")

class GridNode:
    """
    Represents a node in the power grid (generator, consumer, storage).
    """
    def __init__(self, node_id, node_type, capacity, efficiency=0.95, cost_factor=1.0):
        """
        Initialize a grid node.
        
        Parameters:
        -----------
        node_id : str
            Unique identifier for the node
        node_type : str
            Type of node ('generator', 'consumer', 'storage')
        capacity : float
            Maximum power capacity in kW
        efficiency : float
            Efficiency of the node (0-1)
        cost_factor : float
            Relative cost factor for using this node
        """
        self.node_id = node_id
        self.node_type = node_type
        self.capacity = capacity
        self.efficiency = efficiency
        self.cost_factor = cost_factor
        self.current_load = 0.0
        self.history = []  # Track load history
        
        # Storage-specific attributes
        if node_type == 'storage':
            self.stored_energy = 0.0
            self.max_storage = capacity * 4  # 4 hours at max capacity
            self.charge_rate = capacity * 0.8  # Can charge at 80% of max
            self.discharge_rate = capacity  # Can discharge at max capacity
        
        logger.info(f"Created {node_type} node {node_id} with capacity {capacity}kW")
    
    def set_load(self, load_value):
        """
        Set the current load for this node.
        
        Parameters:
        -----------
        load_value : float
            Power load in kW
        
        Returns:
        --------
        float : Actual load set (may be clipped by capacity)
        """
        # For generators and consumers, clip to capacity
        if self.node_type in ['generator', 'consumer']:
            actual_load = max(0, min(load_value, self.capacity))
            self.current_load = actual_load
            self.history.append((datetime.now(), actual_load))
            return actual_load
            
        # For storage nodes, handle charging/discharging
        elif self.node_type == 'storage':
            # Positive load = charging, negative load = discharging
            if load_value > 0:  # Charging
                # Calculate how much can be charged
                max_charge = min(
                    self.charge_rate,  # Limited by charge rate
                    self.max_storage - self.stored_energy  # Limited by remaining capacity
                )
                actual_load = min(load_value, max_charge)
                
                # Calculate energy added after efficiency loss
                energy_added = actual_load * self.efficiency
                self.stored_energy += energy_added
            else:  # Discharging
                # Calculate how much can be discharged
                max_discharge = min(
                    self.discharge_rate,  # Limited by discharge rate
                    self.stored_energy  # Limited by available energy
                )
                actual_load = max(load_value, -max_discharge)
                
                # Update stored energy
                self.stored_energy += actual_load  # Actual_load is negative here
            
            self.current_load = actual_load
            self.history.append((datetime.now(), actual_load))
            return actual_load
    
    def get_state(self):
        """Get the current state of the node."""
        state = {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "capacity": self.capacity,
            "current_load": self.current_load,
            "utilization": self.current_load / self.capacity if self.capacity > 0 else 0
        }
        
        # Add storage-specific state
        if self.node_type == 'storage':
            state.update({
                "stored_energy": self.stored_energy,
                "max_storage": self.max_storage,
                "storage_level": self.stored_energy / self.max_storage
            })
            
        return state

class GridEnvironment(gym.Env):
    """
    Reinforcement Learning environment for power load balancing.
    """
    def __init__(self, nodes, time_step=15):
        """
        Initialize the grid environment.
        
        Parameters:
        -----------
        nodes : list
            List of GridNode objects
        time_step : int
            Time step in minutes for each action
        """
        super(GridEnvironment, self).__init__()
        self.nodes = nodes
        self.time_step = time_step
        self.current_time = datetime.now()
        self.episode_step = 0
        self.max_episode_steps = 24 * 60 // time_step  # 24 hours
        
        # Track total grid metrics
        self.total_demand = 0.0
        self.total_production = 0.0
        self.total_storage = 0.0
        self.grid_balance = 0.0
        self.cost_history = []
        self.balance_history = []
        
        # Define action and observation spaces
        # Actions: Load adjustment for each node (-1 to 1 normalized)
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(len(nodes),),
            dtype=np.float32
        )
        
        # State space: For each node [current_load, capacity_utilization]
        #              Plus global metrics [total_demand, total_production, grid_balance]
        state_vars_per_node = 2
        if any(node.node_type == 'storage' for node in nodes):
            state_vars_per_node = 3  # Add storage level
            
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(nodes) * state_vars_per_node + 3,),
            dtype=np.float32
        )
        
        # For tracking metrics
        self.metrics = {
            'rewards': [],
            'grid_balance': [],
            'costs': [],
            'load_factors': []
        }
        
        # Weather and time factors for simulation
        self.weather_conditions = "sunny"  # Default
        self.temperature = 22.0  # Celsius
        
        logger.info(f"Initialized GridEnvironment with {len(nodes)} nodes and {time_step} minute time steps")
    
    def _get_state(self):
        """
        Get the current state of the environment.
        
        Returns:
        --------
        numpy.ndarray : The state vector
        """
        # Get individual node states
        node_states = []
        for node in self.nodes:
            state = node.get_state()
            node_states.append(state["current_load"])
            node_states.append(state["utilization"])
            
            # Add storage level if applicable
            if node.node_type == 'storage' and "storage_level" in state:
                node_states.append(state["storage_level"])
        
        # Add global metrics
        node_states.extend([self.total_demand, self.total_production, self.grid_balance])
        
        return np.array(node_states, dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.
        
        Returns:
        --------
        state : Initial state
        info : Additional information
        """
        super().reset(seed=seed)
        
        # Reset time and step counter
        self.current_time = datetime.now()
        self.episode_step = 0
        
        # Reset nodes to initial state
        for node in self.nodes:
            if node.node_type == 'generator':
                # Set generators to random initial production
                initial_load = np.random.uniform(0.2, 0.5) * node.capacity
                node.set_load(initial_load)
            elif node.node_type == 'consumer':
                # Set consumers to random initial demand
                initial_load = np.random.uniform(0.1, 0.4) * node.capacity
                node.set_load(initial_load)
            elif node.node_type == 'storage':
                # Set storage to random initial charge level
                node.stored_energy = np.random.uniform(0.1, 0.5) * node.max_storage
                node.current_load = 0.0
        
        # Calculate grid metrics
        self._update_grid_metrics()
        
        # Random weather conditions
        weather_types = ["sunny", "cloudy", "rainy", "windy"]
        self.weather_conditions = random.choice(weather_types)
        self.temperature = np.random.normal(22, 5)  # Random temperature around 22°C
        
        # Get initial state
        state = self._get_state()
        
        return state, {}  # State and info dict (empty for now)
    
    def _update_grid_metrics(self):
        """Update overall grid metrics based on current node states."""
        production = sum(node.current_load for node in self.nodes if node.node_type == 'generator')
        demand = sum(node.current_load for node in self.nodes if node.node_type == 'consumer')
        
        # For storage: positive load = charging (adds to demand), negative = discharging (adds to production)
        storage_load = sum(node.current_load for node in self.nodes if node.node_type == 'storage')
        
        # Update metrics
        self.total_production = production
        if storage_load < 0:
            self.total_production -= storage_load  # Discharging adds to production
            
        self.total_demand = demand
        if storage_load > 0:
            self.total_demand += storage_load  # Charging adds to demand
            
        # Calculate grid balance (positive = surplus, negative = deficit)
        self.grid_balance = self.total_production - self.total_demand
        
        # Calculate total storage
        self.total_storage = sum(node.stored_energy for node in self.nodes if node.node_type == 'storage')
    
    def _calculate_reward(self):
        """
        Calculate the reward for the current state.
        
        Returns:
        --------
        float : Reward value
        """
        # Components of the reward function
        grid_balance_reward = 0.0
        cost_reward = 0.0
        utilization_reward = 0.0
        stability_reward = 0.0
        
        # 1. Grid Balance: Penalize imbalance (squared to penalize larger imbalances more)
        # Normalize by total demand to make it relative
        if self.total_demand > 0:
            balance_factor = self.grid_balance / self.total_demand
            grid_balance_reward = -10.0 * (balance_factor ** 2)
        else:
            grid_balance_reward = 0  # No demand, no imbalance penalty
        
        # 2. Cost: Reward lower cost operation
        # Calculate cost based on node types and their efficiency
        cost = 0.0
        for node in self.nodes:
            if node.node_type == 'generator':
                # Generation cost
                cost += node.current_load * node.cost_factor
            elif node.node_type == 'storage' and node.current_load < 0:
                # Cost of discharging storage (usually low)
                cost += abs(node.current_load) * 0.2 * node.cost_factor
        
        # Normalize cost by total production
        if self.total_production > 0:
            cost_factor = cost / self.total_production
            cost_reward = -5.0 * cost_factor
        else:
            cost_reward = 0
        
        # 3. Utilization: Reward efficient utilization of resources
        # For generators, we want high utilization if they're cheap
        # For storage, we want balanced utilization (not too full or empty)
        utilization_sum = 0.0
        count = 0
        for node in self.nodes:
            if node.node_type == 'generator':
                # Reward high utilization for cheap generators, low for expensive ones
                if node.cost_factor < 0.8:  # Cheap generation
                    utilization_sum += node.current_load / node.capacity
                else:  # Expensive generation
                    utilization_sum += 1.0 - (node.current_load / node.capacity)
                count += 1
            elif node.node_type == 'storage':
                # Reward storage levels that aren't extreme (not too full or empty)
                storage_ratio = node.stored_energy / node.max_storage
                # Gaussian function with peak at 0.5 (mid-level storage)
                storage_reward = np.exp(-((storage_ratio - 0.5) ** 2) / 0.1)
                utilization_sum += storage_reward
                count += 1
        
        if count > 0:
            utilization_reward = 3.0 * (utilization_sum / count)
        
        # 4. Stability: Reward minimal changes in load (to avoid oscillation)
        if len(self.balance_history) > 1:
            last_balance = self.balance_history[-1]
            stability_factor = abs(self.grid_balance - last_balance) / max(1.0, self.total_demand)
            stability_reward = -2.0 * stability_factor
        
        # Combine rewards
        total_reward = grid_balance_reward + cost_reward + utilization_reward + stability_reward
        
        # Add rewards to metrics for tracking
        self.metrics['rewards'].append(total_reward)
        self.metrics['grid_balance'].append(self.grid_balance)
        self.metrics['costs'].append(cost)
        
        # Log the rewards for debugging
        if self.episode_step % 10 == 0:  # Log every 10 steps
            logger.debug(f"Rewards - Balance: {grid_balance_reward:.2f}, Cost: {cost_reward:.2f}, "
                         f"Utilization: {utilization_reward:.2f}, Stability: {stability_reward:.2f}, "
                         f"Total: {total_reward:.2f}")
        
        return total_reward
    
    def step(self, action):
        """
        Take a step in the environment by applying the action.
        
        Parameters:
        -----------
        action : numpy.ndarray
            Action vector with load adjustments for each node
        
        Returns:
        --------
        state : New state after action
        reward : Reward for the action
        terminated : Whether episode is done
        truncated : Whether episode is truncated
        info : Additional information
        """
        # Increment step counter
        self.episode_step += 1
        
        # Store current balance for stability calculation
        self.balance_history.append(self.grid_balance)
        
        # Save current costs
        current_cost = sum(node.current_load * node.cost_factor for node in self.nodes 
                          if node.node_type == 'generator')
        self.cost_history.append(current_cost)
        
        # Update time
        self.current_time += timedelta(minutes=self.time_step)
        
        # Apply action to each node
        for i, node in enumerate(self.nodes):
            # Convert normalized action to load adjustment
            if i < len(action):  # Ensure we have an action for this node
                if node.node_type == 'generator':
                    # For generators, action adjusts output
                    action_value = action[i]
                    # Scale action to a reasonable load change (% of capacity)
                    load_change = action_value * 0.1 * node.capacity  # Max 10% change per step
                    new_load = node.current_load + load_change
                    node.set_load(new_load)
                    
                elif node.node_type == 'storage':
                    # For storage, action determines charging/discharging
                    action_value = action[i]
                    # Scale action to a charge/discharge rate (% of capacity)
                    if action_value > 0:  # Charging
                        charge_rate = action_value * node.charge_rate
                        node.set_load(charge_rate)
                    else:  # Discharging
                        discharge_rate = action_value * node.discharge_rate
                        node.set_load(discharge_rate)
                        
                # Consumers are not controlled by the agent
        
        # Update consumer load based on time of day and weather
        self._update_consumer_demand()
        
        # Update renewable generation based on weather
        self._update_renewable_generation()
        
        # Recalculate grid metrics
        self._update_grid_metrics()
        
        # Get new state, reward, and done flag
        state = self._get_state()
        reward = self._calculate_reward()
        terminated = self.episode_step >= self.max_episode_steps
        truncated = False
        
        # Additional info
        info = {
            'grid_balance': self.grid_balance,
            'total_demand': self.total_demand,
            'total_production': self.total_production,
            'total_storage': self.total_storage,
            'time': self.current_time.strftime('%H:%M')
        }
        
        return state, reward, terminated, truncated, info
    
    def _update_consumer_demand(self):
        """Update consumer demand based on time of day, temperature, etc."""
        # Get hour of day (0-23)
        hour = self.current_time.hour
        
        # Base demand patterns by hour (24-hour profile)
        # Simulates typical daily load curve with morning and evening peaks
        hourly_factors = [
            0.6, 0.5, 0.4, 0.4, 0.5, 0.7,  # 00:00 - 05:59
            0.9, 1.1, 1.2, 1.1, 1.0, 0.9,  # 06:00 - 11:59
            0.9, 1.0, 1.0, 1.0, 1.1, 1.2,  # 12:00 - 17:59
            1.3, 1.2, 1.1, 0.9, 0.8, 0.7   # 18:00 - 23:59
        ]
        
        base_factor = hourly_factors[hour]
        
        # Adjust for temperature (higher demand during temperature extremes)
        temp_factor = 1.0
        if self.temperature > 28:  # Hot day
            temp_factor = 1.0 + (self.temperature - 28) * 0.05  # 5% increase per degree above 28
        elif self.temperature < 15:  # Cold day
            temp_factor = 1.0 + (15 - self.temperature) * 0.03  # 3% increase per degree below 15
        
        # Adjust for weekday/weekend (simplified)
        day_factor = 1.0
        if self.current_time.weekday() >= 5:  # Weekend
            day_factor = 0.9
        
        # Apply small random variation
        random_factor = np.random.uniform(0.95, 1.05)
        
        # Update each consumer node
        for node in self.nodes:
            if node.node_type == 'consumer':
                # Calculate target load
                target_load = node.capacity * base_factor * temp_factor * day_factor * random_factor
                node.set_load(target_load)
    
    def _update_renewable_generation(self):
        """Update renewable generation based on weather, time of day, etc."""
        # Get hour of day (0-23)
        hour = self.current_time.hour
        
        # Solar generation factors by hour
        solar_hourly_factors = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.05,  # 00:00 - 05:59
            0.2, 0.4, 0.6, 0.75, 0.85, 0.9,  # 06:00 - 11:59
            0.95, 0.9, 0.85, 0.75, 0.6, 0.4,  # 12:00 - 17:59
            0.1, 0.0, 0.0, 0.0, 0.0, 0.0      # 18:00 - 23:59
        ]
        
        # Wind generation factors (simpler, less time dependent)
        wind_hourly_factors = [
            0.7, 0.7, 0.8, 0.8, 0.7, 0.6,  # 00:00 - 05:59
            0.5, 0.5, 0.6, 0.7, 0.8, 0.7,  # 06:00 - 11:59
            0.6, 0.5, 0.6, 0.7, 0.8, 0.9,  # 12:00 - 17:59
            0.8, 0.7, 0.7, 0.6, 0.7, 0.7   # 18:00 - 23:59
        ]
        
        # Weather impact on renewables
        solar_weather_factor = {
            "sunny": 1.0,
            "cloudy": 0.5,
            "rainy": 0.2,
            "windy": 0.7
        }
        
        wind_weather_factor = {
            "sunny": 0.6,
            "cloudy": 0.8,
            "rainy": 0.7,
            "windy": 1.2
        }
        
        # Apply random variation
        solar_random = np.random.uniform(0.9, 1.1)
        wind_random = np.random.uniform(0.85, 1.15)
        
        # Update each generator node
        for node in self.nodes:
            if node.node_type == 'generator':
                # Determine generator type from ID (simplified)
                if 'solar' in node.node_id.lower():
                    # Calculate solar generation
                    solar_factor = (solar_hourly_factors[hour] * 
                                   solar_weather_factor.get(self.weather_conditions, 0.7) * 
                                   solar_random)
                    target_load = node.capacity * solar_factor
                    node.set_load(target_load)
                    
                elif 'wind' in node.node_id.lower():
                    # Calculate wind generation
                    wind_factor = (wind_hourly_factors[hour] * 
                                  wind_weather_factor.get(self.weather_conditions, 0.8) * 
                                  wind_random)
                    target_load = node.capacity * wind_factor
                    node.set_load(target_load)
                    
                # Other generators (conventional) remain unchanged

class DQNAgent:
    """
    Deep Q-Network agent for learning optimal load balancing strategies.
    """
    def __init__(self, state_size, action_size, hidden_size=128, learning_rate=0.001,
                 gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 memory_size=10000, batch_size=64, model_path=None):
        """
        Initialize DQN Agent.
        
        Parameters:
        -----------
        state_size : int
            Size of state space
        action_size : int
            Size of action space
        hidden_size : int
            Size of hidden layers
        learning_rate : float
            Learning rate for optimizer
        gamma : float
            Discount factor for future rewards
        epsilon : float
            Exploration rate
        epsilon_decay : float
            Decay rate for epsilon
        epsilon_min : float
            Minimum epsilon value
        memory_size : int
            Size of replay memory
        batch_size : int
            Batch size for training
        model_path : str or None
            Path to save/load model
        """
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.model_path = model_path or "./models/load_balancer_dqn.h5"
        
        # Initialize replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Initialize model
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        logger.info(f"Initialized DQN Agent with state_size={state_size}, action_size={action_size}")
    
    def _build_model(self):
        """Build a neural network model for DQN."""
        model = Sequential()
        model.add(Dense(self.hidden_size, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.hidden_size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        """Update target model to match main model."""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """
        Choose action based on epsilon-greedy policy.
        
        Parameters:
        -----------
        state : numpy.ndarray
            Current state vector
        
        Returns:
        --------
        numpy.ndarray : Action vector
        """
        if np.random.rand() <= self.epsilon:
            # Explore: take random action
            return np.random.uniform(-1, 1, self.action_size)
        
        # Exploit: use model prediction
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return act_values[0]
    
    def replay(self, batch_size=None):
        """
        Train the model using experience replay.
        
        Parameters:
        -----------
        batch_size : int or None
            Batch size for training, uses self.batch_size if None
        
        Returns:
        --------
        float : Loss value from training
        """
        batch_size = batch_size or self.batch_size
        
        # Check if we have enough samples
        if len(self.memory) < batch_size:
            return 0
        
        # Sample minibatch from memory
        minibatch = random.sample(self.memory, batch_size)
        
        # Prepare training data
        states = np.zeros((batch_size, self.state_size))
        targets = np.zeros((batch_size, self.action_size))
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                # Use target model for prediction to stabilize training
                target = reward + self.gamma * np.max(
                    self.target_model.predict(next_state.reshape(1, -1), verbose=0)[0]
                )
            
            # Get model predictions for current state
            target_f = self.model.predict(state.reshape(1, -1), verbose=0)
            # Update the target for the action that was taken
            target_f[0][action] = target
            
            # Store for batch training
            states[i] = state
            targets[i] = target_f[0]
        
        # Train the model
        history = self.model.fit(
            states, targets, epochs=1, verbose=0, batch_size=batch_size
        )
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return history.history['loss'][0] if history.history['loss'] else 0
    
    def load(self, path=None):
        """Load model weights from file."""
        path = path or self.model_path
        if os.path.exists(path):
            self.model.load_weights(path)
            self.target_model.load_weights(path)
            logger.info(f"Loaded model weights from {path}")
            return True
        logger.warning(f"Could not load model from {path}")
        return False
    
    def save(self, path=None):
        """Save model weights to file."""
        path = path or self.model_path
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        self.model.save_weights(path)
        logger.info(f"Saved model weights to {path}")

class PowerLoadBalancer:
    """
    Main class for real-time power load balancing in a smart grid.
    """
    def __init__(self, config_path=None, model_path=None):
        """
        Initialize the Power Load Balancer.
        
        Parameters:
        -----------
        config_path : str or None
            Path to configuration file
        model_path : str or None
            Path to save/load RL model
        """
        self.config_path = config_path or "./config/load_balancer_config.json"
        self.model_path = model_path or "./models/load_balancer_model.h5"
        
        # Load configuration
        self.config = self._load_config()
        
        # Create grid nodes
        self.nodes = self._create_nodes()
        
        # Create environment
        self.env = GridEnvironment(self.nodes, time_step=self.config.get('time_step', 15))
        
        # Create RL agent
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.shape[0]
        self.agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            hidden_size=self.config.get('hidden_size', 128),
            learning_rate=self.config.get('learning_rate', 0.001),
            gamma=self.config.get('gamma', 0.99),
            epsilon=self.config.get('epsilon', 1.0),
            epsilon_decay=self.config.get('epsilon_decay', 0.995),
            epsilon_min=self.config.get('epsilon_min', 0.01),
            memory_size=self.config.get('memory_size', 10000),
            batch_size=self.config.get('batch_size', 64),
            model_path=self.model_path
        )
        
        # Try to load existing model
        self.agent.load()
        
        # Status tracking
        self.is_running = False
        self.total_episodes = 0
        self.best_reward = -np.inf
        
        logger.info("Power Load Balancer initialized")
    
    def _load_config(self):
        """Load configuration from file or use defaults."""
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
                return config
            except Exception as e:
                logger.error(f"Error loading configuration: {str(e)}")
        
        # Default configuration
        default_config = {
            'time_step': 15,  # minutes
            'hidden_size': 128,
            'learning_rate': 0.001,
            'gamma': 0.99,
            'epsilon': 1.0,
            'epsilon_decay': 0.995,
            'epsilon_min': 0.01,
            'memory_size': 10000,
            'batch_size': 64,
            'nodes': [
                {'id': 'solar_array_1', 'type': 'generator', 'capacity': 500, 'efficiency': 0.95, 'cost_factor': 0.3},
                {'id': 'wind_farm_1', 'type': 'generator', 'capacity': 300, 'efficiency': 0.9, 'cost_factor': 0.4},
                {'id': 'natural_gas_1', 'type': 'generator', 'capacity': 800, 'efficiency': 0.75, 'cost_factor': 1.0},
                {'id': 'grid_connection', 'type': 'generator', 'capacity': 1000, 'efficiency': 0.98, 'cost_factor': 1.2},
                {'id': 'battery_storage', 'type': 'storage', 'capacity': 400, 'efficiency': 0.92, 'cost_factor': 0.2},
                {'id': 'residential_area', 'type': 'consumer', 'capacity': 600, 'efficiency': 1.0, 'cost_factor': 0.0},
                {'id': 'commercial_zone', 'type': 'consumer', 'capacity': 800, 'efficiency': 1.0, 'cost_factor': 0.0},
                {'id': 'industrial_park', 'type': 'consumer', 'capacity': 700, 'efficiency': 1.0, 'cost_factor': 0.0}
            ]
        }
        
        logger.info("Using default configuration")
        return default_config
    
    def _create_nodes(self):
        """Create grid nodes from configuration."""
        nodes = []
        for node_config in self.config.get('nodes', []):
            node = GridNode(
                node_id=node_config.get('id'),
                node_type=node_config.get('type'),
                capacity=node_config.get('capacity'),
                efficiency=node_config.get('efficiency', 0.95),
                cost_factor=node_config.get('cost_factor', 1.0)
            )
            nodes.append(node)
        
        if not nodes:
            logger.warning("No nodes defined in configuration. Using default nodes.")
            # Create default nodes if none defined
            nodes = [
                GridNode('solar_default', 'generator', 200, 0.95, 0.3),
                GridNode('grid_default', 'generator', 500, 0.98, 1.2),
                GridNode('battery_default', 'storage', 100, 0.9, 0.2),
                GridNode('consumers_default', 'consumer', 300, 1.0, 0.0)
            ]
        
        return nodes
    
    def train(self, episodes=100, max_steps=None, save_interval=10, render=False):
        """
        Train the load balancing agent.
        
        Parameters:
        -----------
        episodes : int
            Number of episodes to train
        max_steps : int or None
            Maximum steps per episode (uses env default if None)
        save_interval : int
            How often to save the model (episodes)
        render : bool
            Whether to render environment during training
        
        Returns:
        --------
        dict : Training metrics
        """
        max_steps = max_steps or self.env.max_episode_steps
        metrics = {'episode_rewards': [], 'avg_losses': [], 'grid_balances': []}
        
        logger.info(f"Starting training for {episodes} episodes")
        
        for episode in range(1, episodes + 1):
            # Reset environment
            state, _ = self.env.reset()
            total_reward = 0
            losses = []
            
            for step in range(max_steps):
                # Choose action
                action = self.agent.act(state)
                
                # Take action
                next_state, reward, done, _, info = self.env.step(action)
                
                # Remember experience
                self.agent.remember(state, action, reward, next_state, done)
                
                # Train
                loss = self.agent.replay()
                if loss > 0:
                    losses.append(loss)
                
                # Update state and rewards
                state = next_state
                total_reward += reward
                
                # Periodically update target network
                if step % 10 == 0:
                    self.agent.update_target_model()
                
                # Render if requested
                if render and step % 10 == 0:
                    self._render_state()
                
                if done:
                    break
            
            # Update metrics
            metrics['episode_rewards'].append(total_reward)
            metrics['avg_losses'].append(np.mean(losses) if losses else 0)
            metrics['grid_balances'].append(self.env.grid_balance)
            
            # Log progress
            logger.info(f"Episode {episode}/{episodes}, Reward: {total_reward:.2f}, "
                       f"Epsilon: {self.agent.epsilon:.4f}, "
                       f"Grid Balance: {self.env.grid_balance:.2f} kW")
            
            # Save model periodically
            if episode % save_interval == 0:
                self.agent.save()
                
            # Track best model
            if total_reward > self.best_reward:
                self.best_reward = total_reward
                self.agent.save(self.model_path.replace(".h5", "_best.h5"))
        
        # Final save
        self.agent.save()
        self.total_episodes += episodes
        
        logger.info(f"Training completed. Total episodes: {self.total_episodes}")
        return metrics
    
    def run(self, duration=24, time_step=None, render=False):
        """
        Run the load balancer with the trained model.
        
        Parameters:
        -----------
        duration : int
            Duration to run in hours
        time_step : int or None
            Time step in minutes (uses env default if None)
        render : bool
            Whether to render state periodically
            
        Returns:
        --------
        dict : Run metrics
        """
        time_step = time_step or self.env.time_step
        steps = int((duration * 60) / time_step)
        
        # Reset environment
        state, _ = self.env.reset()
        
        # Track metrics
        metrics = {
            'rewards': [],
            'grid_balance': [],
            'total_production': [],
            'total_demand': [],
            'timestamps': []
        }
        
        logger.info(f"Running load balancer for {duration} hours (time_step={time_step}min)")
        
        self.is_running = True
        for step in range(steps):
            if not self.is_running:
                logger.info("Run stopped early.")
                break
                
            # Choose action (exploit only - no exploration)
            epsilon_backup = self.agent.epsilon
            self.agent.epsilon = 0  # No exploration during production run
            action = self.agent.act(state)
            self.agent.epsilon = epsilon_backup
            
            # Take action
            next_state, reward, done, _, info = self.env.step(action)
            
            # Update state
            state = next_state
            
            # Update metrics
            metrics['rewards'].append(reward)
            metrics['grid_balance'].append(self.env.grid_balance)
            metrics['total_production'].append(self.env.total_production)
            metrics['total_demand'].append(self.env.total_demand)
            metrics['timestamps'].append(self.env.current_time.strftime('%H:%M'))
            
            # Render if requested
            if render and step % (60 / time_step) == 0:  # Render every hour
                self._render_state()
                
            # Log progress
            if step % (60 / time_step) == 0:  # Log every hour
                logger.info(f"Hour {step // (60 / time_step)}/{duration}, "
                           f"Balance: {self.env.grid_balance:.2f} kW, "
                           f"Reward: {reward:.2f}")
            
            if done:
                break
        
        self.is_running = False
        logger.info("Run completed.")
        return metrics
    
    def stop(self):
        """Stop an ongoing run."""
        self.is_running = False
        logger.info("Stopping load balancer...")
    
    def _render_state(self):
        """Render the current state (simplified console output)."""
        # Current time
        time_str = self.env.current_time.strftime('%Y-%m-%d %H:%M')
        
        # Node status
        node_status = []
        for node in self.env.nodes:
            if node.node_type == 'generator':
                status = (f"{node.node_id}: {node.current_load:.1f}/{node.capacity:.1f} kW "
                         f"({node.current_load/node.capacity*100:.1f}%)")
                node_status.append(status)
            elif node.node_type == 'consumer':
                status = (f"{node.node_id}: {node.current_load:.1f}/{node.capacity:.1f} kW "
                         f"({node.current_load/node.capacity*100:.1f}%)")
                node_status.append(status)
            elif node.node_type == 'storage':
                status = (f"{node.node_id}: {node.stored_energy:.1f}/{node.max_storage:.1f} kWh "
                         f"({node.stored_energy/node.max_storage*100:.1f}%), "
                         f"Current: {node.current_load:+.1f} kW")
                node_status.append(status)
        
        # Grid balance
        balance_status = (f"Grid Balance: {self.env.grid_balance:+.1f} kW, "
                         f"Production: {self.env.total_production:.1f} kW, "
                         f"Demand: {self.env.total_demand:.1f} kW")
        
        # Weather conditions
        weather_status = f"Weather: {self.env.weather_conditions.title()}, Temp: {self.env.temperature:.1f}°C"
        
        # Print status
        print("\n" + "="*80)
        print(f"Time: {time_str} | {weather_status}")
        print("-"*80)
        print(balance_status)
        print("-"*80)
        for status in node_status:
            print(status)
        print("="*80 + "\n")
    
    def visualize_results(self, metrics, title="Load Balancer Results"):
        """
        Visualize the results from a training or run session.
        
        Parameters:
        -----------
        metrics : dict
            Results metrics
        title : str
            Title for the visualization
        """
        plt.figure(figsize=(15, 12))
        
        # 1. Grid Balance
        plt.subplot(3, 1, 1)
        if 'timestamps' in metrics:
            x = metrics['timestamps']
            # Only show a subset of timestamps for readability
            step = max(1, len(x) // 20)
            plt.plot(range(len(x)), metrics['grid_balance'], label='Grid Balance')
            plt.xticks(range(0, len(x), step), x[::step], rotation=45)
        else:
            plt.plot(metrics['grid_balance'], label='Grid Balance')
        
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.title('Grid Balance (kW)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 2. Production and Demand
        if 'total_production' in metrics and 'total_demand' in metrics:
            plt.subplot(3, 1, 2)
            plt.plot(metrics['total_production'], label='Production')
            plt.plot(metrics['total_demand'], label='Demand')
            plt.title('Energy Production and Demand (kW)')
            plt.grid(True, alpha=0.3)
            plt.legend()
        elif 'episode_rewards' in metrics:
            plt.subplot(3, 1, 2)
            plt.plot(metrics['episode_rewards'], label='Episode Rewards')
            plt.title('Training Rewards')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        # 3. Rewards
        plt.subplot(3, 1, 3)
        plt.plot(metrics['rewards'], label='Rewards')
        plt.title('Reward Values')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.suptitle(title, fontsize=16)
        plt.subplots_adjust(top=0.92)
        
        # Save figure
        os.makedirs('./output', exist_ok=True)
        filename = f"./output/load_balancer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename)
        logger.info(f"Visualization saved to {filename}")
        
        plt.show()

# Example usage function
def run_load_balancer_example():
    """Example function to demonstrate power load balancing."""
    # Create and initialize load balancer
    balancer = PowerLoadBalancer()
    
    # Quick training
    print("Training the load balancer (10 episodes)...")
    training_metrics = balancer.train(episodes=10, render=True)
    
    # Run the balancer
    print("\nRunning the trained load balancer for 24 hours...")
    run_metrics = balancer.run(duration=24, render=True)
    
    # Visualize results
    balancer.visualize_results(run_metrics, title="24-Hour Load Balancing Simulation")
    
    return balancer, run_metrics

# Uncomment to run example
# if __name__ == "__main__":
#     run_load_balancer_example()

"""
SUMMARY:
--------
This module implements a reinforcement learning-based power load balancer for 
smart grids. The system dynamically redistributes energy loads across different
nodes to optimize efficiency, cost, and stability. Key components include:

1. GridNode: Represents individual nodes in the power grid (generators, consumers, storage)
2. GridEnvironment: Reinforcement learning environment simulating the grid dynamics
3. DQNAgent: Deep Q-Network agent for learning optimal load balancing strategies
4. PowerLoadBalancer: Main class orchestrating the load balancing operations

The system can:
- Learn optimal load balancing strategies through reinforcement learning
- Adapt to changing weather conditions and time-of-day variations
- Balance renewable energy sources, conventional generation, and storage
- Optimize for multiple objectives: grid stability, cost, and resource utilization
- Visualize balancing performance and grid metrics

TODO:
-----
- Implement more sophisticated RL algorithms (DDPG, PPO) for continuous action spaces
- Add support for hierarchical grid structures with substations
- Incorporate forecasting models for demand and renewable generation
- Implement communication protocols for integration with real grid systems
- Add more detailed economic models for pricing and cost optimization
- Create interactive dashboard for monitoring and control
"""

  
