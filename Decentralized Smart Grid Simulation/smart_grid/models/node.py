import numpy as np
from datetime import datetime

class SmartGridNode:
    """Base class for all nodes in the smart grid network."""
    
    def __init__(self, node_id, name, node_type, position=(0, 0)):
        self.id = node_id
        self.name = name
        self.type = node_type
        self.x, self.y = position
        self.energy = 0  # Current energy level (positive=producing, negative=consuming)
        self.balance = 0  # Financial balance
        self.agent = None  # AI agent controlling this node
        
    def update(self, delta_time):
        """Update node state. Override in subclasses."""
        pass
    
    def to_dict(self):
        """Convert node to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type,
            'energy': round(self.energy, 1),
            'balance': round(self.balance, 2),
            'x': self.x,
            'y': self.y,
            'agentType': self.agent.type if self.agent else 'NONE',
            'agentState': self.agent.state if self.agent else 'Inactive',
            'agentConfidence': self.agent.confidence if self.agent else 0,
            'strategy': self.agent.strategy if self.agent else 'None',
            'lastDecision': self.agent.last_decision if self.agent else 'None',
            'learningRate': self.agent.learning_rate if self.agent else 0,
            'successRate': self.agent.success_rate if self.agent else 0
        }


class SolarHome(SmartGridNode):
    """Solar-powered home that can produce and consume energy."""
    
    def __init__(self, node_id, name, position, capacity=10):
        super().__init__(node_id, name, 'SOLAR_HOME', position)
        self.capacity = capacity  # Maximum production capacity in kW
        
    def update(self, delta_time, weather_condition=0.8):
        """Update solar production based on time and weather."""
        # Simulate solar production based on time of day
        hour = datetime.now().hour
        
        # Simple solar production model (peak at noon)
        if 6 <= hour <= 18:
            # Calculate solar intensity based on hour (peak at noon)
            time_factor = 1.0 - abs(hour - 12) / 6
            
            # Apply weather condition (0 = no sun, 1 = full sun)
            production = self.capacity * time_factor * weather_condition
            
            # Add some randomness
            production += np.random.normal(0, 0.5)
            
            # Home consumption is relatively constant but with small variations
            consumption = 2 + np.random.normal(0, 0.3)
            
            # Net energy production
            self.energy = max(0, production - consumption)
        else:
            # At night, only consumption
            self.energy = -2 + np.random.normal(0, 0.3)


class WindFarm(SmartGridNode):
    """Wind farm that produces energy based on wind conditions."""
    
    def __init__(self, node_id, name, position, capacity=50):
        super().__init__(node_id, name, 'WIND_FARM', position)
        self.capacity = capacity  # Maximum production capacity in kW
        
    def update(self, delta_time, wind_speed=5.0):
        """Update wind production based on wind speed."""
        # Wind turbines typically start producing at 3-4 m/s and reach 
        # capacity around 12-15 m/s, shutting down at ~25 m/s
        
        if wind_speed < 3.0:
            production = 0
        elif wind_speed > 25.0:
            production = 0  # Safety shutdown
        elif wind_speed > 12.0:
            production = self.capacity
        else:
            # Cubic relationship between wind speed and power
            wind_factor = ((wind_speed - 3) / 9) ** 3
            production = self.capacity * wind_factor
        
        # Add some random fluctuation
        production += np.random.normal(0, production * 0.1)
        
        self.energy = max(0, production)


class Battery(SmartGridNode):
    """Energy storage battery that can charge and discharge."""
    
    def __init__(self, node_id, name, position, capacity=100, charge=50):
        super().__init__(node_id, name, 'BATTERY', position)
        self.capacity = capacity  # Maximum storage in kWh
        self.charge_level = charge  # Current charge in kWh
        self.max_rate = 20  # Maximum charge/discharge rate in kW
        
    def update(self, delta_time, charge_decision=0):
        """
        Update battery state.
        charge_decision: -1 (discharge), 0 (hold), 1 (charge)
        """
        # Convert decision to actual rate (-max_rate to +max_rate)
        rate = charge_decision * self.max_rate
        
        # Apply some noise to the rate
        rate += np.random.normal(0, 1)
        
        # Ensure within limits
        rate = max(-self.max_rate, min(self.max_rate, rate))
        
        # Calculate energy transferred in this time step
        energy_delta = rate * delta_time
        
        # Ensure we don't over-discharge or over-charge
        if self.charge_level + energy_delta < 0:
            energy_delta = -self.charge_level
        elif self.charge_level + energy_delta > self.capacity:
            energy_delta = self.capacity - self.charge_level
        
        # Update charge level
        self.charge_level += energy_delta
        
        # Current energy flow (positive when discharging, negative when charging)
        self.energy = -rate  # Negative rate means charging (consuming energy)


class Consumer(SmartGridNode):
    """Energy consumer that represents a building or industrial facility."""
    
    def __init__(self, node_id, name, position, peak_demand=30):
        super().__init__(node_id, name, 'CONSUMER', position)
        self.peak_demand = peak_demand
        
    def update(self, delta_time):
        """Update consumer energy demand based on time of day."""
        hour = datetime.now().hour
        
        # Model different consumption patterns based on time of day
        if 9 <= hour <= 17:  # Business hours
            demand_factor = 0.8 + 0.2 * np.sin((hour - 9) * np.pi / 8)  # Peak at midday
        elif 18 <= hour <= 22:  # Evening peak
            demand_factor = 0.7 + 0.3 * np.sin((hour - 18) * np.pi / 4)  # Evening peak
        else:  # Night
            demand_factor = 0.3 + 0.1 * np.random.random()
        
        # Apply base demand factor and add randomness
        demand = self.peak_demand * demand_factor * (0.9 + 0.2 * np.random.random())
        
        # Consumers have negative energy (they use energy)
        self.energy = -demand
