import numpy as np

class AgentBase:
    """Base class for all AI agents in the smart grid."""
    
    def __init__(self, agent_type, strategy):
        self.type = agent_type
        self.strategy = strategy
        self.state = "Idle"
        self.confidence = 75 + np.random.random() * 20  # 75-95% confidence
        self.last_decision = "None"
        self.learning_rate = 0.01 + np.random.random() * 0.05  # 0.01-0.06
        self.success_rate = 70 + np.random.random() * 20  # 70-90% success rate
        
        # Reinforcement learning metrics
        self.rewards = []
        self.q_values = {'buy': [], 'sell': [], 'hold': []}
        self.losses = []
        
    def decide(self, node, grid_state):
        """Make a decision for the node based on grid state."""
        # Override in subclasses
        pass
    
    def learn(self, reward):
        """Update agent's knowledge based on reward."""
        # Basic implementation - subclasses can override
        self.rewards.append(reward)
        
        # Simulate Q-value updates
        for action in self.q_values:
            if len(self.q_values[action]) == 0:
                self.q_values[action].append(np.random.random() * 10)
            else:
                prev_q = self.q_values[action][-1]
                new_q = prev_q + self.learning_rate * (reward + np.random.normal(0, 2) - prev_q)
                self.q_values[action].append(new_q)
        
        # Simulate loss updates
        if len(self.losses) == 0:
            self.losses.append(10 + np.random.random() * 5)
        else:
            prev_loss = self.losses[-1]
            new_loss = max(0, prev_loss * (0.95 + np.random.random() * 0.1))
            self.losses.append(new_loss)


class OptimizerAgent(AgentBase):
    """Agent that maximizes energy efficiency and profit."""
    
    def __init__(self):
        super().__init__("OPTIMIZER", "Maximize Profit")
        
    def decide(self, node, grid_state):
        self.state = "Optimizing"
        
        # If node produces excess energy, sell it
        if node.energy > 0:
            self.last_decision = "Sell Excess Energy"
            self.confidence = 80 + np.random.random() * 15
            return "SELL"
        # If energy price is low, consider buying
        elif grid_state['avg_price'] < 0.11:
            self.last_decision = "Buy at Low Price"
            self.confidence = 70 + np.random.random() * 20
            return "BUY"
        else:
            self.last_decision = "Hold Position"
            self.confidence = 60 + np.random.random() * 20
            return "HOLD"


class ForecasterAgent(AgentBase):
    """Agent that predicts future energy needs and prices."""
    
    def __init__(self):
        super().__init__("FORECASTER", "Predict Demand")
        
    def decide(self, node, grid_state):
        self.state = "Forecasting"
        
        # Simple forecasting logic based on historical prices
        price_trend = 0
        if len(grid_state['price_history']) > 2:
            recent_prices = grid_state['price_history'][-3:]
            price_trend = recent_prices[-1] - recent_prices[0]
        
        if price_trend > 0.005:  # Rising price trend
            self.last_decision = "Predict Price Increase"
            self.confidence = 85 + np.random.random() * 10
            return "CHARGE"  # Save energy for later when prices are higher
        elif price_trend < -0.005:  # Falling price trend
            self.last_decision = "Predict Price Decrease"
            self.confidence = 80 + np.random.random() * 15
            return "DISCHARGE"  # Use energy now while prices are higher
        else:
            self.last_decision = "Stable Price Prediction"
            self.confidence = 75 + np.random.random() * 15
            return "HOLD"


class TraderAgent(AgentBase):
    """Agent that negotiates and executes energy exchanges."""
    
    def __init__(self):
        super().__init__("TRADER", "Minimize Cost")
        
    def decide(self, node, grid_state):
        self.state = "Trading"
        
        # If we're a consumer with negative energy (need energy)
        if node.energy < 0:
            # Check if current price is lower than recent average
            recent_avg = sum(grid_state['price_history'][-5:]) / 5 if len(grid_state['price_history']) >= 5 else grid_state['avg_price']
            
            if grid_state['avg_price'] < recent_avg * 0.95:
                self.last_decision = "Buy at Low Price"
                self.confidence = 85 + np.random.random() * 10
                return "BUY_MORE"  # Buy extra to store
            else:
                self.last_decision = "Buy Minimum Required"
                self.confidence = 75 + np.random.random() * 15
                return "BUY_MIN"
        else:
            self.last_decision = "Monitor Market"
            self.confidence = 70 + np.random.random() * 20
            return "MONITOR"


class BalancerAgent(AgentBase):
    """Agent that ensures grid stability and reliability."""
    
    def __init__(self):
        super().__init__("BALANCER", "Balance Grid")
        
    def decide(self, node, grid_state):
        self.state = "Balancing"
        
        # Assess grid balance (are we producing more than consuming?)
        production = grid_state['total_production']
        consumption = grid_state['total_consumption']
        balance = production - consumption
        
        if node.type == 'BATTERY':
            if balance > 10:  # Excess production
                self.last_decision = "Store Excess Energy"
                self.confidence = 90 + np.random.random() * 10
                return "CHARGE"
            elif balance < -10:  # Shortage
                self.last_decision = "Release Stored Energy"
                self.confidence = 85 + np.random.random() * 15
                return "DISCHARGE"
            else:  # Relatively balanced
                self.last_decision = "Maintain Grid Stability"
                self.confidence = 80 + np.random.random() * 15
                return "HOLD"
        else:
            self.last_decision = "Support Grid Operations"
            self.confidence = 75 + np.random.random() * 20
            return "SUPPORT"
