import numpy as np
from datetime import datetime, timedelta

from smart_grid.models.node import SolarHome, WindFarm, Battery, Consumer
from smart_grid.models.agent import OptimizerAgent, ForecasterAgent, TraderAgent, BalancerAgent

class SmartGridSimulation:
    """Main simulation class for the smart grid."""
    
    def __init__(self):
        self.nodes = []
        self.trading_activity = []
        self.price_history = []
        self.ai_performance = []
        self.cumulative_rewards = {}
        self.start_time = datetime.now()
        self.current_time = self.start_time
        self.time_step = 5  # minutes
        
        # Grid statistics
        self.grid_stats = {
            'total_production': 0,
            'total_consumption': 0,
            'trading_volume': 0,
            'avg_price': 0.12,
            'carbon_saved': 0
        }
        
        # Environment conditions
        self.weather = {
            'temperature': 20,  # Celsius
            'cloud_cover': 0.3,  # 0-1 scale
            'wind_speed': 5.0,  # m/s
        }
        
        # Initialize a default set of nodes
        self._initialize_default_grid()
        
    def _initialize_default_grid(self):
        """Set up a default grid with some standard nodes."""
        # Create some solar homes
        for i in range(3):
            position = (100 + i*50, 100 + i*30)
            node = SolarHome(i+1, f"Solar Home {i+1}", position, capacity=5 + i*2)
            node.agent = OptimizerAgent()
            self.nodes.append(node)
        
        # Create a wind farm
        node = WindFarm(4, "Wind Farm", (200, 150), capacity=45)
        node.agent = ForecasterAgent()
        self.nodes.append(node)
        
        # Create a battery storage
        node = Battery(5, "Community Battery", (150, 200), capacity=100, charge=50)
        node.agent = BalancerAgent()
        self.nodes.append(node)
        
        # Create some consumers
        for i in range(2):
            position = (250 + i*40, 100 + i*40)
            node = Consumer(6+i, f"Consumer {i+1}", position, peak_demand=15 + i*5)
            node.agent = TraderAgent()
            self.nodes.append(node)
    
    def update_weather(self):
        """Update environmental conditions."""
        # Time of day affects temperature
        hour = self.current_time.hour
        day_progress = (hour - 6) / 12  # 0 at 6AM, 1 at 6PM
        
        # Temperature varies through the day
        if 6 <= hour < 18:
            # Temperature rises during the day, peaks around 3PM
            peak_offset = abs(day_progress - 0.75) * 4  # 0 at 3PM, increases as we move away
            self.weather['temperature'] = 20 + 8 * (1 - peak_offset)  # Base temp + daily variation
        else:
            # Temperature drops at night
            self.weather['temperature'] = 20 - 5 + np.random.normal(0, 1) 
        
        # Cloud cover changes gradually
        self.weather['cloud_cover'] = min(1.0, max(0.0, 
            self.weather['cloud_cover'] + np.random.normal(0, 0.1)))
        
        # Wind speed changes gradually
        self.weather['wind_speed'] = min(15.0, max(0.0, 
            self.weather['wind_speed'] + np.random.normal(0, 0.5)))
    
    def update(self):
        """Run one simulation step."""
        # Update time
        self.current_time += timedelta(minutes=self.time_step)
        
        # Update weather conditions
        self.update_weather()
        
        # Update all nodes
        for node in self.nodes:
            if isinstance(node, SolarHome):
                # Pass weather condition (1 - cloud_cover)
                node.update(self.time_step/60, 1 - self.weather['cloud_cover'])
            elif isinstance(node, WindFarm):
                # Pass wind speed
                node.update(self.time_step/60, self.weather['wind_speed'])
            elif isinstance(node, Battery):
                # Pass charge decision from agent
                agent_decision = node.agent.decide(node, self.get_grid_state())
                charge_decision = 0  # Default: hold
                if agent_decision == "CHARGE":
                    charge_decision = -1  # Negative = charging (consuming energy)
                elif agent_decision == "DISCHARGE":
                    charge_decision = 1   # Positive = discharging (providing energy)
                node.update(self.time_step/60, charge_decision)
            else:
                # Generic update
                node.update(self.time_step/60)
        
        # Update grid statistics
        self._update_grid_stats()
        
        # Update price history
        self.price_history.append(self.grid_stats['avg_price'])
        if len(self.price_history) > 288:  # Keep last 24 hours (at 5-min intervals)
            self.price_history = self.price_history[-288:]
        
        # Generate trading activity
        self._generate_trades()
        
        # Update AI performance metrics
        if len(self.ai_performance) == 0 or np.random.random() < 0.05:  # Occasionally update
            self._update_ai_performance()
            
        # Update agent learning
        self._update_agent_learning()
    
    def _update_grid_stats(self):
        """Update aggregate grid statistics."""
        total_production = 0
        total_consumption = 0
        
        for node in self.nodes:
            if node.energy > 0:
                total_production += node.energy
            else:
                total_consumption -= node.energy  # Convert negative to positive
        
        # Update grid stats
        self.grid_stats['total_production'] = total_production
        self.grid_stats['total_consumption'] = total_consumption
        
        # Update price based on supply/demand balance
        supply_demand_ratio = total_production / max(1, total_consumption)
        price_adjustment = 0
        
        if supply_demand_ratio > 1.2:  # Excess supply
            price_adjustment = -0.002 * (supply_demand_ratio - 1.2)
        elif supply_demand_ratio < 0.8:  # Excess demand
            price_adjustment = 0.003 * (0.8 - supply_demand_ratio)
        
        # Add some random noise
        price_adjustment += np.random.normal(0, 0.001)
        
        # Apply adjustment with limits
        self.grid_stats['avg_price'] = max(0.08, min(0.20, 
            self.grid_stats['avg_price'] + price_adjustment))
        
        # Carbon saved calculation (simplified)
        # Assume renewable energy saves 0.5 kg CO2 per kWh compared to fossil fuels
        carbon_saved_rate = 0.5  # kg CO2 per kWh
        self.grid_stats['carbon_saved'] += total_production * carbon_saved_rate * (self.time_step / 60)
    
    def _generate_trades(self):
        """Generate trades between nodes."""
        # Clear previous trades
        self.trading_activity = []
        
        # Identify producers and consumers
        producers = [node for node in self.nodes if node.energy > 0]
        consumers = [node for node in self.nodes if node.energy < 0]
        
        # Each consumer tries to fulfill its needs
        for consumer in consumers:
            # Decide how much to buy based on agent
            energy_needed = abs(consumer.energy)
            
            # Randomize the number of producers to buy from
            num_producers = min(len(producers), 1 + int(np.random.random() * 2))
            
            for _ in range(num_producers):
                if not producers:
                    break
                    
                # Select a producer (randomly for now)
                producer_idx = np.random.randint(0, len(producers))
                producer = producers[producer_idx]
                
                # Determine trade amount
                max_available = producer.energy
                trade_amount = min(energy_needed / num_producers, max_available)
                
                if trade_amount > 0.5:  # Minimum trade threshold
                    # Slightly randomize price around current grid average
                    price_factor = 0.95 + np.random.random() * 0.1  # 0.95 - 1.05
                    trade_price = self.grid_stats['avg_price'] * price_factor
                    
                    # Record the trade
                    self.trading_activity.append({
                        'source': producer.id,
                        'target': consumer.id,
                        'amount': round(trade_amount, 1),
                        'price': round(trade_price, 3),
                        'aiDecision': np.random.random() > 0.2  # 80% AI-decided
                    })
                    
                    # Update energy and balance for both nodes
                    producer.energy -= trade_amount
                    consumer.energy += trade_amount
                    
                    producer.balance += trade_amount * trade_price
                    consumer.balance -= trade_amount * trade_price
                    
                    # Update trading volume stat
                    self.grid_stats['trading_volume'] += trade_amount
                    
                    # If producer has little energy left, remove from available producers
                    if producer.energy < 0.5:
                        producers.pop(producer_idx)
                        
                    energy_needed -= trade_amount
                    
                if energy_needed < 0.5:
                    break
    
    def _update_ai_performance(self):
        """Update AI performance metrics."""
        if not self.ai_performance:
            # Initialize with starting values
            self.ai_performance.append({
                'day': 1,
                'efficiency': 50 + np.random.random() * 10,
                'savings': 5 + np.random.random() * 5,
                'agentLearning': 40 + np.random.random() * 10
            })
        else:
            # Get last entry and improve on it
            last = self.ai_performance[-1]
            
            # Improvements diminish over time with some randomness
            efficiency_gain = max(0, 5 * (1 - last['efficiency']/100) + np.random.normal(0, 1))
            savings_gain = max(0, 3 * (1 - last['savings']/50) + np.random.normal(0, 0.5))
            learning_gain = max(0, 6 * (1 - last['agentLearning']/100) + np.random.normal(0, 1.2))
            
            self.ai_performance.append({
                'day': last['day'] + 1,
                'efficiency': min(98, last['efficiency'] + efficiency_gain),
                'savings': min(50, last['savings'] + savings_gain),
                'agentLearning': min(99, last['agentLearning'] + learning_gain)
            })
            
            # Keep only recent history
            if len(self.ai_performance) > 10:
                self.ai_performance = self.ai_performance[-10:]
    
    def _update_agent_learning(self):
        """Update learning metrics for all agents."""
        for node in self.nodes:
            if node.agent:
                # Calculate reward based on current node state
                if node.energy > 0:  # Producing
                    reward = node.energy * self.grid_stats['avg_price']
                else:  # Consuming or neutral
                    reward = node.energy * self.grid_stats['avg_price'] * -0.5  # Penalty for consumption
                
                # Add some balance improvement component to reward
                reward += node.balance * 0.01
                
                # Apply some randomization
                reward = reward * (0.8 + np.random.random() * 0.4)
                
                # Update agent learning
                node.agent.learn(reward)
                
                # Track cumulative rewards
                if node.id not in self.cumulative_rewards:
                    self.cumulative_rewards[node.id] = []
                
                last_cum_reward = self.cumulative_rewards[node.id][-1] if self.cumulative_rewards[node.id] else 0
                self.cumulative_rewards[node.id].append(last_cum_reward + reward)
    
    def get_grid_state(self):
        """Return current state of the grid as a dictionary."""
        return {
            'timestamp': self.current_time.strftime('%H:%M:%S'),
            'total_production': self.grid_stats['total_production'],
            'total_consumption': self.grid_stats['total_consumption'],
            'avg_price': self.grid_stats['avg_price'],
            'price_history': self.price_history[-24:] if len(self.price_history) > 24 else self.price_history
        }
    
    def get_full_state(self):
        """Return complete simulation state for frontend."""
        return {
            'timestamp': self.current_time.strftime('%H:%M:%S'),
            'gridStats': self.grid_stats,
            'priceHistory': [
                {'hour': i, 'price': price} 
                for i, price in enumerate(self.price_history[-24:])
            ],
            'nodeData': [node.to_dict() for node in self.nodes],
            'tradingActivity': self.trading_activity,
            'aiPerformance': self.ai_performance,
            'weather': self.weather
        }
    
    def get_node_rl_metrics(self, node_id):
        """Return reinforcement learning metrics for a specific node."""
        node = next((n for n in self.nodes if n.id == node_id), None)
        if not node or not node.agent:
            return None
        
        agent = node.agent
        rewards = agent.rewards
        
        # Format metrics for front-end
        reward_history = [{'step': i, 'reward': r, 'cumulative': self.cumulative_rewards[node_id][i] 
                           if i < len(self.cumulative_rewards[node_id]) else 0} 
                          for i, r in enumerate(rewards)]
        
        q_value_history = []
        for i in range(len(agent.q_values['buy'])):
            q_value_history.append({
                'step': i,
                'buy': agent.q_values['buy'][i],
                'sell': agent.q_values['sell'][i],
                'hold': agent.q_values['hold'][i]
            })
        
        loss_history = [{'step': i, 'loss': loss} for i, loss in enumerate(agent.losses)]
        
        return {
            'rewardHistory': reward_history,
            'qValueHistory': q_value_history,
            'lossHistory': loss_history
        }
