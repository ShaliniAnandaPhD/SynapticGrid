"""
A reinforcement learning model to simulate and improve robotic waste sorting.
This simulation helps us train our sorting algorithms before deploying them
to the actual robots, saving time and preventing expensive mistakes.
"""

import numpy as np
import pandas as pd
import gym
from gym import spaces
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import random
import time
import os
import json
import logging
from typing import Dict, List, Tuple, Any, Optional

# Setting up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("sorting_sim.log"), logging.StreamHandler()]
)
logger = logging.getLogger("SortingSimulation")

# Waste material categories - might need to update these as we get more specific
WASTE_CATEGORIES = [
    'plastic_recyclable',
    'plastic_non_recyclable',
    'paper',
    'cardboard',
    'glass',
    'metal_recyclable',
    'metal_non_recyclable',
    'organic',
    'electronic',
    'hazardous',
    'mixed',
    'unknown'
]

# Physical properties that can be detected by the robot
MATERIAL_PROPERTIES = [
    'weight',           # in grams
    'size',             # normalized 0-1 value
    'color_r',          # RGB values
    'color_g',
    'color_b',
    'metal_content',    # 0-1 value from metal detector
    'transparency',     # 0-1 value
    'density',          # g/cm³
    'moisture',         # 0-1 value
    'texture',          # smoothness, 0-1 value
    'rigidity',         # how bendy/rigid, 0-1
    'spectral_signature' # simplified to 0-1 value
]


class WasteItem:
    """
    Represents a single waste item with physical properties and true category.
    
    This class simulates the physical properties of waste items that sensors
    would detect in the real world. Our robot doesn't know the true category
    upfront - it has to infer it from the properties.
    """
    
    def __init__(self, category: str = None):
        """
        Create a waste item, either random or from a specific category.
        
        Args:
            category: Optional waste category. If None, a random category is assigned.
        """
        # If no category provided, pick one randomly
        if category is None or category not in WASTE_CATEGORIES:
            self.true_category = random.choice(WASTE_CATEGORIES)
        else:
            self.true_category = category
            
        # Generate properties based on the true category
        self.properties = self._generate_properties()
    
    def _generate_properties(self) -> Dict[str, float]:
        """
        Generate realistic properties based on the waste category.
        
        This is where our domain knowledge about different materials comes in.
        These are simplified - real world is way messier but this works for simulation.
        """
        properties = {}
        
        # Base random values
        for prop in MATERIAL_PROPERTIES:
            properties[prop] = random.random()
        
        # Now adjust based on category - this is super simplified
        if self.true_category == 'plastic_recyclable':
            properties['weight'] = random.uniform(5, 150)  # g
            properties['metal_content'] = random.uniform(0, 0.05)
            properties['transparency'] = random.uniform(0.3, 0.9)
            properties['density'] = random.uniform(0.8, 1.5)  # g/cm³
            properties['moisture'] = random.uniform(0, 0.2)
            properties['texture'] = random.uniform(0.7, 1.0)  # smooth
            
        elif self.true_category == 'plastic_non_recyclable':
            properties['weight'] = random.uniform(5, 200)
            properties['metal_content'] = random.uniform(0, 0.2)
            properties['transparency'] = random.uniform(0, 0.5)
            properties['density'] = random.uniform(0.8, 2.0)
            properties['moisture'] = random.uniform(0, 0.3)
            
        elif self.true_category == 'paper':
            properties['weight'] = random.uniform(1, 100)
            properties['metal_content'] = random.uniform(0, 0.01)
            properties['transparency'] = random.uniform(0, 0.1)
            properties['density'] = random.uniform(0.7, 1.2)
            properties['moisture'] = random.uniform(0, 0.7)  # can get wet
            properties['rigidity'] = random.uniform(0, 0.3)  # very flexible
            
        elif self.true_category == 'cardboard':
            properties['weight'] = random.uniform(10, 500)
            properties['metal_content'] = random.uniform(0, 0.01)
            properties['transparency'] = random.uniform(0, 0.05)
            properties['density'] = random.uniform(0.5, 0.9)
            properties['moisture'] = random.uniform(0, 0.5)
            properties['rigidity'] = random.uniform(0.3, 0.7)
            
        elif self.true_category == 'glass':
            properties['weight'] = random.uniform(50, 800)
            properties['metal_content'] = random.uniform(0, 0.05)
            properties['transparency'] = random.uniform(0.5, 1.0)
            properties['density'] = random.uniform(2.0, 3.0)
            properties['moisture'] = random.uniform(0, 0.3)
            properties['texture'] = random.uniform(0.8, 1.0)  # very smooth
            properties['rigidity'] = random.uniform(0.9, 1.0)  # very rigid
            
        elif self.true_category == 'metal_recyclable':
            properties['weight'] = random.uniform(10, 500)
            properties['metal_content'] = random.uniform(0.7, 1.0)
            properties['transparency'] = random.uniform(0, 0.1)
            properties['density'] = random.uniform(2.0, 8.0)
            properties['moisture'] = random.uniform(0, 0.2)
            properties['texture'] = random.uniform(0.6, 1.0)
            properties['rigidity'] = random.uniform(0.7, 1.0)
            
        elif self.true_category == 'metal_non_recyclable':
            properties['weight'] = random.uniform(10, 800)
            properties['metal_content'] = random.uniform(0.5, 1.0)
            properties['transparency'] = random.uniform(0, 0.1)
            properties['density'] = random.uniform(2.0, 10.0)
            properties['moisture'] = random.uniform(0, 0.3)
            
        elif self.true_category == 'organic':
            properties['weight'] = random.uniform(5, 2000)
            properties['metal_content'] = random.uniform(0, 0.05)
            properties['transparency'] = random.uniform(0, 0.3)
            properties['density'] = random.uniform(0.2, 1.5)
            properties['moisture'] = random.uniform(0.3, 1.0)  # often wet
            properties['rigidity'] = random.uniform(0, 0.5)  # often flexible
            
        elif self.true_category == 'electronic':
            properties['weight'] = random.uniform(20, 5000)
            properties['metal_content'] = random.uniform(0.5, 0.9)
            properties['transparency'] = random.uniform(0, 0.2)
            properties['density'] = random.uniform(0.5, 5.0)  # mixed materials
            properties['moisture'] = random.uniform(0, 0.1)
            properties['spectral_signature'] = random.uniform(0.7, 1.0)  # distinctive
            
        elif self.true_category == 'hazardous':
            properties['weight'] = random.uniform(50, 2000)
            properties['spectral_signature'] = random.uniform(0.8, 1.0)  # distinctive
            
        elif self.true_category == 'mixed' or self.true_category == 'unknown':
            # Just use the random values already assigned
            pass
            
        # Add some noise to make it realistic - 5% error
        for prop in properties:
            noise = random.uniform(-0.05, 0.05) * properties[prop]
            properties[prop] = max(0, min(properties[prop] + noise, 1.0)) if prop != 'weight' and prop != 'density' else properties[prop] + noise
        
        return properties
    
    def get_feature_vector(self) -> np.ndarray:
        """
        Convert properties to a feature vector for the model.
        
        Returns:
            NumPy array of normalized properties
        """
        # Create a feature vector in a consistent order
        feature_vec = []
        
        for prop in MATERIAL_PROPERTIES:
            value = self.properties.get(prop, 0.0)
            
            # Normalize weight and density specially since they have larger ranges
            if prop == 'weight':
                # Normalize weight to 0-1 range (assuming max 5kg)
                value = min(value / 5000.0, 1.0)
            elif prop == 'density':
                # Normalize density to 0-1 range (assuming max 10 g/cm³)
                value = min(value / 10.0, 1.0)
                
            feature_vec.append(value)
            
        return np.array(feature_vec, dtype=np.float32)
    
    def __repr__(self) -> str:
        return f"WasteItem({self.true_category})"


class SortingEnv(gym.Env):
    """
    Reinforcement learning environment for waste sorting.
    
    This simulates a conveyor belt with waste items that a robotic arm
    needs to sort into the correct bins.
    """
    
    def __init__(self, difficulty: str = 'medium', batch_size: int = 10, 
                 noise_level: float = 0.05, contamination_rate: float = 0.1):
        """
        Initialize the sorting environment.
        
        Args:
            difficulty: 'easy', 'medium', or 'hard', affects item ambiguity
            batch_size: How many items per episode
            noise_level: How much sensor noise (0.0-1.0)
            contamination_rate: Rate of mixed/contaminated items (0.0-1.0)
        """
        super(SortingEnv, self).__init__()
        
        self.difficulty = difficulty
        self.batch_size = batch_size
        self.noise_level = noise_level
        self.contamination_rate = contamination_rate
        self.current_item_idx = 0
        self.current_batch = []
        self.episode_rewards = 0
        self.correct_sorts = 0
        
        # Define action and observation space
        # There's one action for each waste category (put in corresponding bin)
        self.action_space = spaces.Discrete(len(WASTE_CATEGORIES))
        
        # Observation is a feature vector of material properties
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(len(MATERIAL_PROPERTIES),), dtype=np.float32
        )
        
        logger.info(f"Created sorting environment with difficulty={difficulty}, "
                   f"batch_size={batch_size}, noise={noise_level}")
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment for a new episode.
        
        Returns:
            Initial observation (feature vector of first waste item)
        """
        # Generate a new batch of waste items
        self.current_batch = self._generate_batch()
        self.current_item_idx = 0
        self.episode_rewards = 0
        self.correct_sorts = 0
        
        # Return the first item's features
        return self._get_observation()
    
    def _generate_batch(self) -> List[WasteItem]:
        """
        Generate a batch of waste items based on current difficulty.
        
        Returns:
            List of WasteItem objects
        """
        batch = []
        
        # Select categories based on difficulty
        if self.difficulty == 'easy':
            # Only use a subset of very distinct categories
            categories = ['plastic_recyclable', 'paper', 'glass', 'metal_recyclable', 'organic']
        elif self.difficulty == 'medium':
            # Use most categories
            categories = [cat for cat in WASTE_CATEGORIES if cat not in ['mixed', 'unknown']]
        else:  # hard
            # Use all categories
            categories = WASTE_CATEGORIES
        
        # Generate items
        for _ in range(self.batch_size):
            # Decide if this will be a contaminated item
            if random.random() < self.contamination_rate:
                item = WasteItem('mixed')
            else:
                category = random.choice(categories)
                item = WasteItem(category)
            
            batch.append(item)
            
        return batch
    
    def _get_observation(self) -> np.ndarray:
        """
        Get the current item's feature vector with added sensor noise.
        
        Returns:
            Feature vector with noise
        """
        if self.current_item_idx >= len(self.current_batch):
            # Shouldn't happen, but just in case
            logger.warning("Tried to get observation past end of batch")
            return np.zeros(len(MATERIAL_PROPERTIES), dtype=np.float32)
            
        # Get the base feature vector
        features = self.current_batch[self.current_item_idx].get_feature_vector()
        
        # Add sensor noise
        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level, size=features.shape)
            # Ensure we don't go below 0 or above 1 after adding noise
            features = np.clip(features + noise, 0, 1)
            
        return features
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a sorting action on the current waste item.
        
        Args:
            action: Integer representing which bin to put the item in
            
        Returns:
            Tuple of (next observation, reward, done, info)
        """
        if self.current_item_idx >= len(self.current_batch):
            logger.warning("Environment called step() when episode was already done")
            return (
                np.zeros(len(MATERIAL_PROPERTIES), dtype=np.float32),
                0,
                True,
                {'correct': False, 'true_category': None, 'predicted_category': None}
            )
        
        # Get the current item
        current_item = self.current_batch[self.current_item_idx]
        true_category = current_item.true_category
        predicted_category = WASTE_CATEGORIES[action]
        
        # Determine reward
        reward = self._calculate_reward(true_category, predicted_category)
        self.episode_rewards += reward
        
        # Check if correct
        correct = (true_category == predicted_category)
        if correct:
            self.correct_sorts += 1
        
        # Move to the next item
        self.current_item_idx += 1
        done = (self.current_item_idx >= len(self.current_batch))
        
        # Get the next observation if not done
        if not done:
            next_obs = self._get_observation()
        else:
            next_obs = np.zeros(len(MATERIAL_PROPERTIES), dtype=np.float32)
        
        # Additional info
        info = {
            'correct': correct,
            'true_category': true_category,
            'predicted_category': predicted_category,
            'accuracy': self.correct_sorts / self.current_item_idx
        }
        
        return (next_obs, reward, done, info)
    
    def _calculate_reward(self, true_category: str, predicted_category: str) -> float:
        """
        Calculate the reward for a sorting action.
        
        We have a complex reward structure that models the real economic and
        environmental impact of correct/incorrect sorting decisions.
        
        Args:
            true_category: The actual category of the waste item
            predicted_category: The category the agent sorted it into
            
        Returns:
            Reward value (positive or negative)
        """
        # Base reward structure
        if true_category == predicted_category:
            return 1.0  # Correct sort
        
        # Special cases for contamination
        if predicted_category == 'recycling' and true_category in ['hazardous', 'electronic']:
            return -5.0  # Big penalty for contaminating recycling with hazardous
            
        if predicted_category == 'organic' and true_category not in ['organic', 'paper']:
            return -3.0  # Big penalty for contaminating compost
            
        if true_category == 'hazardous' and predicted_category != 'hazardous':
            return -4.0  # Big penalty for not identifying hazardous waste
            
        # Small penalty for regular misclassification
        return -1.0


class DQNAgent:
    """
    Deep Q-Network agent for waste sorting.
    
    This implements a reinforcement learning agent that learns to sort waste
    based on the feature vectors of items.
    """
    
    def __init__(self, state_size: int, action_size: int,
                 learning_rate: float = 0.001, gamma: float = 0.95,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.1, batch_size: int = 32,
                 memory_size: int = 10000):
        """
        Initialize the DQN agent.
        
        Args:
            state_size: Size of the state vector (number of features)
            action_size: Number of possible actions (number of waste categories)
            learning_rate: Learning rate for the neural network
            gamma: Discount factor for future rewards
            epsilon: Exploration rate (1.0 = always explore)
            epsilon_decay: Rate at which epsilon decreases
            epsilon_min: Minimum exploration rate
            batch_size: Batch size for training
            memory_size: Size of replay memory
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        # Neural network for predicting Q values
        self.model = self._build_model()
        
        # Target network for stability
        self.target_model = self._build_model()
        self.update_target_model()
        
        # Experience replay memory
        self.memory = []
        self.memory_size = memory_size
        
        logger.info(f"Initialized DQN agent with state_size={state_size}, action_size={action_size}")
    
    def _build_model(self) -> keras.Model:
        """
        Build the neural network model for DQN.
        
        Returns:
            Keras Model
        """
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self) -> None:
        """Update the target model to match the main model weights."""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool) -> None:
        """
        Store experience in replay memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)  # Remove oldest memory if full
            
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Choose an action based on the current state.
        
        Uses epsilon-greedy policy during training.
        
        Args:
            state: Current state vector
            training: Whether we're in training mode (True) or evaluation (False)
            
        Returns:
            Selected action
        """
        if training and np.random.rand() <= self.epsilon:
            # Explore - random action
            return random.randrange(self.action_size)
        
        # Exploit - best action from Q-values
        act_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self) -> float:
        """
        Train the model using experience replay.
        
        Returns:
            Loss value from training
        """
        if len(self.memory) < self.batch_size:
            return 0
            
        # Sample a batch from memory
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = np.zeros((self.batch_size, self.state_size))
        targets = np.zeros((self.batch_size, self.action_size))
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state
            
            # Calculate target Q-value
            target = self.model.predict(np.expand_dims(state, axis=0), verbose=0)[0]
            
            if done:
                target[action] = reward
            else:
                t = self.target_model.predict(np.expand_dims(next_state, axis=0), verbose=0)[0]
                target[action] = reward + self.gamma * np.amax(t)
                
            targets[i] = target
        
        # Train the model
        history = self.model.fit(states, targets, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss
    
    def load(self, name: str) -> None:
        """Load model weights from file."""
        self.model.load_weights(name)
        self.update_target_model()
    
    def save(self, name: str) -> None:
        """Save model weights to file."""
        self.model.save_weights(name)


class SortingSimulation:
    """
    Main class for running the waste sorting simulation.
    
    This class manages the environment, agent, training, evaluation,
    and visualization.
    """
    
    def __init__(self, log_dir: str = "./logs", model_dir: str = "./models"):
        """
        Initialize the simulation.
        
        Args:
            log_dir: Directory for saving logs
            model_dir: Directory for saving models
        """
        self.log_dir = log_dir
        self.model_dir = model_dir
        
        # Make sure directories exist
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        # Default environment settings
        self.env = SortingEnv(difficulty='medium', batch_size=50)
        
        # Create agent
        state_size = len(MATERIAL_PROPERTIES)
        action_size = len(WASTE_CATEGORIES)
        self.agent = DQNAgent(state_size, action_size)
        
        # Training metrics
        self.episode_rewards = []
        self.episode_accuracies = []
        self.train_confusion_matrix = np.zeros((len(WASTE_CATEGORIES), len(WASTE_CATEGORIES)))
        
        logger.info("Sorting simulation initialized")
    
    def train(self, episodes: int = 1000, target_update_freq: int = 10,
             evaluate_freq: int = 100, save_freq: int = 200,
             difficulty_schedule: Dict[int, str] = None) -> None:
        """
        Train the agent on the sorting task.
        
        Args:
            episodes: Number of episodes to train for
            target_update_freq: How often to update target network
            evaluate_freq: How often to evaluate performance
            save_freq: How often to save the model
            difficulty_schedule: Optional dictionary mapping episode numbers to difficulty levels
        """
        logger.info(f"Starting training for {episodes} episodes")
        
        # Setup TensorBoard if available
        try:
            current_time = time.strftime("%Y%m%d-%H%M%S")
            train_log_dir = os.path.join(self.log_dir, f'tensorboard/dqn_{current_time}')
            summary_writer = tf.summary.create_file_writer(train_log_dir)
        except:
            summary_writer = None
            logger.warning("TensorBoard not available, skipping logging")
        
        # Store all episode rewards
        all_rewards = []
        all_accuracies = []
        best_avg_reward = -float('inf')
        
        # Training loop
        for episode in range(1, episodes + 1):
            # Check if we should update difficulty
            if difficulty_schedule and episode in difficulty_schedule:
                new_difficulty = difficulty_schedule[episode]
                logger.info(f"Changing difficulty to {new_difficulty} at episode {episode}")
                self.env = SortingEnv(difficulty=new_difficulty, batch_size=self.env.batch_size)
            
            # Reset environment
            state = self.env.reset()
            episode_reward = 0
            done = False
            
            # Clear confusion matrix for this episode
            confusion_matrix = np.zeros((len(WASTE_CATEGORIES), len(WASTE_CATEGORIES)))
            
            # Episode loop
            while not done:
                # Choose action
                action = self.agent.act(state)
                
                # Take action
                next_state, reward, done, info = self.env.step(action)
                
                # Remember experience
                self.agent.remember(state, action, reward, next_state, done)
                
                # Update state and reward
                state = next_state
                episode_reward += reward
                
                # Update confusion matrix
                if info.get('true_category') is not None:
                    true_idx = WASTE_CATEGORIES.index(info['true_category'])
                    pred_idx = WASTE_CATEGORIES.index(info['predicted_category'])
                    confusion_matrix[true_idx, pred_idx] += 1
                    
            # Train on experiences
            loss = self.agent.replay()
            
            # Update target model periodically
            if episode % target_update_freq == 0:
                self.agent.update_target_model()
            
            # Store metrics
            accuracy = self.env.correct_sorts / self.env.batch_size
            all_rewards.append(episode_reward)
            all_accuracies.append(accuracy)
            
            # Update confusion matrix
            self.train_confusion_matrix = 0.9 * self.train_confusion_matrix + 0.1 * confusion_matrix
            
            # Log to TensorBoard
            if summary_writer:
                with summary_writer.as_default():
                    tf.summary.scalar('episode_reward', episode_reward, step=episode)
                    tf.summary.scalar('accuracy', accuracy, step=episode)
                    tf.summary.scalar('loss', loss, step=episode)
                    tf.summary.scalar('epsilon', self.agent.epsilon, step=episode)
            
            # Print progress
            if episode % 10 == 0:
                avg_reward = np.mean(all_rewards[-10:])
                avg_accuracy = np.mean(all_accuracies[-10:])
                logger.info(f"Episode: {episode}/{episodes}, Reward: {episode_reward:.2f}, "
                          f"Avg Reward: {avg_reward:.2f}, Accuracy: {accuracy:.4f}, "
                          f"Epsilon: {self.agent.epsilon:.4f}")
            
            # Evaluate periodically
            if episode % evaluate_freq == 0:
                eval_reward, eval_accuracy = self.evaluate(5)
                logger.info(f"Evaluation - Reward: {eval_reward:.2f}, Accuracy: {eval_accuracy:.4f}")
                
                # Log evaluation metrics
                if summary_writer:
                    with summary_writer.as_default():
                        tf.summary.scalar('eval_reward', eval_reward, step=episode)
                        tf.summary.scalar('eval_accuracy', eval_accuracy, step=episode)
                
                # Save if best model so far
                avg_reward = np.mean(all_rewards[-evaluate_freq:])
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    self.save_model("best_model")
                    logger.info(f"New best model saved with avg reward: {best_avg_reward:.2f}")
            
            # Save model periodically
            if episode % save_freq == 0:
                self.save_model(f"model_episode_{episode}")
                
                # Save confusion matrix visualization
                self.visualize_confusion_matrix(
                    save_path=os.path.join(self.log_dir, f"confusion_matrix_ep{episode}.png")
                )
        
        # Final save
        self.save_model("final_model")
        logger.info("Training completed")
        
        # Save final metrics
        self.episode_rewards = all_rewards
        self.episode_accuracies = all_accuracies
        
        # Plot and save learning curves
        self.visualize_learning_curves(
            save_path=os.path.join(self.log_dir, "learning_curves.png")
        )
    
    def evaluate(self, episodes: int = 10, difficulty: str = None) -> Tuple[float, float]:
        """
        Evaluate the agent's performance.
        
        Args:
            episodes: Number of episodes to evaluate on
            difficulty: Optional difficulty to use, otherwise uses current env difficulty
            
        Returns:
            Tuple of (average reward, average accuracy)
        """
        # Create a new environment for evaluation
        if difficulty:
            eval_env = SortingEnv(difficulty=difficulty, batch_size=self.env.batch_size)
        else:
            eval_env = self.env
        
        total_rewards = 0
        total_accuracy = 0
        
        # Run evaluation episodes
        for _ in range(episodes):
            state = eval_env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Always use exploitation for evaluation
                action = self.agent.act(state, training=False)
                next_state, reward, done, info = eval_env.step(action)
                
                state = next_state
                episode_reward += reward
            
            total_rewards += episode_reward
            total_accuracy += eval_env.correct_sorts / eval_env.batch_size
        
        # Calculate averages
        avg_reward = total_rewards / episodes
        avg_accuracy = total_accuracy / episodes
        
        return avg_reward, avg_accuracy
    
    def save_model(self, name: str) -> None:
        """Save the agent's model with the given name."""
        path = os.path.join(self.model_dir, f"{name}.h5")
        self.agent.save(path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, name: str) -> None:
        """Load the agent's model with the given name."""
        path = os.path.join(self.model_dir, f"{name}.h5")
        self.agent.load(path)
        logger.info(f"Model loaded from {path}")
    
    def visualize_learning_curves(self, save_path: str = None) -> None:
        """
        Visualize the learning curves (rewards and accuracy).
        
        Args:
            save_path: Optional path to save the figure
        """
        if not self.episode_rewards:
            logger.warning("No training data available for visualization")
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot raw data
        x = range(1, len(self.episode_rewards) + 1)
        ax1.plot(x, self.episode_rewards, 'b-', alpha=0.3, label='Episode Reward')
        ax2.plot(x, self.episode_accuracies, 'g-', alpha=0.3, label='Accuracy')
        
        # Plot smoothed data
        window_size = min(100, len(self.episode_rewards) // 10)
        if window_size > 1:
            smoothed_rewards = np.convolve(self.episode_rewards, 
                                         np.ones(window_size)/window_size, 
                                         mode='valid')
            smoothed_acc = np.convolve(self.episode_accuracies,
                                     np.ones(window_size)/window_size,
                                     mode='valid')
            
            # Plot smoothed data
            smooth_x = range(window_size, len(self.episode_rewards) + 1)
            ax1.plot(smooth_x, smoothed_rewards, 'b-', linewidth=2, label=f'Smoothed (window={window_size})')
            ax2.plot(smooth_x, smoothed_acc, 'g-', linewidth=2, label=f'Smoothed (window={window_size})')
        
        # Add labels and legends
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Rewards')
        ax1.legend()
        ax1.grid(True)
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Sorting Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Learning curves saved to {save_path}")
        
        plt.close()
    
    def visualize_confusion_matrix(self, normalize: bool = True, save_path: str = None) -> None:
        """
        Visualize the confusion matrix of sorting predictions.
        
        Args:
            normalize: Whether to normalize the matrix
            save_path: Optional path to save the figure
        """
        # Clone the matrix to avoid modifying the original
        cm = self.train_confusion_matrix.copy()
        
        if normalize:
            # Normalize by row (true categories)
            row_sums = cm.sum(axis=1, keepdims=True)
            if (row_sums != 0).all():  # Avoid division by zero
                cm = cm / row_sums
        
        # Plot
        plt.figure(figsize=(12, 10))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        # Add category labels
        tick_marks = np.arange(len(WASTE_CATEGORIES))
        plt.xticks(tick_marks, WASTE_CATEGORIES, rotation=45, ha='right')
        plt.yticks(tick_marks, WASTE_CATEGORIES)
        
        # Add text annotations
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True Category')
        plt.xlabel('Predicted Category')
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.close()
    
    def run_demo(self, num_items: int = 20, difficulty: str = 'hard',
                render: bool = True) -> None:
        """
        Run a demonstration of the sorting system.
        
        Args:
            num_items: Number of items to sort
            difficulty: Difficulty level for the demo
            render: Whether to print detailed results to console
        """
        # Create a demo environment
        demo_env = SortingEnv(difficulty=difficulty, batch_size=num_items)
        state = demo_env.reset()
        
        # Create arrays to store results
        results = []
        
        # Sort each item
        done = False
        while not done:
            # Choose action (no exploration)
            action = self.agent.act(state, training=False)
            
            # Take action
            next_state, reward, done, info = demo_env.step(action)
            
            # Store result
            item = demo_env.current_batch[demo_env.current_item_idx - 1]
            results.append({
                'true_category': item.true_category,
                'predicted_category': info['predicted_category'],
                'correct': info['correct'],
                'reward': reward,
                'properties': {p: item.properties.get(p) for p in MATERIAL_PROPERTIES}
            })
            
            # Update state
            state = next_state
        
        # Display results
        if render:
            print("\n=== Waste Sorting Demo Results ===")
            print(f"Difficulty: {difficulty}")
            print(f"Items sorted: {num_items}")
            print(f"Correct sorts: {demo_env.correct_sorts} ({demo_env.correct_sorts/num_items*100:.1f}%)")
            print(f"Total reward: {demo_env.episode_rewards:.2f}")
            print("\nDetailed results:")
            
            for i, res in enumerate(results):
                print(f"\nItem {i+1}:")
                print(f"  True category: {res['true_category']}")
                print(f"  Predicted: {res['predicted_category']}")
                print(f"  Correct: {'✓' if res['correct'] else '✗'}")
                print(f"  Reward: {res['reward']:.1f}")
                print("  Key properties:")
                props = res['properties']
                # Print most distinctive properties
                print(f"    Weight: {props['weight']:.1f}g")
                print(f"    Metal content: {props['metal_content']:.2f}")
                print(f"    Transparency: {props['transparency']:.2f}")
                print(f"    Moisture: {props['moisture']:.2f}")
        
        return results


def run_simulation(num_episodes: int = 1000, model_name: str = None, mode: str = 'train'):
    """
    Main function to run the sorting simulation.
    
    Args:
        num_episodes: Number of training episodes
        model_name: Optional model name to load
        mode: 'train', 'evaluate', or 'demo'
    """
    # Create simulation
    sim = SortingSimulation()
    
    # Load model if specified
    if model_name:
        sim.load_model(model_name)
    
    # Run in specified mode
    if mode == 'train':
        # Define difficulty schedule - gradually increase difficulty
        difficulty_schedule = {
            1: 'easy',
            int(num_episodes * 0.3): 'medium',
            int(num_episodes * 0.7): 'hard'
        }
        
        # Train the agent
        sim.train(
            episodes=num_episodes,
            target_update_freq=10,
            evaluate_freq=100,
            save_freq=200,
            difficulty_schedule=difficulty_schedule
        )
        
    elif mode == 'evaluate':
        # Evaluate on different difficulties
        for difficulty in ['easy', 'medium', 'hard']:
            reward, accuracy = sim.evaluate(episodes=10, difficulty=difficulty)
            print(f"Difficulty: {difficulty}")
            print(f"  Avg Reward: {reward:.2f}")
            print(f"  Accuracy: {accuracy:.4f}")
    
    elif mode == 'demo':
        # Run a demo
        sim.run_demo(num_items=20, difficulty='medium')
    
    else:
        logger.error(f"Unknown mode: {mode}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Waste Sorting Simulation')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'demo'],
                      help='Mode to run the simulation in')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--model', type=str, help='Model to load (optional)')
    
    args = parser.parse_args()
    
    run_simulation(num_episodes=args.episodes, model_name=args.model, mode=args.mode)

"""
SUMMARY:
========
This module implements a reinforcement learning simulation for training robotic waste sorting 
systems. The simulation creates a virtual environment where an AI agent learns to identify and 
sort different types of waste based on their physical properties.

Key components:
1. WasteItem - Represents waste objects with physical properties like weight, color, etc.
2. SortingEnv - A gym environment simulating a conveyor belt with waste items to sort
3. DQNAgent - Deep Q-Network agent that learns optimal sorting strategies
4. SortingSimulation - Manages the training, evaluation, and visualization of results

The system supports different difficulty levels and can generate visualizations of 
learning progress and confusion matrices to analyze sorting accuracy.

TODO:
=====
1. Add support for multi-sensor fusion (camera, spectrometer, weight sensor)
2. Implement more sophisticated RL algorithms (A2C, PPO) for comparison
3. Create a more realistic physics simulation for oddly shaped items
4. Add transfer learning capability to apply knowledge to new waste types
5. Implement a curriculum learning approach with better difficulty progression
6. Create a web-based visualization dashboard for monitoring training
7. Add simulation of sensor failures and recovery strategies
8. Integrate real waste sorting data to improve realism
9. Implement batch processing simulation instead of single item sorting
10. Add support for distributed training to speed up learning
11. Add an anomaly detection module for identifying unusual waste items
12. Implement a more sophisticated reward function based on economic value
"""
