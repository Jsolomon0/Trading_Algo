import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# ============================================
# 1. LSTM-based Actor-Critic Network
# ============================================

class LSTMActorCritic(nn.Module):
    """
    LSTM-based Actor-Critic Network for Deep Reinforcement Learning.
    
    Architecture:
    - LSTM layers for temporal feature extraction
    - Actor head: outputs action probabilities (Buy, Sell, Hold)
    - Critic head: outputs state value estimate
    """
    
    def __init__(self, input_size=6, hidden_size=128, num_lstm_layers=2, num_actions=3):
        super(LSTMActorCritic, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        
        # LSTM backbone for temporal pattern recognition
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=0.2 if num_lstm_layers > 1 else 0
        )
        
        # Actor head (policy network) - outputs action probabilities
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_actions),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value network) - outputs state value
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
    def forward(self, x, hidden=None):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            hidden: Optional hidden state tuple (h0, c0)
            
        Returns:
            action_probs: Action probability distribution
            state_value: Estimated value of the current state
            hidden: Updated hidden state
        """
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Use the last time step's output
        last_output = lstm_out[:, -1, :]
        
        # Get action probabilities and state value
        action_probs = self.actor(last_output)
        state_value = self.critic(last_output)
        
        return action_probs, state_value, hidden
    
    def init_hidden(self, batch_size=1, device='cpu'):
        """Initialize hidden state for LSTM."""
        h0 = torch.zeros(self.num_lstm_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_lstm_layers, batch_size, self.hidden_size).to(device)
        return (h0, c0)


# ============================================
# 2. Experience Replay Buffer
# ============================================

class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions."""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


# ============================================
# 3. Deep RL Trading Agent (PPO-style)
# ============================================

class DRLTradingAgent:
    """
    Deep Reinforcement Learning Trading Agent using LSTM backbone.
    Uses Proximal Policy Optimization (PPO) algorithm.
    """
    
    def __init__(
        self,
        input_size=6,
        hidden_size=128,
        num_lstm_layers=2,
        sequence_length=30,
        learning_rate=0.0001,
        gamma=0.99,
        epsilon=0.2,
        device=None
    ):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sequence_length = sequence_length
        self.gamma = gamma
        self.epsilon = epsilon  # PPO clipping parameter
        
        # Initialize network
        self.network = LSTMActorCritic(
            input_size=input_size,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            num_actions=3  # Buy (0), Sell (1), Hold (2)
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Experience buffer
        self.replay_buffer = ReplayBuffer(capacity=10000)
        
        # Hidden state for online inference
        self.hidden = None
        
        # Action mapping
        self.action_map = {0: 'BUY', 1: 'SELL', 2: 'HOLD'}
        
        # Training metrics
        self.training_losses = []
        
    
    def select_action0(self, state_sequence, deterministic=False):
        """
        Select an action given the current state sequence.
        
        Args:
            state_sequence: Numpy array of shape (sequence_length, input_size)
            deterministic: If True, select the action with highest probability
            
        Returns:
            action: Selected action (0=Buy, 1=Sell, 2=Hold)
            action_name: Action name string
            action_prob: Probability of selected action
        """
        self.network.eval()
        
        with torch.no_grad():
            # 1. Convert to tensor
            if isinstance(state_sequence, np.ndarray):
                # Standard input from environment (2D: [sequence_length, input_size])
                state_tensor = torch.FloatTensor(state_sequence).unsqueeze(0).to(self.device)
                # Add the required batch dimension (3D: [batch_size, sequence_length, input_size])
                state_tensor = state_tensor.unsqueeze(0)
            else:
                # Input from testing (already 3D tensor: [batch_size, sequence_length, input_size])
                state_tensor = state_sequence.to(self.device)
            
            # 2. **CRITICAL FIX: Ensure 3D shape before model forward pass**
            # This removes any accidental extra batch dimensions (e.g., changing [1, 1, 30, 6] to [1, 30, 6])
            state_tensor = state_tensor.squeeze()
            if state_tensor.dim() == 2:
                # If squeezing reduced it to 2D ([30,6]), add the batch dimension back
                state_tensor = state_tensor.unsqueeze(0)
                           
            # --- ORIGINAL CODE ---
            # Convert to tensor and add batch dimension
            # state_tensor = torch.FloatTensor(state_sequence).unsqueeze(0).to(self.device)
            
            # Get action probabilities
            action_probs, _, self.hidden = self.network(state_tensor, self.hidden)
            action_probs = action_probs.squeeze(0)
            
            if deterministic:
                action = torch.argmax(action_probs).item()
            else:
                # Sample from the probability distribution
                action = torch.multinomial(action_probs, 1).item()
            
            action_prob = action_probs[action].item()
        
        return action, self.action_map[action], action_prob
    
    # Temperture scaling for better action selection
    def select_action(self, state_sequence, deterministic=False, temperature=1.0):

        """ Args:
          temperature: Higher = more random, Lower = more decisive,
            Use 0.5 for more confident decisions
            Use 1.0 for normal
            Use 2.0 for more exploration"""
        self.network.eval()
    
        with torch.no_grad():
            state_tensor = self._prepare_input(state_sequence)
            action_probs, _, self.hidden = self.network(state_tensor, self.hidden)
            action_probs = action_probs.squeeze(0)
        
        # Apply temperature scaling
            if temperature != 1.0:
                action_probs = torch.pow(action_probs, 1.0 / temperature)
                action_probs = action_probs / action_probs.sum()
        
            if deterministic:
                action = action_probs.argmax().item()
            else:
                action = torch.multinomial(action_probs, 1).item()
        
        return action, self.action_map[action], action_probs[action].item()
    
    def reset_hidden_state(self):
        """Reset the LSTM hidden state (call at the start of each episode)."""
        self.hidden = None
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def compute_returns(self, rewards, dones, next_values):
        """Compute discounted returns with GAE (Generalized Advantage Estimation)."""
        returns = []
        advantages = []
        
        gae = 0
        for i in reversed(range(len(rewards))):
            if dones[i]:
                delta = rewards[i] - 0  # Terminal state has no next value
                gae = delta
            else:
                delta = rewards[i] + self.gamma * next_values[i] - 0  # Simplified
                gae = delta + self.gamma * 0.95 * gae  # Lambda = 0.95 for GAE
            
            returns.insert(0, gae + 0)  # Simplified return calculation
            advantages.insert(0, gae)
        
        return returns, advantages
    
    def train_step(self, batch_size=32, epochs=4):
        """
        Train the network using PPO algorithm.
        
        Args:
            batch_size: Number of samples per training batch
            epochs: Number of epochs to train on each batch
            
        Returns:
            loss: Average training loss
        """
        if len(self.replay_buffer) < batch_size:
            return None
        
        self.network.train()
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        
        total_loss = 0
        
        for _ in range(epochs):
            # Forward pass
            action_probs, state_values, _ = self.network(states_tensor)
            state_values = state_values.squeeze(-1)
            
            # Calculate advantages (simplified version)
            advantages = rewards_tensor - state_values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Actor loss (policy loss with PPO clipping)
            action_log_probs = torch.log(action_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze(1) + 1e-8)
            
            # PPO clipped objective
            ratio = torch.exp(action_log_probs)  # Simplified - should use old_log_probs
            clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
            actor_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            
            # Critic loss (value loss)
            critic_loss = nn.MSELoss()(state_values, rewards_tensor)
            
            # Entropy bonus for exploration
            entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1).mean()
            
            # Total loss
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / epochs
        self.training_losses.append(avg_loss)
        
        return avg_loss
    
    def save_model(self, filepath):
        """Save model weights to file."""
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_losses': self.training_losses
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model weights from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_losses = checkpoint.get('training_losses', [])
        print(f"Model loaded from {filepath}")


# ============================================
# 4. Trading Environment (Reward Calculator)
# ============================================

class TradingEnvironment:
    """
    Trading environment that calculates rewards based on actions.
    """
    
    def __init__(self, initial_balance=10000.0, position_size=1.0, transaction_cost=0.001):
        self.initial_balance = initial_balance
        self.position_size = position_size
        self.transaction_cost = transaction_cost
        
        self.reset()
    
    def reset(self):
        """Reset the environment to initial state."""
        self.balance = self.initial_balance
        self.position = 0  # 0: No position, 1: Long, -1: Short
        self.entry_price = 0
        self.portfolio_value = self.initial_balance
        self.trades = []
        

    def calculate_reward0(self, action, current_price, prev_price):
        reward = 0
        done = False
        
        price_change = (current_price - prev_price) / prev_price
        
        # BUY action (0)
        if action == 0:
            if self.position == 0:  # Enter long
                self.position = 1
                self.entry_price = current_price
                reward = -self.transaction_cost * 0.1  # Reduce transaction cost penalty
            elif self.position == 1:  # Already long - reward holding
                reward = price_change * 100  # AMPLIFY reward for price increase
            elif self.position == -1:  # Close short, enter long
                pnl = (self.entry_price - current_price) / self.entry_price
                reward = pnl * 100 - self.transaction_cost * 2
                self.position = 1
                self.entry_price = current_price
        
        # SELL action (1)
        elif action == 1:
            if self.position == 0:  # Enter short
                self.position = -1
                self.entry_price = current_price
                reward = -self.transaction_cost * 0.1
            elif self.position == -1:  # Already short - reward holding
                reward = -price_change * 100  # AMPLIFY reward for price decrease
            elif self.position == 1:  # Close long, enter short
                pnl = (current_price - self.entry_price) / self.entry_price
                reward = pnl * 100 - self.transaction_cost * 2
                self.position = -1
                self.entry_price = current_price
        
        # HOLD action (2)
        elif action == 2:
            if self.position == 1:  # Holding long
                reward = price_change * 50  # Reward for price increase
            elif self.position == -1:  # Holding short
                reward = -price_change * 50  # Reward for price decrease
            else:  # No position - PENALIZE inaction
                reward = -0.1  # Significant penalty for doing nothing
        
        # Update portfolio value
        if self.position == 1:
            self.portfolio_value = self.balance + (current_price - self.entry_price) / self.entry_price * self.balance
        elif self.position == -1:
            self.portfolio_value = self.balance + (self.entry_price - current_price) / self.entry_price * self.balance
        else:
            self.portfolio_value = self.balance
        
        # Check if should end episode
        if self.portfolio_value < self.initial_balance * 0.5:
            done = True
            reward -= 100  # Large penalty for losing too much
        
        return reward, done

    #Fix Reward function
    def calculate_reward(self, action, current_price, prev_price):
        """
        Calculate reward based on action and price movement.
        
        Args:
            action: 0=Buy, 1=Sell, 2=Hold
            current_price: Current asset price
            prev_price: Previous asset price
            
        Returns:
            reward: Calculated reward value
            done: Whether episode is done
        """
        reward = 0
        done = False
        
        price_change = (current_price - prev_price) / prev_price
        
        # BUY action
        if action == 0:
            if self.position == 0:  # Enter long position
                self.position = 1
                self.entry_price = current_price
                reward = -self.transaction_cost  # Transaction cost
            elif self.position == 1:  # Already long
                reward = price_change  # Reward for holding profitable position
            elif self.position == -1:  # Close short and enter long
                pnl = (self.entry_price - current_price) / self.entry_price
                reward = pnl - 2 * self.transaction_cost
                self.position = 1
                self.entry_price = current_price
        
        # SELL action
        elif action == 1:
            if self.position == 0:  # Enter short position
                self.position = -1
                self.entry_price = current_price
                reward = -self.transaction_cost
            elif self.position == -1:  # Already short
                reward = -price_change  # Reward for profitable short
            elif self.position == 1:  # Close long and enter short
                pnl = (current_price - self.entry_price) / self.entry_price
                reward = pnl - 2 * self.transaction_cost
                self.position = -1
                self.entry_price = current_price
        
        # HOLD action
        elif action == 2:
            if self.position == 1:  # Holding long
                reward = price_change * 0.5  # Partial reward for holding
            elif self.position == -1:  # Holding short
                reward = -price_change * 0.5
            else:  # No position
                reward = -0.0001  # Small penalty for inaction
        
        # Update portfolio value
        if self.position == 1:
            self.portfolio_value = self.balance + (current_price - self.entry_price) / self.entry_price * self.balance
        elif self.position == -1:
            self.portfolio_value = self.balance + (self.entry_price - current_price) / self.entry_price * self.balance
        else:
            self.portfolio_value = self.balance
        
        # Check if episode should end (e.g., excessive loss)
        if self.portfolio_value < self.initial_balance * 0.5:
            done = True
            reward -= 10  # Large penalty for losing too much
        
        return reward, done
    

# ============================================
# 5. Usage Example
# ============================================

if __name__ == "__main__":
    # Initialize agent
    agent = DRLTradingAgent(
        input_size=6,  # Number of Ichimoku features
        hidden_size=128,
        num_lstm_layers=2,
        sequence_length=30,  # Use 30 time steps of history
        learning_rate=0.0001,
        gamma=0.99
    )
    
    # Initialize environment
    env = TradingEnvironment(initial_balance=10000.0)
    
    print("DRL Trading Agent initialized successfully!")
    print(f"Device: {agent.device}")
    print(f"Network parameters: {sum(p.numel() for p in agent.network.parameters()):,}")
    
    # Example: Simulated feature sequence (30 timesteps x 6 features)
    dummy_features = np.random.randn(30, 6)
    
    # Select action
    action, action_name, action_prob = agent.select_action(dummy_features)
    print(f"\nExample action: {action_name} (probability: {action_prob:.3f})")
    
    # Example training loop structure
    print("\n--- Training Loop Example ---")
    print("for episode in range(num_episodes):")
    print("    env.reset()")
    print("    agent.reset_hidden_state()")
    print("    ")
    print("    for step in range(max_steps):")
    print("        action, _, _ = agent.select_action(state_sequence)")
    print("        reward, done = env.calculate_reward(action, current_price, prev_price)")
    print("        agent.store_transition(state, action, reward, next_state, done)")
    print("        ")
    print("        if done:")
    print("            break")
    print("    ")
    print("    # Train after each episode")
    print("    loss = agent.train_step(batch_size=32, epochs=4)")