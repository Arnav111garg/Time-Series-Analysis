import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import warnings; warnings.filterwarnings('ignore')

#
# TRADING ENVIRONMENT
#

class TradingEnvironment:
    """
    A single-asset trading environment with continuous state space.

    State: [ret_1d, ret_5d, ret_21d, rv_5d, rv_21d, vol_ratio,
            mom_1m, mom_3m, rsi, bb_pos, position, days_held]
    Action: {0: short, 1: flat, 2: long} (DQN) or in [-1, 1] (PPO)
    Reward: daily P&L - transaction cost * |position_change|
    """
    def __init__(self, returns, tc=0.0010, initial_wealth=1.0, mode='discrete', lookback=60):
        """
        returns: pd.Series of daily log returns (as fractions)
        tc: transaction cost per unit position change
        mode: 'discrete' for DQN, 'continuous' for PPO
        """
        self.returns = np.array(returns)
        self.tc = tc
        self.W0 = initial_wealth
        self.mode = mode
        self.lookback = lookback
        self.T = len(returns)

        # Pre-compute features
        self._precompute_features()
        self.reset()

    def _precompute_features(self):
        """Pre-compute all state features to avoid redundant calculation."""
        r = pd.Series(self.returns)
        features = pd.DataFrame()
        features['ret_1d'] = r
        features['ret_5d'] = r.rolling(5).mean()
        features['ret_21d'] = r.rolling(21).mean()
        features['rv_5d'] = r.rolling(5).std() + 1e-10
        features['rv_21d'] = r.rolling(21).std() + 1e-10
        features['vol_ratio'] = (r.rolling(5).std() + 1e-10) / (r.rolling(21).std() + 1e-10)
        features['mom_1m'] = r.rolling(21).sum()
        features['mom_3m'] = r.rolling(63).sum()

        # RSI
        delta = r.diff()
        gain = delta.clip(lower=0).ewm(com=13).mean()
        loss = (-delta).clip(lower=0).ewm(com=13).mean()
        features['rsi'] = (100 - 100 / (1 + gain / (loss + 1e-10))) / 50 - 1
        features['bb_pos'] = (r - r.rolling(20).mean()) / (2 * r.rolling(20).std() + 1e-10)
        self.features = features.fillna(0).values

    def reset(self):
        """Reset environment to start."""
        self.t = self.lookback
        self.position = 0.0
        self.wealth = self.W0
        self.pnl_hist = []
        return self._get_state()

    def _get_state(self):
        """Return normalized feature vector + portfolio state."""
        feat = self.features[self.t].copy()
        state = np.append(feat, [self.position, min(self.t / 252, 1.0)])
        return state.astype(np.float32)

    def step(self, action):
        """
        Take action, advance time, return (next_state, reward, done, info).
        action: int {0,1,2} for discrete or float in [-1, 1] for continuous
        """
        if self.mode == 'discrete':
            target_pos = {0: -1.0, 1: 0.0, 2: 1.0}[int(action)]
        else:
            target_pos = float(np.clip(action, -1, 1))

        # Transaction cost
        pos_change = abs(target_pos - self.position)
        tc_paid = self.tc * pos_change

        # Execute trade
        self.position = target_pos

        # Advance time
        self.t += 1

        # P&L: return on position
        ret_t = self.returns[self.t]
        pnl = self.position * ret_t - tc_paid
        self.wealth *= (1 + pnl)
        self.pnl_hist.append(pnl)

        # Reward: can use P&L, log-wealth change, or Sharpe-adjusted
        reward = pnl

        done = (self.t >= self.T - 1)
        next_state = self._get_state() if not done else np.zeros(12, dtype=np.float32)
        info = {'wealth': self.wealth, 'pnl': pnl, 'position': self.position, 'ret': ret_t, 'tc': tc_paid}
        return next_state, reward, done, info

    @property
    def state_dim(self):
        return len(self._get_state())

    @property
    def action_dim(self):
        return 3 if self.mode == 'discrete' else 1


#
# DQN AGENT
#

class QNetwork(nn.Module):
    """Deep Q-Network for discrete trading actions."""
    def __init__(self, state_dim, action_dim=3, hidden=[128, 128, 64]):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    """
    DQN agent with experience replay and target network.
    Uses Double DQN to reduce overestimation bias.
    """
    def __init__(self, state_dim, action_dim=3,
                 lr=1e-3, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.05, epsilon_decay=0.998,
                 buffer_size=10000, batch_size=64,
                 target_update_freq=100):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.target_update = target_update_freq
        self.step_count = 0

        # Networks
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr, weight_decay=1e-5)
        self.replay = deque(maxlen=buffer_size)
        self.loss_fn = nn.HuberLoss() # robust to outliers

    def act(self, state, deterministic=False):
        """Epsilon-greedy action selection."""
        if not deterministic and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            q_vals = self.q_net(torch.FloatTensor(state).unsqueeze(0))
            return q_vals.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.replay.append((state, action, reward, next_state, done))

    def train_step(self):
        """Sample mini-batch and update Q-network (Double DQN)."""
        if len(self.replay) < self.batch_size:
            return 0.0

        batch = random.sample(self.replay, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))

        # Current Q-values
        q_curr = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Double DQN: use Q-net to select action, target-net to evaluate
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(1)
            q_next = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            q_target = rewards + self.gamma * q_next * (1 - dones)

        loss = self.loss_fn(q_curr, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

#
# TRAINING AND EVALUATION
#

print("Loading data and setting up environment...")
# Safely extract Adjusted Close to support different yfinance structures
data = yf.download("SPY", start="2000-01-01", end="2024-01-01")
if isinstance(data.columns, pd.MultiIndex):
    if 'Adj Close' in data.columns.get_level_values(0):
        spy = data['Adj Close'].iloc[:, 0]
    else:
        spy = data['Close'].iloc[:, 0]
else:
    if 'Adj Close' in data.columns:
        spy = data['Adj Close']
    else:
        spy = data['Close']

rets = np.log(spy / spy.shift(1)).dropna()

# Train / test split
train_rets = rets['2000-01-01':'2019-12-31'].values
test_rets = rets['2020-01-01':'2024-01-01'].values

train_env = TradingEnvironment(train_rets, tc=0.0010, mode='discrete')
test_env = TradingEnvironment(test_rets, tc=0.0010, mode='discrete')

agent = DQNAgent(state_dim=train_env.state_dim, action_dim=3,
                 lr=5e-4, epsilon=1.0, epsilon_min=0.05,
                 epsilon_decay=0.9995, buffer_size=20000,
                 batch_size=128)

# Training loop
print(f"\nTraining DQN agent on {len(train_rets)} days...")
n_episodes = 80
episode_rewards = []
episode_sharpes = []

for ep in range(n_episodes):
    state = train_env.reset()
    total_reward = 0
    losses = []

    while True:
        action = agent.act(state)
        next_state, reward, done, info = train_env.step(action)
        agent.remember(state, action, reward, next_state, done)
        loss = agent.train_step()
        if loss > 0:
            losses.append(loss)
        state = next_state
        total_reward += reward
        if done:
            break

    pnl_series = np.array(train_env.pnl_hist)
    ep_sharpe = (pnl_series.mean() * 252) / (pnl_series.std() * np.sqrt(252) + 1e-10)
    episode_rewards.append(total_reward)
    episode_sharpes.append(ep_sharpe)

    if (ep + 1) % 10 == 0:
        print(f"Ep {ep+1:3d}/{n_episodes}: "
              f"TotalReward={total_reward:.4f}, "
              f"Sharpe={ep_sharpe:.3f}, "
              f"Epsilon={agent.epsilon:.3f}, "
              f"Loss={np.mean(losses) if losses else 0:.6f}")

# Out-of-sample evaluation
print("\nOut-of-sample evaluation (2020-2024)...")
state = test_env.reset()
agent.epsilon = 0.0 # greedy policy

positions = []
oos_pnls = []
oos_actions = []

while True:
    action = agent.act(state, deterministic=True)
    next_state, reward, done, info = test_env.step(action)
    positions.append(info['position'])
    oos_pnls.append(info['pnl'])
    oos_actions.append(action)
    state = next_state
    if done:
        break

oos_pnls = np.array(oos_pnls)
bh_pnls = test_rets[1:] # buy-and-hold

def perf_stats(pnls, label):
    ann_ret = pnls.mean() * 252
    ann_vol = pnls.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    cum = (1 + pnls).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    print(f"\n{label}:")
    print(f" Ann. Return: {ann_ret * 100:+.2f}%")
    print(f" Ann. Vol: {ann_vol * 100:.2f}%")
    print(f" Sharpe: {sharpe:.3f}")
    print(f" Max Drawdown: {max_dd * 100:.2f}%")
    print(f" Calmar: {ann_ret / abs(max_dd) if max_dd != 0 else np.nan:.3f}")
    return pd.Series(cum)

print("\nOut-of-Sample Performance (2020-2024):")
cum_dqn = perf_stats(oos_pnls, "DQN Agent")
cum_bh = perf_stats(bh_pnls[:len(oos_pnls)], "Buy & Hold")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("DQN Trading Agent: Training and Evaluation", fontsize=13, fontweight='bold')

# 1. Training curves
axes[0, 0].plot(episode_rewards, 'navy', lw=1.2, label='Episode Reward')
axes[0, 0].plot(pd.Series(episode_rewards).rolling(10).mean(), 'red', lw=2, label='10-ep Rolling Mean')
axes[0, 0].set_title("Training: Episode Rewards")
axes[0, 0].set_xlabel("Episode")
axes[0, 0].legend()

axes[0, 1].plot(episode_sharpes, 'darkgreen', lw=1.2)
axes[0, 1].axhline(0, color='k', lw=0.5)
axes[0, 1].set_title("Training: Episode Sharpe Ratios")
axes[0, 1].set_xlabel("Episode")

# 2. OOS performance
axes[1, 0].plot(range(len(cum_dqn)), cum_dqn.values, 'darkgreen', lw=2, label='DQN Agent')
axes[1, 0].plot(range(len(cum_bh[:len(oos_pnls)])), cum_bh[:len(oos_pnls)].values, 'navy', lw=1.5, linestyle='--', label='Buy & Hold')
axes[1, 0].set_title("Out-of-Sample Cumulative Wealth (2020-2024)")
axes[1, 0].set_ylabel("Wealth")
axes[1, 0].legend()

# 3. Action distribution
action_labels = {0: 'Short (-1)', 1: 'Flat (0)', 2: 'Long (+1)'}
action_counts = [oos_actions.count(a) for a in range(3)]
colors_a = ['firebrick', 'gray', 'steelblue']
axes[1, 1].bar([action_labels[a] for a in range(3)], action_counts, color=colors_a, alpha=0.8, edgecolor='k')
for i, (val, col) in enumerate(zip(action_counts, colors_a)):
    axes[1, 1].text(i, val + 5, f'{val / len(oos_actions) * 100:.1f}%', ha='center', fontsize=10)
axes[1, 1].set_title("OOS Action Distribution")
axes[1, 1].set_ylabel("Count")

plt.tight_layout()
plt.savefig("dqn_trading_agent.pdf", dpi=150)
plt.show()

