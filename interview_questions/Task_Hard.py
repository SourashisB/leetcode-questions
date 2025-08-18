"""
FinCatch Hard Task Implementation
=================================

Author: Sourashis Bhowmik
Date:   Aug 18, 2025

Components
----------
1. Market Analyzer (Q1)
   - Simulates a Large Language Model (LLM) with an MLP + rule-based override.
   - Maps indicator inputs (1 to 100) and (1111 to 9999).

2. Agent Policy (Q2)
   - Policy network that makes trading decisions (actions ∈ [1, 1000]).
   - Reward function based on digit-matching rules.

3. Market Environment (Q3)
   - Simulated environment that returns next states and delayed rewards.
   - Sparse rewards mimic the difficulty of financial RL settings.

Talked about in Discussion.txt 

4. Deep Q-Learning (DQN)
   - Standard RL baseline to train the agent with replay buffer and target net.

5. Curriculum Learning Extension
   - Progressive training stages (easy → hard).
   - Stage 1: reward for any digit match.
   - Stage 2: reward only for ≥2 matches.
   - Stage 3: original sparse reward environment.
   - Demonstrates advanced training techniques for sparse-reward RL.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# =========================================================
# Q1: Market Analyzer (MLP + rule override)
# =========================================================
class MarketAnalyzer(nn.Module):
    """
    MarketAnalyzer: Simple MLP to approximate an LLM's market analysis.

    - Input: integer (1–100), representing a market indicator
    - Output: integer (1–10000), representing analysis state

    Special Case:
      - Inputs 1–9 map to "1111–9999" directly (rule-based).
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

def market_analysis(input_value, model=None):
    """Wrapper that applies special-case overrides for inputs 1–9."""
    if 1 <= input_value <= 9:
        return int(str(input_value) * 4)  # e.g. 1→1111
    else:
        if model is not None:
            inp = torch.tensor([[input_value]], dtype=torch.float32)
            output = model(inp).item()
            return int(max(1, min(10000, round(output))))
        else:
            return random.randint(10, 10000)

# =========================================================
# Q2: Agent Policy + Reward
# =========================================================
class AgentPolicy(nn.Module):
    """
    AgentPolicy: Policy network that maps LLM state → action ∈ [1, 1000].
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.net(state)

def compute_reward(llm_output, action):
    """
    Reward function:
      - 1 digit match: 10 points
      - 2 digit matches: 20 points
      - 3 digit matches: 100 points
      - else: 0
    """
    digits = set(str(llm_output))
    action_digits = str(action)
    matches = sum(d in action_digits for d in digits)
    if matches == 1:
        return 10
    elif matches == 2:
        return 20
    elif matches == 3:
        return 100
    else:
        return 0

# =========================================================
# Q3: Market Environment
# =========================================================
class MarketEnv:
    """
    MarketEnv: Simulates market interaction.

    Mechanics:
      - State = 1111, 2222, 3333, etc. depending on digit matches.
      - Step: agent picks action → environment counts digit matches.
      - Terminal condition:
          * State stabilizes (matches previous state), OR
          * 10 steps reached.
      - Reward:
          * If final state == 3: 100 / steps
          * Else: 0
    """

    def __init__(self):
        self.state = None
        self.prev_state = None
        self.steps = 0

    def reset(self, llm_output=1111):
        self.state = llm_output
        self.prev_state = None
        self.steps = 0
        return self.state

    def step(self, action):
        self.steps += 1
        matches = sum(d in str(action) for d in str(self.state))
        new_state = int(str(matches) * 4)

        done = False
        reward = 0

        # Terminal condition
        if self.prev_state == matches or self.steps >= 10:
            done = True
            if matches == 3:
                reward = 100 / self.steps
            else:
                reward = 0

        self.prev_state = matches
        self.state = new_state
        return new_state, reward, done

# =========================================================
# DQN components
# =========================================================
class QNetwork(nn.Module):
    """Q-Network: Approximates Q(s,a) for DQN."""
    def __init__(self, state_dim=1, action_dim=1000, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    """Cyclic replay buffer for experience replay in DQN."""
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return (
            torch.tensor(state, dtype=torch.float32).unsqueeze(1),
            torch.tensor(action, dtype=torch.long),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(next_state, dtype=torch.float32).unsqueeze(1),
            torch.tensor(done, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)

# =========================================================
# Curriculum Environment (reward shaping stages)
# =========================================================
class CurriculumMarketEnv(MarketEnv):
    """
    CurriculumMarketEnv: Extension of MarketEnv with staged difficulty.

    Stage 1: reward for ANY digit matches.
    Stage 2: reward only for ≥2 matches.
    Stage 3: original sparse-reward MarketEnv.
    """
    def __init__(self, stage=1):
        super().__init__()
        self.stage = stage

    def step(self, action):
        self.steps += 1
        matches = sum(d in str(action) for d in str(self.state))
        new_state = int(str(matches) * 4)

        done = False
        reward = 0

        # Stage-specific reward shaping
        if self.stage == 1:
            reward = matches * 10
        elif self.stage == 2:
            if matches >= 2:
                reward = matches * 10
        elif self.stage == 3:
            if self.prev_state == matches or self.steps >= 10:
                done = True
                if matches == 3:
                    reward = 100 / self.steps
                else:
                    reward = 0

        if self.steps >= 20:
            done = True

        self.prev_state = matches
        self.state = new_state
        return new_state, reward, done

# =========================================================
# Training: DQN
# =========================================================
def train_dqn(episodes=200, gamma=0.99, batch_size=64, lr=1e-3):
    """
    Train agent using vanilla DQN on MarketEnv.
    Demonstrates challenges with sparse rewards.
    """
    env = MarketEnv()
    qnet = QNetwork()
    target_qnet = QNetwork()
    target_qnet.load_state_dict(qnet.state_dict())
    optimizer = optim.Adam(qnet.parameters(), lr=lr)
    buffer = ReplayBuffer()

    epsilon, epsilon_min, epsilon_decay = 1.0, 0.05, 0.995
    update_target_every = 20
    rewards_history = []

    for ep in range(episodes):
        state = env.reset(1111)
        total_reward = 0

        for t in range(20):
            # Epsilon-greedy exploration
            if random.random() < epsilon:
                action = random.randint(1, 1000)
            else:
                with torch.no_grad():
                    q_values = qnet(torch.tensor([[state]], dtype=torch.float32))
                    action = q_values.argmax().item() + 1

            next_state, reward, done = env.step(action)
            buffer.push(state, action-1, reward, next_state, done)
            total_reward += reward
            state = next_state

            # Training step
            if len(buffer) >= batch_size:
                s, a, r, s2, d = buffer.sample(batch_size)
                q_values = qnet(s).gather(1, a.unsqueeze(1)).squeeze()
                with torch.no_grad():
                    max_next_q = target_qnet(s2).max(1)[0]
                    target = r + gamma * max_next_q * (1 - d)
                loss = nn.MSELoss()(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        rewards_history.append(total_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if ep % update_target_every == 0:
            target_qnet.load_state_dict(qnet.state_dict())

        if ep % 20 == 0:
            print(f"[DQN] Episode {ep}, Reward={total_reward:.2f}, Eps={epsilon:.2f}")

    return qnet, rewards_history

# =========================================================
# Training: Curriculum DQN
# =========================================================
def train_with_curriculum(episodes=600):
    """
    Train agent using Curriculum Learning.
    Progresses through stages as performance improves.
    """
    stage = 1
    env = CurriculumMarketEnv(stage=stage)
    qnet = QNetwork()
    target_qnet = QNetwork()
    target_qnet.load_state_dict(qnet.state_dict())
    optimizer = optim.Adam(qnet.parameters(), lr=1e-3)
    buffer = ReplayBuffer()

    epsilon, epsilon_min, epsilon_decay = 1.0, 0.05, 0.995
    update_target_every = 20
    rewards_history = []

    for ep in range(episodes):
        state = env.reset(1111)
        total_reward = 0

        for t in range(20):
            if random.random() < epsilon:
                action = random.randint(1, 1000)
            else:
                with torch.no_grad():
                    q_values = qnet(torch.tensor([[state]], dtype=torch.float32))
                    action = q_values.argmax().item() + 1

            next_state, reward, done = env.step(action)
            buffer.push(state, action-1, reward, next_state, done)
            total_reward += reward
            state = next_state

            if len(buffer) >= 64:
                s, a, r, s2, d = buffer.sample(64)
                q_values = qnet(s).gather(1, a.unsqueeze(1)).squeeze()
                with torch.no_grad():
                    max_next_q = target_qnet(s2).max(1)[0]
                    target = r + 0.99 * max_next_q * (1 - d)
                loss = nn.MSELoss()(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        rewards_history.append(total_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if ep % update_target_every == 0:
            target_qnet.load_state_dict(qnet.state_dict())

        # Stage progression
        if ep % 100 == 0 and ep > 0:
            avg_reward = np.mean(rewards_history[-100:])
            if avg_reward > 50 and stage < 3:
                stage += 1
                env.stage = stage
                print(f"🔼 Progressing to Stage {stage} at Episode {ep}")

        if ep % 50 == 0:
            print(f"[Curriculum] Ep {ep}, Stage {stage}, Reward={total_reward:.2f}, Eps={epsilon:.2f}")

    return qnet, rewards_history

# =========================================================
# Entry point
# =========================================================
if __name__ == "__main__":
    print("=== Training with DQN ===")
    trained_qnet, rewards = train_dqn(episodes=200)

    print("\n=== Training with Curriculum DQN ===")
    trained_curr_qnet, rewards_curr = train_with_curriculum(episodes=600)