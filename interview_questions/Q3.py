import random
import numpy as np

# -----------------------------
# Environment Definition
# -----------------------------
class MarketEnv:
    def __init__(self, max_steps=10):
        self.max_steps = max_steps
        self.reset()

    def reset(self, initial_state=1111):
        self.state = initial_state
        self.prev_state = None
        self.steps = 0
        return self.state

    def step(self, action):
        """
        Action: integer in [1,1000]
        Returns: next_state, done, reward
        """
        self.steps += 1
        state_str = str(self.state)
        digit = state_str[0]  # repeating digit from XXXX
        next_state = str(action).count(digit)  # count matches

        done = False
        reward = 0

        # If state repeats → episode ends
        if self.prev_state is not None and next_state == self.prev_state:
            done = True
            if next_state == 3:
                reward = 100 / self.steps
        elif self.steps >= self.max_steps:
            done = True
            reward = 0

        self.prev_state = next_state

        # LLM generates new XXXX state based on next_state
        if next_state in [1, 2, 3]:
            self.state = int(str(next_state) * 4)
        else:
            self.state = random.randint(1000, 9999)

        return self.state, reward, done


# -----------------------------
# Simple Q-learning Agent
# -----------------------------
class QLearningAgent:
    def __init__(self, state_space=10000, action_space=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}  # dict-based (s,a) → value
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_space = action_space

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(1, self.action_space)
        qs = [self.get_q(state, a) for a in range(1, self.action_space+1)]
        best_action = np.argmax(qs) + 1
        return best_action

    def update(self, state, action, reward, next_state):
        old_q = self.get_q(state, action)
        future_qs = [self.get_q(next_state, a) for a in range(1, self.action_space+1)]
        max_future_q = max(future_qs) if future_qs else 0
        new_q = old_q + self.alpha * (reward + self.gamma * max_future_q - old_q)
        self.q_table[(state, action)] = new_q


# -----------------------------
# Testing Specific Cases
# -----------------------------
def test_case_1():
    env = MarketEnv()
    state = env.reset(1111)
    # Interaction 1
    action = 111
    state, reward, done = env.step(action)
    # Interaction 2
    if not done:
        action = 333
        state, reward, done = env.step(action)
    return reward

def test_case_2():
    env = MarketEnv()
    state = env.reset(1111)
    # Interaction 1
    action = 199
    state, reward, done = env.step(action)
    # Interaction 2
    if not done:
        action = 213
        state, reward, done = env.step(action)
    # Interaction 3
    if not done:
        action = 111
        state, reward, done = env.step(action)
    # Interaction 4
    if not done:
        action = 333
        state, reward, done = env.step(action)
    return reward


# -----------------------------
# Main Run
# -----------------------------
if __name__ == "__main__":
    print("Test Case 1 Reward:", test_case_1())  # Expected 50
    print("Test Case 2 Reward:", test_case_2())  # Expected 25

    # Optional: Train Q-learning agent
    env = MarketEnv()
    agent = QLearningAgent()

    episodes = 100
    for ep in range(episodes):
        state = env.reset(1111)
        total_reward = 0
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        if ep % 10 == 0:
            print(f"Episode {ep}, Total Reward: {total_reward}")