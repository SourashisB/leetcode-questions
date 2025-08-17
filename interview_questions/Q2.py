import torch
import torch.nn as nn
import torch.optim as optim
import random

# -----------------------------
# Policy Network Definition
# -----------------------------
class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 32),   # input: LLM output (scalar)
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1)    # output: action α (scalar)
        )

    def forward(self, x):
        return self.layers(x)


# -----------------------------
# Reward Function
# -----------------------------
def compute_reward(state, action):
    """
    state: LLM output (int, e.g. 1111, 2222, ...)
    action: agent's action (int between 1 and 1000)
    """
    state_str = str(state)
    # Only apply reward if state is XXXX type (all digits same)
    if len(set(state_str)) == 1 and len(state_str) == 4:
        target_digit = state_str[0]
        action_str = str(action)
        count = action_str.count(target_digit)
        if count == 1:
            return 10
        elif count == 2:
            return 20
        elif count == 3:
            return 100
    return 0


# -----------------------------
# Training Data Generator
# -----------------------------
def generate_data(n_samples=500):
    X, y = [], []
    for _ in range(n_samples):
        # Random LLM state: either XXXX form or random number
        if random.random() < 0.5:
            d = str(random.randint(1, 9))
            state = int(d * 4)
        else:
            state = random.randint(1, 10000)

        # Random action α ∈ [1,1000]
        action = random.randint(1, 1000)

        # Reward
        reward = compute_reward(state, action)

        X.append([state])
        y.append([reward])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# -----------------------------
# Train Policy Network
# -----------------------------
def train_policy(model, epochs=500, lr=0.001):
    X_train, y_train = generate_data(1000)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")


# -----------------------------
# Agent Decision Function
# -----------------------------
def decide_action(model, state):
    x = torch.tensor([[state]], dtype=torch.float32)
    action = int(model(x).item())
    action = max(1, min(action, 1000))  # clamp into valid range
    return action


# -----------------------------
# Example Run (Standalone)
# -----------------------------
if __name__ == "__main__":
    # Train the policy network
    model = PolicyNetwork()
    train_policy(model)

    # Example Q2 standalone test
    test_states = [1111, 2222, 3333, 5678]
    for s in test_states:
        action = decide_action(model, s)
        reward = compute_reward(s, action)
        print(f"State: {s} | Action: {action} | Reward: {reward}")

    # -----------------------------
    # Inference-time connection to Q1
    # -----------------------------
    try:
        from Q1 import MLP as MarketAnalyzer, predict as predict_state

        print("\nConnecting to Q1 outputs at inference:")
        q1_model = MarketAnalyzer()
        # (Assume Q1 is already trained, otherwise call train_model(q1_model))

        for indicator in [1, 5, 10, 50, 100]:
            state = predict_state(q1_model, indicator)   # Q1 generates state
            action = decide_action(model, state)         # Q2 decides action
            reward = compute_reward(state, action)       # Evaluate reward
            print(f"Indicator: {indicator} → State: {state} → Action: {action} → Reward: {reward}")

    except ImportError:
        print("\nQ1 module not found. Only running Q2 standalone.")