import torch
import torch.nn as nn
import torch.optim as optim
import random

# MLP Definition
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 32),   # input layer
            nn.ReLU(),
            nn.Linear(32, 64),  # hidden layer
            nn.ReLU(),
            nn.Linear(64, 1)    # output layer
        )

    def forward(self, x):
        return self.layers(x)


# One stop training data generation function
def generate_data(n_samples=1000):
    X, y = [], []
    for _ in range(n_samples):
        inp = random.randint(1, 100)
        if inp in range(1, 10):  # special mapping
            out = int(str(inp) * 4)  # 1 -> 1111, 2 -> 2222 according to brief
        else:
            out = random.randint(1, 10000)
        X.append([inp])
        y.append([out])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Train the network
def train_model(model, epochs=500, lr=0.001):
    X_train, y_train = generate_data()
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

# Prediction function with rule enforcement
def predict(model, inp):
    if 1 <= inp <= 9:
        return int(str(inp) * 4)  # enforce special rule
    x = torch.tensor([[inp]], dtype=torch.float32)
    output = model(x).item()
    # Clamp into valid range
    return max(1, min(int(output), 10000))

# Run everything
if __name__ == "__main__":
    model = MLP()
    train_model(model)

    # Test predictions
    for test_inp in [1, 5, 9, 10, 50, 100]:
        print(f"Input: {test_inp} → Output: {predict(model, test_inp)}")