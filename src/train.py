from data_loader import load_data
from model import MatrixFactorization
import numpy as np

def train():
    data, num_users, num_items = load_data("data/u.data")

    ratings = [r for (_, _, r) in data]
    mu = np.mean(ratings)

    model = MatrixFactorization(
        num_users = num_users,
        num_items=num_items,
        num_factors=20,
        lr = 0.01,
        reg = 0.1
    )

    model.mu = mu

    epochs = 20
    for epoch in range(epochs):
        mse = model.train_one_epoch(data)
        print(f"Epoch {epoch +1}/{epochs} - MSE: {mse:.4f}")

if __name__ == "__main__":
    train()