from data_loader import load_data
from model import MatrixFactorization
import numpy as np
import random
import math

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
    random.shuffle(data)
    split = int(0.8 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    epochs = 20
    for epoch in range(epochs):
        mse = model.train_one_epoch(train_data)
        print(f"Epoch {epoch +1}/{epochs} - MSE: {mse:.4f}")
        rmse = np.sqrt(mse)
        print(f"Epoch {epoch+1}/{epochs} - RMSE: {rmse:.4f}")

    val_error = 0.0
    
    baseline_error = 0.0
    for _,_,r in val_data:
        baseline_error =+ (r-mu)**2
    baseline_rmse = np.sqrt(baseline_error/len(val_data))
    print(f"Baseline RMSE (global mean): {baseline_rmse:.4f}")

    for u,i,r in val_data:
        rating_val = model.predict(u, i)
        err = r - rating_val
        val_error += err ** 2 

    val_mse = val_error/len(val_data)
    val_rmse = np.sqrt(val_mse)
    print(f"\nValidaiton RMSE: {val_rmse:.4f}")

if __name__ == "__main__":
    train()