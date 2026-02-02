from data_loader import load_data
from model import MatrixFactorization
import numpy as np
import random

random.seed(42)
np.random.seed(42)

def evaluate(model, val_data):
    error = 0.0
    for u,i,r in val_data:
        r_hat = model.safe_predict(u,i)
        error += (r-r_hat)**2
    mse = error/len(val_data)
    rmse = np.sqrt(mse)
    return rmse

def train():
    data, num_users, num_items = load_data("data/u.data")
    ratings = [r for (_, _, r) in data]
    mu = np.mean(ratings)

    random.shuffle(data)
    split = int(0.8 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    baseline_error = 0.0
    for _,_,r in val_data:
        baseline_error += (r-mu)**2
    baseline_rmse = np.sqrt(baseline_error/len(val_data))
    print(f"Baseline RMSE (global mean): {baseline_rmse:.4f}")

    factors_list = [10,20,40]
    lr_list = [0.005, 0.01]
    reg_list = [0.05, 0.1]

    results = []
    
    for f in factors_list:
        for lr in lr_list:
            for reg in reg_list:
                print(f"\nTraining with factors={f}, lr={lr}, reg={reg}")

                model = MatrixFactorization(
                    num_users = num_users,
                    num_items=num_items,
                    num_factors=20,
                    lr = 0.01,
                    reg = 0.1
                )
    
                model.mu = mu

                max_epochs = 80
                patience = 5

                best_val_rmse = float("inf")
                epochs_without_improvement = 0

                train_rmse_history = []
                val_rmse_history = []

                for epoch in range(max_epochs):
                    random.shuffle(train_data)
                    train_mse = model.train_one_epoch(train_data)
                    train_rmse = np.sqrt(train_mse)

                    val_rmse = evaluate(model, val_data)

                    train_rmse_history.append(train_rmse)
                    val_rmse_history.append(val_rmse)

                    print(
                        f"Epoch {epoch+1}"
                        f"Train RMSE: {train_rmse: .4f}"
                        f"Val RMSE: {val_rmse: .4f}"
                    )

                    if val_rmse< best_val_rmse:
                        best_val_rmse = val_rmse
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement+=1

                    if epochs_without_improvement >= patience:
                        print("Early stopping triggered")
                        break

                results.append({
                    "factors":f,
                    "lr": lr,
                    "reg": reg,
                    "val_rmse": val_rmse
                })

                print(f"Validation RMSE: {val_rmse: .4f}")
    
    best = min(results, key=lambda x:x["val_rmse"])
    
    print("\n")
    print("BEST HYPERPARAMETER SET:")
    print("\n")
    print(f"Factors: ,{best['factors']}")
    print(f"Learning rate: ,{best['lr']}")
    print(f"Regularization: ,{best['reg']}")
    print(f"Validation RMSE: ,{best['val_rmse']:.4f}")

if __name__ == "__main__":
    train()