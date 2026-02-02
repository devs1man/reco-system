# Recommendation System using Matrix Factorization (from Scratch)

This project implements a **collaborative filtering recommendation system from scratch** using **biased matrix factorization** and **stochastic gradient descent (SGD)**.  
The model is trained and evaluated on the **MovieLens 100K** dataset and tuned using proper **train/validation splitting**, **hyperparameter search**, and **early stopping**.

---

## Problem Statement

Given historical user–item ratings, predict how a user would rate unseen items.  
This is a classic **recommendation systems** problem commonly used in platforms like Netflix, Spotify, and Amazon.

---

## Dataset

- **MovieLens 100K**
- 100,000 ratings
- 943 users
- 1,682 movies
- Ratings on a scale of **1–5**

Each data point is a triplet: (user_id, movie_id, rating)

---

## Model

We use **biased matrix factorization**, where each rating is modeled as:

\[
\hat{r}\_{ui} = \mu + b_u + b_i + p_u^T q_i
\]

Where:

- \( \mu \) = global average rating
- \( b_u \) = user bias
- \( b_i \) = item bias
- \( p_u \) = latent user vector
- \( q_i \) = latent item vector

This allows the model to capture both:

- **systematic tendencies** (some users rate higher, some movies are popular)
- **personal preferences** (latent interactions)

---

## Training Method

- **Optimization**: Stochastic Gradient Descent (SGD)
- **Loss**: Mean Squared Error (MSE)
- **Regularization**: L2 regularization to prevent overfitting
- **Evaluation Metric**: Root Mean Squared Error (RMSE)

The model is trained on **80% of the data**, with **20% held out for validation**.

---

## Hyperparameter Tuning

A controlled grid search was performed over:

- Number of latent factors
- Learning rate
- Regularization strength

Best configuration found:

| Hyperparameter | Value |
| -------------- | ----- |
| Latent factors | 10    |
| Learning rate  | 0.01  |
| Regularization | 0.1   |

---

## Early Stopping

To avoid overfitting:

- Validation RMSE is monitored each epoch
- Training stops when validation error stops improving for several epochs

This significantly improves generalization performance.

---

## Results

| Model                        | Validation RMSE |
| ---------------------------- | --------------- |
| Global mean baseline         | ~1.14           |
| Matrix factorization (tuned) | **0.9169**      |

The final model **significantly outperforms the baseline**, demonstrating effective learning of user–item interactions.

---

---

## How to Run

1. Create and activate a virtual environment
2. Install dependencies:
   ```bash
   pip install numpy pandas
   python src/train.py
   ```
