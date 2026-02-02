import numpy as np

class MatrixFactorization:
    def __init__(self, num_users, num_items, num_factors = 20, lr=0.01, reg=0.1):
        self.num_users = num_users
        self.num_items = num_items
        self.f = num_factors
        self.lr = lr
        self.reg = reg

        #initialises matrix of shape (num, f) with random entries from the gaussian distribution
        self.P = np.random.normal(0,0.1, (num_users, self.f))
        self.Q = np.random.normal(0, 0.1, (num_items, self.f))
        
        #biases for the users and items
        self.bu = np.zeros(num_users)
        self.bi = np.zeros(num_items)

        #initialised global average
        self.mu = 0.0

    #function to predict the rating final
    def safe_predict(self, u, i):
        if u>= self.num_users and i >= self.num_items:
            return self.mu
        
        elif u >= self.num_users:
            return self.mu + self.bi[i]
        
        elif i >= self.num_items:
            return self.mu + self.bu[u]
        
        else:
            return self.mu + self.bu[u] + self.bi[i] + np.dot(self.P[u], self.Q[i])
        

    

    def train_one_epoch(self, data):
        total_error = 0.0

        for u, i, r in data:
            r_hat = self.safe_predict(u,i)
            err = r - r_hat

            total_error += err**2

            self.bu[u] += self.lr * (err - self.reg * self.bu[u])
            self.bi[i] += self.lr * (err - self.reg * self.bi[i])

            #stores the initial value of the user matrix and uses this for future calculations
            Pu = self.P[u].copy()

            self.P[u] += self.lr * (err * self.Q[i] - self.reg * self.P[u])
            self.Q[i] += self.lr *(err * Pu - self.reg * self.Q[i])

        return total_error/len(data)