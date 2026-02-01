from src.data_loader import load_data

data, num_users, num_items = load_data("data/u.data")

print(len(data))
print(num_users, num_items)
