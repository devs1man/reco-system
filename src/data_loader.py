import pandas as pd

def load_data(path):
    df = pd.read_csv(
    "data/u.data",
    sep="\t",
    header = None,
    names = ["userId","movieId","rating","timestamp"]
)

#drops the timestamp column
    df = df.drop(columns=["timestamp"])

    user_ids = df["userId"].unique()
    movie_ids = df["movieId"].unique()

    user_to_index = {uid: idx for idx, uid in enumerate(user_ids)}
    movie_to_index = {mid: idx for idx, mid in enumerate(movie_ids)}

    df["user"] = df["userId"].map(user_to_index)
    df["item"] = df["movieId"].map(movie_to_index)

    data = list(zip(df["user"],df["item"],df["rating"]))

    num_users = len(user_to_index)
    num_items = len(movie_to_index)

    return data, num_items, num_users
