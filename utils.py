import pickle as pkl


def save_embeddings(file_path, data):
    with open(file_path, "wb") as f:
        pkl.dump(data, f)


def load_embeddings(file_path):
    with open(file_path, "rb") as f:
        data = pkl.load(f)
    return data
