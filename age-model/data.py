import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings_data):
        self.data = embeddings_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        embedding, age = self.data[idx]
        embedding = torch.tensor(embedding, dtype=torch.float32)
        age = torch.tensor(int(age), dtype=torch.long)
        return embedding, age
