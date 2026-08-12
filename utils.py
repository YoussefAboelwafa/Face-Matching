import matplotlib.pyplot as plt
import pickle as pkl


def show_image(img):
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def save_embeddings(file_path, data):
    with open(file_path, "wb") as f:
        pkl.dump(data, f)


def load_embeddings(file_path):
    with open(file_path, "rb") as f:
        data = pkl.load(f)
    return data
