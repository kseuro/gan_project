import torch
from torch import nn, optim
from torchvision import transforms, datasets
from torch.autograd.variable import Variable
from pathlib import Path
import requests
import pickle
import gzip

DATA = 'mnist.pkl.gz'

def mnist_data():
    # Path to data
    data = '/media/hdd1/kai/datasets'
    DATA_PATH = Path("data")
    PATH = DATA_PATH / "mnist"

    PATH.mkdir(parents=True, exist_ok=True)

    URL = "http://deeplearning.net/data/mnist/"
    FILENAME = "mnist.pkl.gz"

    if not (PATH / FILENAME).exists():
            content = requests.get(URL + FILENAME).content
            (PATH / FILENAME).open("wb").write(content)


    with gzip.open( (PATH + DATA).as_posix(), "rb") as f:
        ( (x_train, y_train), (x_valid, y_valid), _) = pickle.load(f,
                                                        encdoing = "latin-1")
    x_train, y_train, x_valid, y_valid = map (
        torch.tensor, (x_train, y_train, x_valid, y_valid)
    )
    # n, c = x_train.shape()
    # x_train, x_train.shape(), y_train.min(), y_train.max()

    return x_train, y_train, x_valid, y_valid

x_train, y_train, x_valid, y_valid = mnist_data()
