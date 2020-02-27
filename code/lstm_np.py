import numpy as np


def sigmoid(x):
    return 1.0/(1+np.exp(-x))


def dsigmoid(y):
    return y*(y-1)


def tanh(x):
    return np.tanh(x)


def dtanh(y):
    return 1-y*y


def forward():
    pass


def backward():
    pass


def forward_backward():
    pass


def sample():
    pass


def train():
    pass


def main():
    train()


if __name__ == '__main__':
    main()



