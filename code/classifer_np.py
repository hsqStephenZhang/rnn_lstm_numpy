import numpy as np
import matplotlib.pyplot as plt

num_classes = 3  # the classes of the points generated
num_dim = 2  # the total dimension of the points
points_per_class = 100  # points per class


def generate_data():
    X = np.zeros((num_classes * points_per_class, num_dim))
    y = np.zeros((num_classes * points_per_class), dtype=np.int16)
    for j in range(num_classes):
        ix = range(j * points_per_class, (j + 1) * points_per_class)
        r = np.linspace(0.0, 1, points_per_class)  # radius
        t = np.linspace(j * 4, j * 4 + 4, points_per_class) + \
            np.random.randn(points_per_class) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j

    plt.scatter(X[:, 0], X[:, 1], c=y, marker='o')
    plt.show()
    return X, y


def Linear_classifier(X, y):
    np.random.seed(7)
    W = np.random.rand(num_dim, num_classes) * 0.01
    b = np.zeros((1, num_classes))

    step_size = 1e-0
    reg = 1e-3
    num_examples = num_classes * points_per_class

    for i in range(200):
        score = np.dot(X, W) + b
        exp_score = np.exp(score)
        # num_examples*num_classes
        probs = exp_score / np.sum(exp_score, axis=1, keepdims=True)
        # num_examples*num_classes
        logprobs = -np.log(probs[range(num_examples), y])
        data_loss = np.sum(logprobs) / num_examples
        reg_loss = 0.5 * reg * np.sum(W * W)
        loss = data_loss + reg_loss

        if i % 10 == 0:
            print("round{},training loss:{}".format(i // 10 + 1, loss))

        dscores = probs
        dscores[range(num_examples), y] -= 1
        dscores /= num_examples

        dW = np.dot(X.T, dscores)
        db = np.sum(dscores, axis=0)

        dW += reg * W

        W -= dW * step_size
        b -= db * step_size

    predicted_class = np.argmax(np.dot(X, W) + b, axis=1)
    print(
        "training accuracy:{}".format(
            np.sum(
                predicted_class == y) /
            num_examples))

    plt.scatter(X[:, 0], X[:, 1], c=predicted_class, marker='o')
    plt.show()


def neural_classifier(X, y):
    num_examples = num_classes * points_per_class
    num_hidden = 100
    step_size = 1e-0
    reg = 1e-3
    np.random.seed(7)

    Wxh = 0.01 * np.random.rand(num_dim, num_hidden)
    Why = 0.01 * np.random.rand(num_hidden, num_classes)
    bh = np.zeros((1, num_hidden))
    by = np.zeros((1, num_classes))

    for i in range(10000):
        hidden = np.maximum(0, np.dot(X, Wxh) + bh)
        score = np.dot(hidden, Why) + by
        exp_score = np.exp(score)
        probs = exp_score / np.sum(exp_score, axis=1, keepdims=True)
        logprobs = -np.log(probs[range(num_examples), y])
        data_loss = np.sum(logprobs) / num_examples
        reg_loss = 0.5 * reg * np.sum(Why * Why) + \
            0.5 * reg * np.sum(Wxh * Wxh)
        loss = data_loss + reg_loss

        if i % 100 == 0:
            print("round:{},loss:{}".format(i // 100 + 1, loss))

        dscore = probs
        dscore[range(num_examples), y] -= 1
        dscore /= num_examples  # 之前的data_loss除以了num_examples，所以这里也需要相应除以该数值

        dWhy = np.dot(hidden.T, dscore)
        dby = np.sum(dscore, axis=0, keepdims=True)
        dhidden = np.dot(dscore, Why.T)
        dhidden[hidden <= 0] = 0
        dWxh = np.dot(X.T, dhidden)
        dbh = np.sum(dhidden, axis=0, keepdims=True)

        dWhy += reg * Why
        dWxh += reg * Wxh

        Why -= step_size * dWhy
        by -= step_size * dby
        hidden -= step_size * dhidden
        Wxh -= step_size * dWxh
        bh -= step_size * dbh

    hidden = np.maximum(0, np.dot(X, Wxh) + bh)
    result = np.dot(hidden, Why) + by
    predicted_class = np.argmax(result,axis=1)

    print("accuracy{}".format( np.sum(predicted_class == y) / num_examples))

    plt.scatter(X[:, 0], X[:, 1], c=predicted_class, marker="o")
    plt.show()


def main():
    X, y = generate_data()
    # Linear_classifier(X, y)
    neural_classifier(X, y)


if __name__ == '__main__':
    main()
