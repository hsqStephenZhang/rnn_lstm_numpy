import numpy as np

data = open("../../Datasets/shakespear.txt", 'r').read()
chars = list(set(data))
data_len, vocab_size = len(data), len(chars)

print("chars:{},vocabulary:{}".format(data_len, vocab_size))
char_to_ix = {char: i for i, char in enumerate(chars)}
ix_to_char = {i: char for i, char in enumerate(chars)}

hidden_size = 100
rnn_len = 25
learning_rate = 0.2

Wxh = 0.01 * np.random.rand(hidden_size, vocab_size)
Why = 0.01 * np.random.rand(vocab_size, hidden_size)
Whh = 0.01 * np.random.rand(hidden_size, hidden_size)

bh = np.zeros((hidden_size, 1))
by = np.zeros((vocab_size, 1))


def loss_function(inputs, targets, h_prev):
    """
    inputs is the map of the characters in a string to the index in the vocabulary
    targets is the sequence that put inputs a timestamp forward,which is the rnn tries to fit
    h_prev is the init hidden layer
    """

    xs, hs, ys, ps = {}, {}, {}, {}
    loss = 0
    hs[-1] = np.copy(h_prev)

    for t in range(len(inputs)):
        xs[t] = np.zeros([vocab_size, 1])
        xs[t][inputs[t]] = 1  # create ont-hot code
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t - 1]) + bh)
        ys[t] = np.dot(Why, hs[t]) + by
        exp_y = np.exp(ys[t])
        ps[t] = exp_y / np.sum(exp_y)  # probilities on each timestamp
        loss -= np.log(ps[t][targets[t], 0])

    dWxh, dWhy, dWhh = np.zeros_like(
        Wxh), np.zeros_like(Why), np.zeros_like(Whh)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])

    for t in reversed(range(len(inputs))):  # gradient是多个轮次的累计总和
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1
        dWhy += np.dot(dy, hs[t].T)
        dby += np.copy(dy)
        dh = np.dot(Why.T, dy) + dhnext
        dhraw = (1 - hs[t] * hs[t]) * dh
        dbh += dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t - 1].T)
        dhnext = np.dot(Whh.T, dhraw)

    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -4, 4, out=dparam)

    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1]


def sample(h, seed_ix, n):
    """
    sample a sequence of integers from the model
    h is memory state, seed_ix is seed letter for first time step
    :return the one-hot code of a sequence of letters in the shakespear.txt
    """
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    for t in range(n):
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        p = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size), p=p.ravel())  # 下采样
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)
    return ixes


def train():
    n, p = 0, 0
    h_prev = np.zeros((hidden_size, 1))

    mWxh, mWhh, mWhy = np.zeros_like(
        Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    mbh, mby = np.zeros_like(bh), np.zeros_like(
        by)  # memory variables for Adagrad
    smooth_loss = -np.log(1 / vocab_size) * rnn_len

    while True:
        if p + rnn_len + 1 >= data_len or n == 0:
            h_prev = np.zeros((hidden_size, 1))
            p = 0

        inputs = [char_to_ix[char] for char in data[p:p + rnn_len]]
        targets = [char_to_ix[char] for char in data[p + 1:p + 1 + rnn_len]]

        if n % 100 == 0:
            sample_ix = sample(h_prev, inputs[0], 200)
            strings = "".join(ix_to_char[ix] for ix in sample_ix)
            print("----\n {} \n----".format(strings))
            # print(sample_ix)
            # print(inputs)
            # print(targets)
            # break

        loss, dWxh, dWhh, dWhy, dbh, dby, hprev = loss_function(
            inputs, targets, h_prev)
        smooth_loss = smooth_loss * 0.999 + loss * 0.001
        if n % 100 == 0:
            print('iter %d, loss: %f' % (n, smooth_loss))

        # perform parameter update with Adagrad
        for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                      [dWxh, dWhh, dWhy, dbh, dby],
                                      [mWxh, mWhh, mWhy, mbh, mby]):
            mem += dparam * dparam
            param -= learning_rate * dparam / \
                np.sqrt(mem + 1e-8)  # adagrad update/here we put 1e-8 to prevent zero-devide error

        p += rnn_len  # move to next chunk
        n += 1  # iteration counter


def main():
    train()


if __name__ == '__main__':
    main()
