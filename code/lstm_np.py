import numpy as np
import random
# init the parameters

data = open("../dataset/shakespear.txt", 'r').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
char_to_ix = {char: i for i, char in enumerate(chars)}
ix_to_char = {i: char for i, char in enumerate(chars)}


hidden_size = 100
total_size = hidden_size + vocab_size
lstm_num = 25  # the length of the input sequence
learning_rate = 1e-1


class Param(object):
    """
    this is a single parameter,including the original weight matrix,
    as well as the derivitives,and the momentum for AdaGrad
    """

    def __init__(self, name, value, dtype=float):
        self.name = name
        self.v = value
        self.d = np.zeros_like(value, dtype=dtype)
        self.m = np.zeros_like(value, dtype=dtype)


class Parameters(object):

    """
    storage the parametes in a class
    """

    def __init__(self):
        self.Wf = Param("W_f", 0.1 * np.random.rand(hidden_size, total_size)+0.5)
        self.Wi = Param("W_i", 0.1 * np.random.rand(hidden_size, total_size)+0.5)
        self.Wc = Param("W_c", 0.1 * np.random.rand(hidden_size, total_size)+0.5)
        self.Wo = Param("W_o", 0.1 * np.random.rand(hidden_size, total_size)+0.5)
        self.Wv = Param("W_v", 0.1 * np.random.rand(vocab_size, hidden_size)+0.5)
        self.bf = Param("B_f", np.zeros((hidden_size, 1)))
        self.bi = Param("B_i", np.zeros((hidden_size, 1)))
        self.bc = Param("B_c", np.zeros((hidden_size, 1)))
        self.bo = Param("B_o", np.zeros((hidden_size, 1)))
        self.bv = Param("B_v", np.zeros((vocab_size, 1)))

    def all(self):
        return [
            self.Wc,
            self.Wf,
            self.Wi,
            self.Wo,
            self.Wv,
            self.bc,
            self.bf,
            self.bi,
            self.bo,
            self.bv]


parameters = Parameters()

# some functions as well as the derivitive of them


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def dsigmoid(y):
    return y * (y - 1)


def tanh(x):
    return np.tanh(x)


def dtanh(y):
    return 1 - y * y


def forward(x, h_prev, c_prev, p=parameters):
    assert x.shape == (vocab_size, 1)
    assert h_prev.shape == (hidden_size, 1)
    assert c_prev.shape == (hidden_size, 1)

    z = np.row_stack((h_prev, x))
    f = sigmoid(np.dot(p.Wf.v, z) + p.bf.v)
    i = sigmoid(np.dot(p.Wi.v, z) + p.bi.v)
    c_bar = tanh(np.dot(p.Wc.v, z) + p.bc.v)
    o = sigmoid(np.dot(p.Wo.v, z) + p.bo.v)

    c = f * c_prev + i * c_bar
    h = o * tanh(c)
    v = np.dot(p.Wv.v, h) + p.bv.v  # the output vector of prediction
    y = np.exp(v) / np.sum(np.exp(v))  # the probility of the each single word

    return z, f, i, c_bar, c, o, h, v, y


def backward(
        target,
        dh_next,
        dc_next,
        c_prev,
        z,
        f,
        i,
        c_bar,
        c,
        o,
        h,
        v,
        y,
        p=parameters):
    assert z.shape == (total_size, 1)
    assert v.shape == (vocab_size, 1)
    assert y.shape == (vocab_size, 1)

    for param in [dh_next, dc_next, c_prev, f, i, c_bar, c, o, h]:
        assert param.shape == (hidden_size, 1)

    dv = np.copy(y)
    dv[target] -= 1

    p.Wv.d += np.dot(dv, h.T)
    p.bv.d += dv

    dh = np.dot(p.Wv.v.T, dv)
    dh += dh_next
    do = dh * tanh(c)
    do_raw = do * dsigmoid(o)
    p.Wo.d += np.dot(do_raw, z.T)
    p.bo.d += do_raw

    dc = np.copy(dc_next)
    dc += dh * o * dtanh(c)
    dc_bar = dc * i
    dc_bar_raw = dc_bar * dtanh(c_bar)
    p.Wc.d += np.dot(dc_bar_raw, z.T)
    p.bc.d += dc_bar_raw

    di = dc * c_bar
    di = dsigmoid(i) * di
    p.Wi.d += np.dot(di, z.T)
    p.bi.d += di

    df = dc * c_prev
    df = dsigmoid(f) * df
    p.Wf.d += np.dot(df, z.T)
    p.bf.d += df

    dz = (np.dot(p.Wf.v.T, df)
          + np.dot(p.Wi.v.T, di)
          + np.dot(p.Wc.v.T, dc_bar)
          + np.dot(p.Wo.v.T, do))

    clip_gradients(p=parameters)

    dh_prev = dz[:hidden_size, :]
    dc_prev = f * dc

    return dh_prev, dc_prev


def backward2(target, dh_next, dC_next, C_prev,
             z, f, i, C_bar, C, o, h, v, y,
             p=parameters):
    assert z.shape == (vocab_size + hidden_size, 1)
    assert v.shape == (vocab_size, 1)
    assert y.shape == (vocab_size, 1)

    for param in [dh_next, dC_next, C_prev, f, i, C_bar, C, o, h]:
        assert param.shape == (hidden_size, 1)

    dv = np.copy(y)
    dv[target] -= 1

    p.Wv.d += np.dot(dv, h.T)
    p.bv.d += dv

    dh = np.dot(p.Wv.v.T, dv)
    dh += dh_next
    do = dh * tanh(C)
    do = dsigmoid(o) * do
    p.Wo.d += np.dot(do, z.T)
    p.bo.d += do

    dC = np.copy(dC_next)
    dC += dh * o * dtanh(tanh(C))
    dC_bar = dC * i
    dC_bar = dtanh(C_bar) * dC_bar
    p.Wc.d += np.dot(dC_bar, z.T)
    p.bc.d += dC_bar

    di = dC * C_bar
    di = dsigmoid(i) * di
    p.Wi.d += np.dot(di, z.T)
    p.bi.d += di

    df = dC * C_prev
    df = dsigmoid(f) * df
    p.Wf.d += np.dot(df, z.T)
    p.bf.d += df

    dz = (np.dot(p.Wf.v.T, df)
          + np.dot(p.Wi.v.T, di)
          + np.dot(p.Wc.v.T, dC_bar)
          + np.dot(p.Wo.v.T, do))
    dh_prev = dz[:hidden_size, :]
    dC_prev = f * dC

    return dh_prev, dC_prev


def clear_gradients(p=parameters):
    for item in p.all():
        item.d.fill(0)


def clip_gradients(p=parameters):
    for item in p.all():
        np.clip(item.d, -1, 1, out=item.d)


def update_parameters(lr, p=parameters):
    for item in p.all():
        item.m += item.d * item.d
        item.v -= lr * item.d / np.sqrt(item.m + 1e-8)


def loss_func(inputs, targets, h_prev, c_prev, p=parameters):
    x_s, z_s, f_s, i_s = {}, {}, {}, {}
    c_s, c_bar_s, o_s, h_s = {}, {}, {}, {}
    v_s, y_s = {}, {}

    h_s[-1] = h_prev
    c_s[-1] = c_prev
    loss = 0

    for t in range(len(inputs)):
        x_s[t] = np.zeros((vocab_size, 1))
        x_s[t][inputs[t]] = 1  # Input character

        (z_s[t], f_s[t], i_s[t],
         c_bar_s[t], c_s[t], o_s[t], h_s[t],
         v_s[t], y_s[t]) = \
            forward(x_s[t], h_s[t - 1], c_s[t - 1])  # Forward pass

        loss -= np.log(y_s[t][targets[t], 0])

    clear_gradients()
    dh_next = np.zeros_like(h_prev)
    dc_next = np.zeros_like(c_prev)

    for t in reversed(range(len(inputs))):
        dh_next, dc_next = backward(targets[t], dh_next, dc_next, c_prev, z_s[t],
                                    f_s[t], i_s[t], c_bar_s[t], c_s[t], o_s[t], h_s[t],
                                    v_s[t], y_s[t])

    clip_gradients()

    return loss, h_s[len(inputs) - 1], c_s[len(inputs) - 1]


def sample(c_prev, h_prev, seed_ix, length, p=parameters):
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    h = h_prev
    c = c_prev
    indexes = []
    for t in range(length):
        z, f, i, c_bar, c, o, h, v, y = forward(x, h, c, p)
        index = np.random.choice(range(x.shape[0]), p=y.ravel())
        x = np.zeros((vocab_size,1))
        x[index] = 1
        indexes.append(index)

    return indexes


def train():
    np.random.seed(7)
    random.seed(7)
    iterations, p = 0, 0
    h_prev = np.zeros((hidden_size, 1))
    c_prev = np.zeros((hidden_size, 1))
    smooth_loss = -np.log(1.0 / vocab_size) * lstm_num

    while True:
        if p + lstm_num + 1 >= len(data) or iterations == 0:
            p = 0
            h_prev.fill(0)
            c_prev.fill(0)

        inputs = [char_to_ix[char] for char in data[p:p + lstm_num]]
        targets = [char_to_ix[char] for char in data[p + 1:p + lstm_num + 1]]

        loss, h_prev, c_prev = loss_func(inputs, targets, h_prev, c_prev)

        loss = 0.99 * loss + 0.01 * smooth_loss
        if iterations % 100 ==0 :
            print("iterations:{},loss:{}".format(iterations, loss))
            sample_ix=sample(c_prev,h_prev,seed_ix=inputs[0],length=40)
            print("---/\n {} \n/---".format("".join(ix_to_char[ix] for ix in sample_ix)))

        update_parameters(learning_rate, parameters)

        p += lstm_num
        iterations += 1


def main():
    train()


if __name__ == '__main__':
    main()
