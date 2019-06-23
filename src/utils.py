import numpy as np
import os


def generate_arrays(x, batch_size=64, shuffle=True):
    """
    Batch generator.
    :param x:
    :param batch_size:
    :param shuffle:
    :return:
    """
    idx = np.arange(len(x))
    if shuffle:
        np.random.shuffle(idx)
    batches = [idx[range(batch_size * i, min(len(x), batch_size * (i + 1)))] for i in
               range(len(x) // batch_size + 1)]
    while True:
        for i in batches:
            xx = x[i]
            yield xx

