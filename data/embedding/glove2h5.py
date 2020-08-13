"""Convert raw GloVe word vector text file to h5."""
import h5py
import numpy as np


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def tofloat(value):
    if isfloat(value):
        return float(value)
    else:
        return 0


glove_vectors = [
    line.strip().split()
    for line in open('glove.840B.300d.txt', 'r')
]

vocab = [line[0] for line in glove_vectors]
vectors = np.array(
    [[tofloat(val) for val in line[-300:]] for line in glove_vectors]).astype(np.float32)
vocab = '\n'.join(vocab)

f = h5py.File('glove.840B.300d.h5', 'w')
f.create_dataset(data=vectors, name='embedding')
f.create_dataset(data=vocab, name='words_flatten')
f.close()
