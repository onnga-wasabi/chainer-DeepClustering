from pathlib import Path
import gzip
import pickle
import numpy as np

DATADIR = Path(__file__).resolve().parents[1] / 'data'


def load_cifar(kind=10):
    cifar = DATADIR / 'cifar-10-batches-py'
    train_x = []
    train_y = []
    val_x = []
    val_y = []
    if kind == 10:
        fnames = ['{}/data_batch_{}'.format(cifar, i)
                  for i in range(1, 6)]
        for fname in fnames:
            with open(fname, 'rb') as f:
                dict = pickle.load(f, encoding='bytes')
            [train_x.append(datum) for datum in dict[b'data']]
            [train_y.append(label) for label in dict[b'labels']]
        fname = '{}/test_batch'.format(cifar)
        with open(fname, 'rb') as f:
            dict = pickle.load(f, encoding='bytes')
            [val_x.append(datum) for datum in dict[b'data']]
            [val_y.append(label) for label in dict[b'labels']]
    elif kind == 100:
        pass

    train_x = np.array(train_x)
    train_x = train_x.reshape(-1, 3, 32, 32)
    train_x = train_x.transpose(0, 2, 3, 1)
    train_y = np.array(train_y)
    val_x = np.array(val_x)
    val_x = val_x.reshape(-1, 3, 32, 32)
    val_x = val_x.transpose(0, 2, 3, 1)
    val_y = np.array(val_y)

    return train_x, train_y, val_x, val_y


def read_gz(kind):
    fashion = DATADIR / 'fashion-mnist'
    with gzip.open('{}/{}-images-idx3-ubyte.gz'.format(fashion, kind), 'rb') as rf:
        images = np.frombuffer(rf.read(), dtype=np.uint8, offset=16)
    with gzip.open('{}/{}-labels-idx1-ubyte.gz'.format(fashion, kind), 'rb') as rf:
        labels = np.frombuffer(rf.read(), dtype=np.uint8, offset=8)
    return images, labels


def load_fashion_mnist():
    train_x, train_y = read_gz('train')
    val_x, val_y = read_gz('t10k')
    return train_x.reshape(-1, 1, 28, 28), train_y, val_x.reshape(-1, 1, 28, 28), val_y


if __name__ == '__main__':
    load_fashion_mnist()
