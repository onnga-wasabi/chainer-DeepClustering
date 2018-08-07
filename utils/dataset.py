import pickle
import numpy as np

DATADIR = './data/cifar-10-batches-py/'


def load_cifar(kind=10):
    train_x = []
    train_y = []
    val_x = []
    val_y = []
    if kind == 10:
        fnames = ['{}data_batch_{}'.format(DATADIR, i)
                  for i in range(1, 6)]
        for fname in fnames:
            with open(fname, 'rb') as f:
                dict = pickle.load(f, encoding='bytes')
            [train_x.append(datum) for datum in dict[b'data']]
            [train_y.append(label) for label in dict[b'labels']]
        fname = '{}test_batch'.format(DATADIR)
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
