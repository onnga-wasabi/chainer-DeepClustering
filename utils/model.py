import chainer
import chainer.links as L
import chainer.functions as F


class Alex(chainer.Chain):
    def __init__(self, n_class=10):
        super(Alex, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 96, 3, 1, 1)
            self.conv2 = L.Convolution2D(None, 256, 3, 1, 1)
            self.conv3 = L.Convolution2D(None, 384, 3, 1, 1)
            self.conv4 = L.Convolution2D(None, 384, 3, 1, 1)
            self.conv5 = L.Convolution2D(None, 256, 3, 1, 1)
            self.fc6 = L.Linear(None, 4096)
            self.fc7 = L.Linear(None, 4096)
            self.fc8 = L.Linear(None, n_class)

    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 2)
        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h))
        return self.fc8(h)
