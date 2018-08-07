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

    def encode(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 2)

        return h


class MLP(chainer.Chain):
    def __init__(self, n_class):
        super(MLP, self).__init__()
        with self.init_scope():
            self.fc1 = L.Linear(None, 4096)
            self.fc2 = L.Linear(None, 4096)
            self.fc3 = L.Linear(None, n_class)

    def __call__(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)


class DeepClusteringClassifier(L.Classifier):
    def __call__(self, *args, **kwargs):
        if isinstance(self.label_key, int):
            if not (-len(args) <= self.label_key < len(args)):
                msg = 'Label key %d is out of bounds' % self.label_key
                raise ValueError(msg)
            t = args[self.label_key]
            if self.label_key == -1:
                args = args[:-1]
            else:
                args = args[:self.label_key] + args[self.label_key + 1:]
        elif isinstance(self.label_key, str):
            if self.label_key not in kwargs:
                msg = 'Label key "%s" is not found' % self.label_key
                raise ValueError(msg)
            t = kwargs[self.label_key]
            del kwargs[self.label_key]

        self.y = None
        self.loss = None
        self.accuracy = None
        embed = self.predictor.encode()
        km = DeepClusteringKMeans(X)
        self.y = self.predictor(*args, **kwargs)
        self.loss = self.lossfun(self.y, t)
        chainer.reporter.report({'loss': self.loss}, self)
        if self.compute_accuracy:
            self.accuracy = self.accfun(self.y, t)
            chainer.reporter.report({'accuracy': self.accuracy}, self)

        return self.loss
