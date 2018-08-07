import argparse
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda
from chainer.datasets.cifar import (
    get_cifar10,
    get_cifar100,
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from utils.model import Alex
from utils.dataset import load_cifar


def parse():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--dataset', '-d', default='cifar10',
                        help='dataset')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    return parser.parse_args()


def main():
    args = parse()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    train_x, train_y, val_x, val_y = load_cifar()
    class_labels = 10

    model = L.Classifier(Alex(class_labels))
    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(1).use()
        model.to_gpu()  # Copy the model to the GPU

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))

    batch = 100
    for epoch in range(1, args.epoch + 1):

        # k-means
        embeds = []
        append = embeds.append
        for itr in range(0, len(train_x), batch):
            x = train_x[itr:itr + batch]
            x = cuda.to_gpu(x.astype('f'))
            x = cuda.to_cpu(model.predictor.encode(x).data)
            [append(embed.flatten()) for embed in x]
        embeds = np.array(embeds)
        pca = PCA(n_components=256)
        embeds = pca.fit_transform(embeds)
        print('pca owari')
        km = KMeans(n_clusters=100)
        ys = km.fit_predict(embeds)
        print('kmeans owari')
        print(np.unique(ys))

        # prediction
        idx = np.random.permutation(len(train_x))
        for itr in range(0, len(train_x), batch):
            x = train_x[idx][itr:itr + batch]
            y = ys[idx][itr:itr + batch]
            x = cuda.to_gpu(x.astype('f'))
            y = cuda.to_gpu(y)

            loss = model(x, y)
            model.cleargrads()
            loss.backward()
            optimizer.update()
        print('gakushuusitaze-')

        # validation
        acc = 0
        for itr in range(0, len(val_x), batch):
            x = val_x[itr:itr + batch]
            y = val_y[itr:itr + batch]
            x = cuda.to_gpu(x.astype('f'))
            y = cuda.to_gpu(y)
            model(x, y)
            acc += model.accuracy.data

        print(acc / (len(val_x) / batch))
    model.to_cpu()
    chainer.serializers.save_npz('100model.npz', model)


if __name__ == '__main__':
    main()
