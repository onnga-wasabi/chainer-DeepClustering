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
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--clusters', '-c', type=int, default=10,)
    return parser.parse_args()


def main():
    args = parse()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print()

    train_x, train_y, val_x, val_y = load_cifar()
    class_labels = args.clusters

    model = L.Classifier(Alex(class_labels))
    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))

    batch = args.batchsize
    for epoch in range(1, args.epoch + 1):

        # k-means
        print(epoch)
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
        km = KMeans(n_clusters=class_labels)
        ys = km.fit_predict(embeds)

        # prediction
        acc = 0
        idx = np.random.permutation(len(train_x))
        for itr in range(0, len(train_x), batch):
            x = train_x[idx][itr:itr + batch]
            y = ys[idx][itr:itr + batch]
            x = cuda.to_gpu(x.astype('f'))
            y = cuda.to_gpu(y)

            model.cleargrads()
            loss = model(x, y)
            loss.backward()
            optimizer.update()
            acc += model.accuracy.data
        print(acc / (len(train_x) / batch))

    model.to_cpu()
    chainer.serializers.save_npz('{}_model.npz'.format(class_labels), model)


if __name__ == '__main__':
    main()
