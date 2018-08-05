import argparse
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer.datasets.cifar import (
    get_cifar10,
    get_cifar100,
)
from utils.model import Alex


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

    if args.dataset == 'cifar10':
        print('Using CIFAR10 dataset.')
        class_labels = 10
        train, test = get_cifar10()
    elif args.dataset == 'cifar100':
        print('Using CIFAR100 dataset.')
        class_labels = 100
        train, test = get_cifar100()
    else:
        raise RuntimeError('Invalid dataset choice.')

    model = L.Classifier(Alex(class_labels))
    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)
    train = chainer.training.updaters.StandardUpdater(train_iter, optimizer, device=args.gpu)

    trainer = chainer.training.Trainer(train, stop_trigger=(args.epoch, 'epoch'))
    trainer.extend(chainer.training.extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(chainer.training.extensions.LogReport())
    trainer.extend(chainer.training.extensions.PrintReport(
        ['epoch', 'main/accuracy', 'validation/main/accuracy']
    ))
    trainer.extend(chainer.training.extensions.ProgressBar())
    trainer.run()


if __name__ == '__main__':
    main()
