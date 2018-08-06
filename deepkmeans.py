import chainer
import chainer.links as L
import chainer.functions as F

from utils.model import Alex


def main():
    train, test = chainer.datasets.cifar.get_cifar10()
    model = Alex(n_class=10)


if __name__ == '__main__':
    main()
