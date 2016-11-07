#! coding:utf-8
from chainer.datasets import tuple_dataset
import numpy as np
import glob
np.random.seed(100)       # 数値はなんでもいい


def make_datasets(sig=None):
    # Make Signal
    if sig is None:
        Ns = 512
        sig = np.random.rand(512)
        sig = sig - np.mean(sig)
    else:
        Ns = sig.size

    sig = sig / np.std(sig)

    def fft(sig, n=512):
        specturm = np.fft.fft(sig, n)
        return specturm
    spec = fft(sig, 512)
    real_spec = np.real(spec)[:int(Ns/2)]
    assert real_spec.size == int(Ns/2)

    sig.astype(dtype=np.float32)
    real_spec.astype(dtype=np.float32)

    return sig, real_spec


def get_datas(N=10000, Nthresh=9000):

    signal_data, spectrum_data = list(), list()

    for i in range(N):
        sig, rspec = make_datasets()
        signal_data.append(sig)
        spectrum_data.append(rspec)

    train = tuple_dataset.TupleDataset(
        signal_data[0:Nthresh], spectrum_data[0:Nthresh])
    test = tuple_dataset.TupleDataset(
        signal_data[Nthresh:],  spectrum_data[Nthresh:])

    return train, test


if __name__ == '__main__':
    
    # Param
    train_size = 10000
    thresh = 9000

    train, test = get_datas(N=train_size, Nthresh=thresh)
