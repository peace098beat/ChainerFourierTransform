import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import time
from matplotlib import pyplot as plt

from training import MyChain
model = MyChain(100)
serializers.load_npz('chainer_fourier.model', model)
train_losses = np.load("chainer_fourier.loss.npy")

# 
_sig = 0.0
t = np.linspace(0,1,512)
for f in [1,10,20,30,40,60,70,80,100,500,1000]:
	_sig += np.sin(2*np.pi*f*t)
_noise = np.random.rand(512)
_noise = _noise - np.mean(_noise)
_sig = _sig + _noise


import data
sig, rspec = data.make_datasets(_sig)
est_spec = model.get_predata(sig)

rspec = rspec.reshape((rspec.size, 1))
est_spec = est_spec.reshape((est_spec.size, 1))
print("rspec shape:{}".format(rspec.shape))
print("est spec shape:{}".format(est_spec.shape))

fig = plt.figure()
ax1 = fig.add_subplot(411)
ax2 = fig.add_subplot(412)
ax3 = fig.add_subplot(413)
ax4 = fig.add_subplot(414)
axes = [ax1, ax2, ax3, ax4]
ax1.plot(rspec, c='r', label="rspec", linewidth=1.0)
ax2.plot(est_spec, c='b', label="estimate", linewidth=1.0)
ax3.plot(sig, c='k', label="signal")
ax4.plot(train_losses, label="loss")
for ax in axes:
	ax.legend()

fig.savefig("post.png")
plt.close()
