from __future__ import division, print_function, absolute_import
import numpy as np
import deepdish as dd
import h5py
import os
import caffe
import re
import sys
import subprocess
from subprocess import Popen, PIPE

from .trainer import train_model, temperature

DIR = os.path.abspath('cifar10_hdf5')
DEVICE_ID = 1

g_seed = 0

def selection(y):
    return y >= 0


def transform(x):
    X = x[:,10:-10,10:-10].reshape(x.shape[0], -1)
    X1 = X
    return X1


def create_weighted_db(X, y, weights):
    X = X.reshape(-1, 3, 32, 32)
    train_fn = os.path.join(DIR, 'regression.h5')

    with h5py.File(train_fn, 'w') as f:
        f['data'] = X
        f['label'] = y.astype(np.float32)
        f['sample_weight'] = weights
    with open(os.path.join(DIR, 'regression.txt'), 'w') as f:
        print(train_fn, file=f)


mean_x = dd.io.load(os.path.join(DIR, 'tr40k-mean.h5'))
X = np.concatenate([dd.io.load_cifar_10('training', ret='x', count=10000, offset=10000*i, x_dtype=np.float32) for i in range(4)])
N = X.shape[0]
#X = X.transpose(0, 2, 3, 1)
X *= 255.0
X -= mean_x

y = np.concatenate([dd.io.load_cifar_10('training', ret='y', count=10000, offset=10000*i) for i in range(4)])

te_X, te_y = dd.io.load_cifar_10('testing', x_dtype=np.float32)
te_X *= 255.0
te_X -= mean_x
te_N = te_X.shape[0]

if 0:
    x, y = dd.io.load_mnist('training')
    II = selection(y)
    x = x[II]
    y = y[II]
    y -= y.min()
    N = y.shape[0]

    X1 = transform(x)

    te_x, te_y = dd.io.load_mnist('testing')
    te_N = te_y.shape[0]
    te_II = selection(te_y)
    te_x = te_x[te_II]
    te_y = te_y[te_II]
    te_y -= te_y.min()

    te_X1 = transform(te_x)

K = 10

w = np.empty((N, K))
p = np.ones((N, K)) / K

#F = lambda x: np.zeros(x.shape[0])

#params = []
#fs = []

FXs = np.zeros((N, K))
te_FXs = np.zeros((te_N, K))

Y = np.zeros((N, K))
Y[np.arange(N), y] = 1

if len(sys.argv) == 0:
    rnd = sys.argv[0]
else:
    rnd = np.random.randint(100000)

log_fn = 'output/output_{}.log'.format(rnd)
print('Output to', log_fn)

all_losses = []

with open(log_fn, 'w') as logfile:

    for loop in range(200):
        w[:] = p * (1 - p)
        print('normalize constant', w.mean())
        wN = w / w.mean(1).sum() * N
        print('min/max w', wN.min(), wN.max())
        #w[:] = np.arange(w.size).reshape(w.shape)
        #w[:] = 1.0
        z = (Y - p) / w
        z /= 3
        eps = 3
        print('min/max z', z.min(), z.max())
        #z = z.clip(-eps, eps)

        all_fmj = np.zeros((N, K))
        all_te_fmj = np.zeros((te_N, K))

        create_weighted_db(X, z, wN)

        name = 'regression_{}_loop{}'.format(rnd, loop)
        bare_conf_fn = 'regression_bare.prototxt'
        conf_fn = 'regression_solver.prototxt.template2'

        #steps = [(0.001, 400, 400), (0.0001, 400, 800), (0.00001, 400, 1200)]
        steps = [(0.001, 60000, 60000), (0.0001, 5000, 65000), (0.00001, 5000, 70000)]
        #steps = [(0.001, 60000, 70000+60000), (0.0001, 5000, 70000+65000), (0.00001, 5000, 70000 + 70000)]
        #steps = [(0.001, 1000, 1000)]
        #steps = [(0.001, 1000, 1000), (0.0001, 1000, 2000), (0.00001, 1000, 3000)]
        base = 'models/regression_basic_loop0_iter_70000'

        if loop == 0 and False:
            # Just load a pre-calculated version instead
            model_fn = base + '.caffemodel'
            net = caffe.Classifier(bare_conf_fn, model_fn, image_dims=(32, 32))
            net.set_phase_test()
            net.set_mode_gpu()
            net.set_device(DEVICE_ID)
            info = {}

            all_fmj = dd.io.load('all_fmj0.h5')
            all_te_fmj = dd.io.load('all_te_fmj0.h5')
        else:
            #warmstart_fn = base + '.solverstate'
            net, info = train_model(name, conf_fn, bare_conf_fn, steps, seed=g_seed, logfile=logfile, device_id=DEVICE_ID)#, warmstart=warmstart_fn)

            all_fmj = net.predict(X.transpose(0, 2, 3, 1), oversample=False)
            all_te_fmj = net.predict(te_X.transpose(0, 2, 3, 1), oversample=False)

        g_seed += 1

        #if loop == 0:
            #dd.io.save('all_fmj0_eps_inf.h5', all_fmj)
            #dd.io.save('all_te_fmj0_eps_inf.h5', all_te_fmj)

        fs = (K - 1) / K * (all_fmj - dd.apply_once(np.mean, all_fmj, [1]))
        te_fs = (K - 1) / K * (all_te_fmj - dd.apply_once(np.mean, all_te_fmj, [1]))

        FXs += fs
        te_FXs += te_fs
        exp_FXs = np.exp(FXs)
        p[:] = exp_FXs / dd.apply_once(np.sum, exp_FXs, [1])

        T = 100.0

        # Raise temperature
        p[:] = temperature(p, T) 

        print('test rate:', (te_FXs.argmax(-1) == te_y).mean(), 'train rate:', (FXs.argmax(-1) == y).mean())

        all_losses.append(info)

losses_fn = 'losses/losses_{}.h5'.format(rnd)
print('Saving losses to ', losses_fn)
dd.io.save(losses_fn, all_losses)
