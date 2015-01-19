from __future__ import division, print_function, absolute_import 
import numpy as np
import amitgroup as ag
import deepdish as dd
import os
import caffe
import re
import sys
import subprocess
from subprocess import Popen, PIPE

from deepdish.util.caffe.trainer import train_model

CONF_DIR = 'confs'
DATASET = 'cifar100'
conf_name = 'regression100'


if DATASET == 'cifar100':
    DIR = os.path.abspath('cifar100_hdf5')
else:
    DIR = os.path.abspath('cifar10_hdf5')
DEVICE_ID = 1

g_seed = 0

def selection(y):
    return y >= 0


def transform(x):
    X = x[:,10:-10,10:-10].reshape(x.shape[0], -1)
    X1 = X
    return X1


def create_weighted_db(X, y, weights, name='regression'):
    X = X.reshape(-1, 3, 32, 32)
    train_fn = os.path.join(DIR, '{}.h5'.format(name))

    dd.io.save(train_fn, dict(data=X,
                              label=y.astype(np.float32),
                              sample_weight=weights), compress=False)
    with open(os.path.join(DIR, '{}.txt'.format(name)), 'w') as f:
        print(train_fn, file=f)


#mean_x = dd.io.load(os.path.join(DIR, 'tr40k-mean.h5'))
if DATASET == 'cifar100':
    X = np.concatenate([ag.io.load_cifar_100('training', ret='x', count=10000, offset=10000*i, x_dtype=np.float32) for i in range(5)])
else:
    X = np.concatenate([ag.io.load_cifar_10('training', ret='x', count=10000, offset=10000*i, x_dtype=np.float32) for i in range(4)])

X *= 255.0
mean_x = X.mean(0)
N = X.shape[0]
#X = X.transpose(0, 2, 3, 1)
X -= mean_x

if DATASET == 'cifar100':
    y = np.concatenate([ag.io.load_cifar_100('training', ret='y', count=10000, offset=10000*i) for i in range(5)])
else:
    y = np.concatenate([ag.io.load_cifar_10('training', ret='y', count=10000, offset=10000*i) for i in range(4)])

if DATASET == 'cifar100':
    te_X, te_y = ag.io.load_cifar_100('testing', x_dtype=np.float32)
else:
    te_X, te_y = ag.io.load_cifar_10('testing', x_dtype=np.float32)
te_X *= 255.0
te_X -= mean_x
te_N = te_X.shape[0]

K = len(np.unique(y))
te_Y = np.zeros((te_N, K))
te_Y[np.arange(te_N), te_y] = 1

create_weighted_db(te_X, te_Y, np.ones((te_X.shape[0], K), dtype=np.float32), name='regression_test')

if 0:
    x, y = ag.io.load_mnist('training')
    II = selection(y)
    x = x[II]
    y = y[II]
    y -= y.min()
    N = y.shape[0]

    X1 = transform(x)

    te_x, te_y = ag.io.load_mnist('testing')
    te_N = te_y.shape[0]
    te_II = selection(te_y)
    te_x = te_x[te_II]
    te_y = te_y[te_II]
    te_y -= te_y.min()

    te_X1 = transform(te_x)

w = np.empty((N, K))
p = np.ones((N, K)) / K

#F = lambda x: np.zeros(x.shape[0])

#params = []
#fs = []

FXs = np.zeros((N, K))
te_FXs = np.zeros((te_N, K))

Y = np.zeros((N, K))
Y[np.arange(N), y] = 1

if len(sys.argv) == 2:
    rnd = sys.argv[1]
else:
    rnd = np.random.randint(100000)

log_fn = 'output/output_{}.log'.format(rnd)
print('Output to', log_fn)

all_losses = []

with open(log_fn, 'w') as logfile:

    for loop in range(0, 20):
        w[:] = p * (1 - p)
        print('normalize constant', w.mean())
        wN = w / w.mean(1).sum() * N
        print('min/max w', wN.min(), wN.max())
        #w[:] = np.arange(w.size).reshape(w.shape)
        #w[:] = 1.0
        z = (Y - p) / w

        #z = 0.37 * z - 0.7

        zstd = 15 #z.std()
        z /= zstd
        eps = 3
        print('min/max z', z.min(), z.max())
        #z = z.clip(-eps, eps)

        all_fmj = np.zeros((N, K))
        all_te_fmj = np.zeros((te_N, K))

        create_weighted_db(X, z, wN)

        name = '{}_{}_loop{}'.format(conf_name, rnd, loop)
        conf_fn = os.path.join(CONF_DIR, '{}.prototxt'.format(conf_name))
        bare_conf_fn = os.path.join(CONF_DIR, '{}_bare.prototxt'.format(conf_name))
        solver_conf_fn = os.path.join(CONF_DIR, 'solver.prototxt.template')

        steps = \
        [
            (0.001, 0.004, 100),
            (0.0001, 0.004, 100),
        ]

        if loop == 0 and False:
            net = caffe.Classifier(bare_conf_fn, 'models/regression100_51458_loop0_iter_70000.caffemodel')
            all_fmj = net.forward_all(data=X).values()[0].squeeze(axis=(2,3))
            all_te_fmj = net.forward_all(data=te_X).values()[0].squeeze(axis=(2,3))

            all_fmj *= zstd
            all_te_fmj *= zstd
            info = {}

            if 0:
                # Just load a pre-calculated version instead
                model_fn = base + '.caffemodel'
                net = caffe.Classifier(bare_conf_fn, model_fn, image_dims=(32, 32))
                net.set_phase_test()
                net.set_mode_gpu()
                net.set_device(DEVICE_ID)

                all_fmj = dd.io.load('all_fmj0_eps_inf.h5')
                all_te_fmj = dd.io.load('all_te_fmj0_eps_inf.h5')
        else:
            #warmstart_fn = base + '.solverstate'
            #warmstart_fn = 'models/regression100_6916_loop0_iter_70000.solverstate'
            #warmstart_fn = 'models/adaboost100_35934_loop0_iter_70000.solverstate'
            warmstart_fn = None
            net, info = train_model(name, solver_conf_fn, conf_fn, bare_conf_fn, steps, seed=g_seed, logfile=logfile, device_id=DEVICE_ID, warmstart=warmstart_fn)

            all_fmj = net.forward_all(data=X).values()[0].squeeze(axis=(2,3))
            all_te_fmj = net.forward_all(data=te_X).values()[0].squeeze(axis=(2,3))

            all_fmj *= zstd
            all_te_fmj *= zstd

        g_seed += 1

        #if loop == 0:
            #dd.io.save('all_fmj0_eps_inf.h5', all_fmj)
            #dd.io.save('all_te_fmj0_eps_inf.h5', all_te_fmj)

        fs = (K - 1) / K * (all_fmj - ag.apply_once(np.mean, all_fmj, [1]))
        te_fs = (K - 1) / K * (all_te_fmj - ag.apply_once(np.mean, all_te_fmj, [1]))

        T = 200.0

        FXs += fs / T
        te_FXs += te_fs / T
        exp_FXs = np.exp(FXs)
        p[:] = exp_FXs / ag.apply_once(np.sum, exp_FXs, [1])

        print(loop+1, 'test rate:', (te_FXs.argmax(-1) == te_y).mean(), 'train rate:', (FXs.argmax(-1) == y).mean())

        all_losses.append(info)

losses_fn = 'info/info_{}.h5'.format(rnd)
print('Saving info to ', losses_fn)
dd.io.save(losses_fn, all_losses)

print('Output saved to', log_fn)
