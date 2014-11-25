from __future__ import print_function, division
import numpy as np
import amitgroup as ag
import deepdish as dd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
import os
import h5py
import caffe
import subprocess
import sys

from deepdish.util.caffe.trainer import train_model

DATASET = 'cifar10'

if DATASET == 'cifar100':
    DIR = os.path.abspath('cifar100_hdf5')
else:
    DIR = os.path.abspath('cifar10_hdf5')
DEVICE_ID = 1

g_seed = 0
g_loop = 0

g_rnd = np.random.randint(100000)

def create_weighted_db(X, y, weights, name='boost'):
    X = X.reshape(-1, 3, 32, 32)
    train_fn = os.path.join(DIR, name + '.h5')

    dd.io.save(train_fn, dict(data=X,
                              label=y.astype(np.float32),
                              sample_weight=weights), compress=False)
    with open(os.path.join(DIR, name + '.txt'), 'w') as f:
        print(train_fn, file=f)


class CNN(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def get_params(self, deep=False):
        return {}

    def fit(self, X, y, sample_weight=None):
        global g_seed
        global g_loop
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0], np.float32)
            print('Calling fit with sample_weight None')
        else:
            sample_weight *= X.shape[0]
            print('Calling fit with sample_weight sum', sample_weight.sum())

        #sample_weight = np.ones(X.shape[0], np.float32)

        #II = sample_weight > 0
        #X = X[II]
        #y = y[II]
        #sample_weight = sample_weight[II]

        #sample_weight = np.ones(X.shape[0])
        w = sample_weight
        #sample_weight[:10] = 0.0
        #w[:1000] = 0.0
        #w = sample_weight
        #w0 = w / w.sum()
        #print('Weight entropy:', -np.sum(w0 * np.log2(w0)))
        print('Weight max:', w.max())
        print('Weight min:', w.min())
        #import sys; sys.exit(0)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Set up weighted database
        create_weighted_db(X, y, sample_weight)

        #steps = [(0.001, 2000, 2000)]
        steps = [(0.001, 0.004, 60000), (0.0001, 0.004, 5000), (0.00001, 0.004, 5000)]
        #steps = [(0.00001, 10000, 10000), (0.000001, 5000, 15000), (0.0000001, 5000, 20000)]
        #steps = [(0.001, 10000, 10000)]
        #steps = [(0.001, 200, 1000)]

        name = os.path.join(CONF_DIR, 'adaboost_{}_loop{}'.format(g_rnd, g_loop))
        bare_conf_fn = os.path.join(CONF_DIR, 'boost_bare.prototxt')
        conf_fn = os.path.join(CONF_DIR, 'solver.prototxt.template')
        #bare_conf_fn = 'regaug_bare.prototxt'
        #conf_fn = 'regaug_solver.prototxt.template'

        net, info = train_model(name, conf_fn, bare_conf_fn, steps,
                                seed=g_seed, device_id=DEVICE_ID)

        loss_fn = 'info/info_{}_loop{}.h5'.format(g_rnd, g_loop)
        dd.io.save(loss_fn, info)
        print('Saved to', loss_fn)

        g_loop += 1

        print('Classifier set up')

        self.net_ = net

    def predict_proba(self, X):
        X = X.reshape(-1, 3, 32, 32)
        #X = X.transpose(0, 2, 3, 1)
        prob = np.zeros((X.shape[0], self.n_classes_))

        M = 2500
        for k in range(int(np.ceil(X.shape[0] / M))):
            y = self.net_.forward_all(data=X[k*M:(k+1)*M]).values()[0].squeeze(axis=(2,3))
            prob[k*M:(k+1)*M] = y

        T = 30.0

        eps = 0.0001

        #prob = prob.clip(eps, 1-eps)

        log_prob = np.log(prob)
        print('log_prob', log_prob.min(), log_prob.max())
        #log_prob = log_prob.clip(min=-4, max=4)
        new_prob = np.exp(log_prob / T)
        new_prob /= dd.apply_once(np.sum, new_prob, [1])

        return new_prob

    def predict(self, X):
        prob = self.predict_proba(X)
        return prob.argmax(-1)

if __name__ == '__main__':
    if len(sys.argv) == 2:
        g_rnd = sys.argv[1]

    #clf = CNN()
    #mean_x = dd.io.load(os.path.join(DIR, 'tr40k-mean.h5'))
    if DATASET == 'cifar100':
        X = np.concatenate([ag.io.load_cifar_100('training', ret='x', count=10000, offset=10000*i, x_dtype=np.float32) for i in range(5)])
    else:
        X = np.concatenate([ag.io.load_cifar_10('training', ret='x', count=10000, offset=10000*i, x_dtype=np.float32) for i in range(4)])
    #X = X.transpose(0, 2, 3, 1)
    X *= 255.0
    mean_x = X.mean(0)
    X -= mean_x
    #te_X, te_y = ag.io.load_cifar_100('testing', x_dtype=np.float32)

    if DATASET == 'cifar100':
        y = np.concatenate([ag.io.load_cifar_100('training', ret='y', count=10000, offset=10000*i) for i in range(5)])
    else:
        y = np.concatenate([ag.io.load_cifar_10('training', ret='y', count=10000, offset=10000*i) for i in range(4)])
    if 1:
        #w = None
        #clf.fit(X, y, sample_weight=w)

        # Test
        if DATASET == 'cifar100':
            te_X, te_y = ag.io.load_cifar_100('testing', x_dtype=np.float32)
        else:
            te_X, te_y = ag.io.load_cifar_10('testing', x_dtype=np.float32)
        te_X *= 255.0
        te_X -= mean_x
        create_weighted_db(te_X, te_y, np.ones(te_X.shape[0], dtype=np.float32), name='test')


        clf = AdaBoostClassifier(base_estimator=CNN(), algorithm='SAMME.R', n_estimators=10,
                                 random_state=0)
        #clf = BaggingClassifier(base_estimator=CNN(), max_samples=0.9, bootstrap=False,
        #                        n_estimators=3, random_state=0)

        clf.fit(X.reshape(X.shape[0], -1), y)
        #print('weights', clf.estimator_weights_)
        #print('train errors', clf.estimator_errors_)
        if 1:
            for i, score in enumerate(clf.staged_score(X.reshape(X.shape[0], -1), y)):
                print(i+1, 'train score', score)
        else:
            print('train score', clf.score(X.reshape(X.shape[0], -1), y))

        #err = clf.estimator_errors_
        #alphas = np.log((1 - err) / err) + np.log(clf.n_classes_ - 1)
        #clf.estimator_weights_ = alphas
        #print('new weights', clf.estimator_weights_)
        if 1:
            for i, score in enumerate(clf.staged_score(te_X.reshape(te_X.shape[0], -1), te_y)):
                print(i+1, 'test score', score)
        else:
            score = clf.score(te_X.reshape(te_X.shape[0], -1), te_y)
            print('test score', score)

    else:
        clf = CNN()
        rs = np.random.RandomState(0)
        #w = rs.uniform(1, 100, size=X.shape[0])
        #w /= w.sum()
        w = None
        clf.fit(X.reshape(X.shape[0], -1), y, sample_weight=w)

        if 0:
            clf = CNN()
            rs = np.random.RandomState(1)
            w = rs.uniform(1, 100, size=X.shape[0])
            w /= w.sum()
            clf.fit(X.reshape(X.shape[0], -1), y, sample_weight=w)
