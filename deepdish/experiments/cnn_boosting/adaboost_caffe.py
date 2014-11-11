from __future__ import division, print_function, absolute_import 
import numpy as np
import deepdish as dd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
import os
import h5py
import caffe
import subprocess
import sys

from .trainer import train_model

DIR = os.path.abspath('cifar10_hdf5')

g_seed = 0
g_loop = 0

g_rnd = np.random.randint(100000)

def create_weighted_db(X, y, weights):
    X = X.reshape(-1, 3, 32, 32)
    train_fn = os.path.join(DIR, 'adaboost.h5')

    with h5py.File(train_fn, 'w') as f:
        f['data'] = X
        f['label'] = y.astype(np.float32)
        f['sample_weight'] = weights
    with open(os.path.join(DIR, 'adaboost.txt'), 'w') as f:
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

        #steps = [(0.001, 60000, 60000), (0.0001, 5000, 65000), (0.00001, 5000, 70000)]
        #steps = [(0.001, 30000, 30000)]
        steps = [(0.001, 200, 1000)]

        name = 'adaboost_{}_loop{}'.format(g_rnd, g_loop)
        bare_conf_fn = 'adaboost_bare.prototxt'
        conf_fn = 'adaboost_solver.prototxt.template'

        net, info = train_model(name, conf_fn, bare_conf_fn, steps, seed=g_seed)

        loss_fn = 'losses/losses_{}_loop{}.h5'.format(g_rnd, g_loop)
        dd.io.save(loss_fn, info)
        print('Saved to', loss_fn)

        g_loop += 1

        print('Classifier set up')

        self.net_ = net

    def predict_proba(self, X):
        X = X.reshape(-1, 3, 32, 32)
        X = X.transpose(0, 2, 3, 1)
        prob = np.zeros((X.shape[0], 10))

        M = 2500
        for k in range(int(np.ceil(X.shape[0] / M))):
            y = self.net_.predict(X[k*M:(k+1)*M], oversample=False)
            prob[k*M:(k+1)*M] = y

        T = 10.0

        log_prob = np.log(prob)
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
    mean_x = dd.io.load(os.path.join(DIR, 'tr40k-mean.h5'))
    X = np.concatenate([dd.io.load_cifar_10('training', ret='x', count=10000, offset=10000*i, x_dtype=np.float32) for i in range(4)])
    #X = X.transpose(0, 2, 3, 1)
    X *= 255.0
    X -= mean_x

    y = np.concatenate([dd.io.load_cifar_10('training', ret='y', count=10000, offset=10000*i) for i in range(4)])


    if 1:
        #w = None
        #clf.fit(X, y, sample_weight=w)

        clf = AdaBoostClassifier(base_estimator=CNN(), algorithm='SAMME.R', n_estimators=3,
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

        # Test
        te_X, te_y = dd.io.load_cifar_10('testing', x_dtype=np.float32)
        te_X *= 255.0
        te_X -= mean_x
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
