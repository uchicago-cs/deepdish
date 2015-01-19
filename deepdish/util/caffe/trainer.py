from __future__ import division, print_function, absolute_import 
import numpy as np
import amitgroup as ag
import h5py
import os
import caffe
import re
import sys
import subprocess
from subprocess import Popen, PIPE

TOOLS_DIR = os.path.expandvars('$CAFFE_DIR/build/tools')
CAFFE_BIN = os.path.join(TOOLS_DIR, 'caffe')

def temperature(y, T):
    y = y**(1 / T)
    return y / ag.apply_once(np.sum, y, [1])

def train_model(name, conf_fn, bare_conf_fn, steps, logfile=None, seed=0,
        models_dir='models', confs_dir='confs', device_id=0, warmstart=None, start_iter=0,
        image_dims=(32, 32)):
    iters = []
    losses = []
    learning_rates = []

    output_matcher = re.compile(r'Iteration (\d+), loss = ([0-9.]+)')
    accuracy_matcher = re.compile(r'Test net output #0: accuracy = ([\d.]+)')
    with open(conf_fn) as f:
        raw_conf = f.read()
    last_iter = None
    max_iter = start_iter
    for i, (base_lr, weight_decay, plus_iter) in enumerate(steps):
        max_iter += plus_iter
        out = os.path.join(models_dir, name)
        conf = raw_conf.format(seed=seed, base_lr=base_lr, snapshot=0,
                               weight_decay=weight_decay, out=out, max_iter=max_iter, device_id=device_id)
        conf_fn = '{name}_solver_{i}.prototxt'.format(name=name, i=i)
        full_conf_fn = os.path.join(confs_dir, conf_fn)

        with open(full_conf_fn, 'w') as f:
            f.write(conf)


        if last_iter is not None:
            warmstart_str = "--snapshot={out}_iter_{iter}.solverstate".format(
                out=out, iter=last_iter)
        else:
            if warmstart is not None:
                warmstart_str = "--snapshot=" + warmstart
            else:
                warmstart_str = ""

        cmd = """{bin} train --solver={conf_fn} {warmstart}""".format(
                bin=CAFFE_BIN, conf_fn=full_conf_fn, warmstart=warmstart_str)
        print('+' + cmd)
        pr = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

        for line in iter(pr.stderr.readline, ""):
            match = output_matcher.search(line)
            if match is not None:
                iteration = int(match.group(1))
                loss = float(match.group(2))
                iters.append(iteration)
                losses.append(loss)
                learning_rates.append(base_lr)
                print('{:7} {:10.05f} (lr={})'.format(iteration, loss, base_lr))
            match = accuracy_matcher.search(line)
            if match is not None:
                acc = float(match.group(1))
                learning_rates.append(base_lr)
                print('accurancy = {} ({:.2f}%)'.format(acc, 100*(1 - acc)))
            if logfile is not None:
                print(line, end='', file=logfile)

        #assert pr.returncode == 0

        last_iter = max_iter

    print('Loading classifier')
    model_fn = '{out}_iter_{iter}.caffemodel'.format(out=out, iter=max_iter)
    net = caffe.Classifier(bare_conf_fn, model_fn, image_dims=image_dims)
    net.set_phase_test()
    net.set_mode_gpu()
    net.set_device(device_id)
    info = dict(iterations=np.array(iters, dtype=np.int64),
                losses=np.asarray(losses),
                learning_rates=np.asarray(learning_rates),
                last_iter=max_iter)
    return net, info
