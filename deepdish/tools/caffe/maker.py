from __future__ import division, print_function, absolute_import
import argparse
import re
import os
import sys
import numpy as np
from string import Template
import shutil
import json

__version__ = 6

DATA_DIR = os.environ['MAKER_DATA_DIR']

# The data files should be hdf5 and located in $MAKER_DATA_DIR. Each hdf5 should have
# a corresponding .txt file (same name before the extension) with the absolute path
# to the hdf5 file.
DATASETS = {
    'cifar10w40': ([3, 32, 32], ('cifar10_w_tr40k', 'cifar10_w_val')),
    'cifar100w40': ([3, 32, 32], ('cifar100_w_tr40k', 'cifar100_w_val')),
    'cifar100wT': ([3, 32, 32], ('cifar100_w_tr40k', 'cifar100_w_tr40k')),
    'cifar10': ([3, 32, 32], ('cifar10_train', 'cifar10_test')),
    'cifar100': ([3, 32, 32], ('cifar100_train', 'cifar100_test')),
}


def _parse_params(rest):
    params = {}
    v = rest.split('/')
    for s in v:
        key, value = s.split('=', 1)
        params[key] = value
    return params


def trainer(network, name, iters, seeds, device=0, init='',
            test_scores=False, test_responses=False, 
            plot_loss=False, solver_params={}):
    d = {}
    d['name'] = name
    d['seeds'] = seeds - 1
    s = Template("""#!/bin/bash
#SBATCH --job-name=mk${name}
#SBATCH --output=backup-log.out
#SBATCH --error=backup-log.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=pi-yaliamit
#SBATCH --mail-type=END
#SBATCH --mail-user=larsson
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

set -euo pipefail
IFS=$$'\\n\\t'

BIN=$$CAFFE_DIR/build/tools/caffe.bin

echo "LOG: OUT" > log.out
echo "LOG: ERR" > log.err

for SEED in {0..$seeds}; do
""").substitute(d)
    s  += """    echo "----- seed $SEED --------------------" > >(tee -a log.err) \n"""
    cur_iter = 0
    caffemodels = []
    batch = solver_params.get('batch', 100)
    base_model = 'models/v{version}_model_s${{SEED}}_base_iter_0'.format(version=__version__)
    for i, it in enumerate([0] + iters):
        cur_iter += it
        caffemodels.append('models/v{version}_model_s${{SEED}}_iter_{iter}'.format(iter=cur_iter, version=__version__))

    # If we're doing external initialization, run python script here
    if init:
        s += """    $BIN train --solver=solver0_s${{SEED}}.prototxt \n""".format(i=1+i)
        s += """    python -m deepdish.tools.caffe.initialize -i {i} -o {o} -s {seed} "{initstyle}"\n"""\
                        .format(i=base_model, o=caffemodels[0], seed='${SEED}',
                                initstyle=init)

    #cur_iter = 0
    s += "    $BIN train --solver=solver{i}_s${{SEED}}.prototxt ".format(i=1)
    if init:
        s += """ --snapshot={fn}.solverstate""".format(fn=caffemodels[0])
    s += " > >(tee -a log.out) 2> >(tee -a log.err >&2) \n"


    # Test model (TODO: this is an ugly and brittle line)
    _, testfile = DATASETS[network['layers']['data']['args'][0]][1]

    base_args = "{data} {bare} {caffemodel} -d {device} -n {name} -s {seed}".format(
            data=os.path.join(DATA_DIR, testfile + '.h5'),
            bare='bare.prototxt',
            caffemodel=caffemodels[-1] + '.caffemodel',
            device=device,
            name=name,
            seed='${SEED}')

    if test_responses:
        if 1:
            ext_args = "-o responses/response_s{}.h5 -c 5000".format('${SEED}')
            s += "    python -m deepdish.tools.caffe.measure_responses {} {}\n".format(base_args, ext_args)
        else:
            #import glob
            #grammar = re.compile(r'v\d+_model_s(\d+)_iter_(\d+)\.caffemodel')
            for seed in range(seeds):
                for iteration in range(1000, cur_iter+1, 1000):
                    #m = grammar.match(fn)
                    #seed = int(m.group(1))
                    #iteration = int(m.group(2))
                    fn = 'models/v{version}_model_s{seed}_iter_{iter}.caffemodel'.format(
                            version=__version__,
                            seed=seed,
                            iter=iteration)


                    base_args = "{data} {bare} {caffemodel} -d {device} -n {name} -s {seed}".format(
                            data=os.path.join(DATA_DIR, testfile + '.h5'),
                            bare='bare.prototxt',
                            caffemodel=fn,
                            device=device,
                            name=name,
                            seed=seed)
                    ext_args = "-o responses/response_s{}_iter_{}.h5 -c 5000".format(seed, iteration)
                    s += "    python -m deepdish.tools.caffe.measure_responses {} {}\n".format(base_args, ext_args)


            #ext_args = 

    if test_scores:
        ext_args = "-o scores/score_s{}.h5".format('${SEED}')
        s += "    python -m deepdish.tools.caffe.tester {} {}\n".format(base_args, ext_args)

    if plot_loss:
        s += "    python -m deepdish.tools.caffe.parse_log log.err -o loss.h5\n"
        s += "    python -m deepdish.tools.caffe.plot_loss loss.h5 -o log-loss\n"


    cur_iter += it

    s += "done"
    return s


def solver(seed=0, device=0, lr=0.001, decay=0.001, 
           iters=[0], snapshot=10000, base=False, params={}):
    d = {}
    newdef = params.get('new-definition')
    momenta = ""
    momentum = params.get('momentum', 0.9)
    if isinstance(momentum, list):
        for m in momentum:
            momenta += "momenta: {}\n".format(m)
        momentum = 0.0

    if newdef:
        lr0 = lr * (1 - momentum)
    else:
        lr0 = lr
    d['version'] = __version__
    d['seed'] = seed
    d['lr'] = lr0
    d['momentum'] = momentum
    d['momenta'] = momenta
    batch = params.get('batch', 100)
    #d['iter'] = (it * 100) // batch
    d['device_id'] = device

    d['decay'] = decay
    d['snapshot'] = min(snapshot, params.get('snapshot', snapshot))
    d['base_suffix'] = '_base' if base else ''
    d['solver'] = params.get('solver_type', 'SGD')
    # TODO: Read this 10000 from the datasets
    #d['test_iter'] = 100#00 // batch
    d['test_iter'] = 400#00 // batch
    d['test_interval'] = 1000#00 // batch
    d['display'] = 400#00 // batch
    d['gamma'] = params.get('gamma', 0.1)

    steps = ""
    cur_iter = 0
    for iter in iters:
        cur_iter += iter
        steps += "stepvalue: {}\n".format(cur_iter)

    d['steps'] = steps
    d['iter'] = cur_iter
    d['momentum_correction'] = params.get('momentum_correction', 1)
    s = Template("""# version: $version
net: "main.prototxt"
test_iter: $test_iter
test_interval: $test_interval
base_lr: $lr
momentum_correction: $momentum_correction
momentum: $momentum
$momenta
weight_decay: $decay
lr_policy: "multistep"
gamma: $gamma
$steps
display: $display
max_iter: $iter
snapshot: $snapshot
snapshot_prefix: "models/v${version}_model_s${seed}${base_suffix}"
solver_type: $solver
solver_mode: GPU
random_seed: ${seed}
device_id: ${device_id}
""").substitute(d)
    return s

def f_data(last_layer, layer_no, netsize, dataset, params={}, solver_params={}):
    dim, files = DATASETS[dataset]
    train = os.path.join(DATA_DIR, files[0])
    test = os.path.join(DATA_DIR, files[1])
    s = Template("""# version: $version
name: "main"
layers {
  name: "main"
  type: HDF5_DATA
  top: "data"
  top: "label"
  #top: "sample_weight"
  hdf5_data_param {
    source: "$train.txt"
    batch_size: $batch
  }
  include: { phase: TRAIN }
}

layers {
  name: "main"
  type: HDF5_DATA
  top: "data"
  top: "label"
  #top: "sample_weight"
  hdf5_data_param {
    source: "$test.txt"
    batch_size: $batch
  }
  include: { phase: TEST }
}
""").substitute(dict(train=train, test=test,
                     batch=solver_params.get('batch', 100),
                     version=__version__))

    s_bare = Template("""# version: $version
name: "bare"
input: "data"
input_dim: 1
input_dim: $dim0
input_dim: $dim1
input_dim: $dim2
""").substitute(dict(dim0=dim[0], dim1=dim[1], dim2=dim[2], version=__version__))

    return 'data', 'data', 0, tuple(dim), s, s_bare

def f_conv(last_layer, layer_no, netsize, size, num, params={}, solver_params={}):
    name = 'conv{}'.format(layer_no+1)
    pad = int(size) // 2
    init_method = params.get('initialization', solver_params.get('initialization'))

    if init_method == 'xavier':
        weight_filler = """
          type: "xavier" """
    else:
        weight_filler = """
          type: "gaussian"
          std: {std}""".format(std=params.get('std', solver_params.get('std', 0.01)))

    s = Template("""
layers {
  name: "$name"
  type: CONVOLUTION
  bottom: "$last_name"
  top: "$name"
  weight_decay: $decay
  weight_decay: 0
  blobs_lr: $lr
  blobs_lr: $bias_lr
  convolution_param {
    num_output: $num
    pad: $pad
    kernel_size: $size
    stride: 1
    weight_filler {
        $weight_filler
    }
    bias_filler {
      type: "constant"
    }
  }
}
    """).substitute(dict(name=name, last_name=last_layer, size=size, num=num, pad=pad,
                         lr=params.get('lr', 1), weight_filler=weight_filler,
                         decay=params.get('decay', 1),
                         bias_lr=params.get('bias_lr', params.get('lr', 2))))

    new_netsize = (int(num),) + tuple([(ns + 2*pad) - (int(size) - 1) for ns in netsize[1:]])

    return name, name, 1, new_netsize, s, s


def f_pool(last_layer, layer_no, netsize, size, stride, operation, params={}, solver_params={}):
    name = 'pool{}'.format(layer_no)
    op = operation.upper()
    s = Template("""
layers {
  name: "$name"
  type: POOLING
  bottom: "$last_name"
  top: "$name"
  pooling_param {
    pool: $op
    kernel_size: $size
    stride: $stride
  }
}
    """).substitute(dict(name=name, last_name=last_layer, size=size, stride=stride, op=op))

    new_netsize = (netsize[0],) + tuple([(ns // int(stride)) for ns in netsize[1:]])

    return name, name, 0, new_netsize, s, s


def f_relu(last_layer, layer_no, netsize, params={}, solver_params=None):
    name = 'relu{}'.format(layer_no)
    s = Template("""
layers {
  name: "$name"
  type: RELU
  bottom: "$last_name"
  top: "$last_name"
}
    """).substitute(dict(name=name, last_name=last_layer))
    return name, last_layer, 0, netsize, s, s


def f_dropout(last_layer, layer_no, netsize, rate, params={}, solver_params=None):
    name = 'drop{}'.format(layer_no)
    s = Template("""
layers {
  name: "$name"
  type: DROPOUT
  bottom: "$last_name"
  top: "$last_name"
  dropout_param {
    dropout_ratio: $rate
  }
}
    """).substitute(dict(name=name, last_name=last_layer, rate=rate))
    return name, last_layer, 0, netsize, s, s


def f_lrn(last_layer, layer_no, netsize, size, params={}, solver_params={}):
    if size is None:
        size = 3
    name = 'norm{}'.format(layer_no)
    s = Template("""
layers {
  name: "$name"
  type: LRN
  bottom: "$last_name"
  top: "$name"
  lrn_param {
    norm_region: WITHIN_CHANNEL
    local_size: $size
    alpha: 5e-05
    beta: 0.75
  }
}
    """).substitute(dict(name=name, last_name=last_layer, size=size))
    return name, name, 0, netsize, s, s


def f_ip(last_layer, layer_no, netsize, num, params={}, solver_params={}):
    name = 'ip{}'.format(layer_no+1)
    s = Template("""
layers {
  name: "$name"
  type: INNER_PRODUCT
  bottom: "$last_name"
  top: "$name"
  blobs_lr: $lr
  blobs_lr: $bias_lr
  weight_decay: $decay
  weight_decay: 0
  inner_product_param {
    num_output: $num
    weight_filler {
      type: "gaussian"
      std: $std
    }
    bias_filler {
      type: "constant"
    }
  }
}
    """).substitute(dict(name=name, last_name=last_layer, num=num,
                         decay=params.get('decay', 1),
                         std=params.get('std', 0.01), lr=params.get('lr', 1),
                         bias_lr=params.get('bias_lr', params.get('lr', 2))))
    new_netsize = (int(num), 1, 1)
    return name, name, 1, new_netsize, s, s

def f_softmax(last_layer, layer_no, netsize, params={}, solver_params=None):
    s = Template("""
layers {
  name: "accuracy"
  type: ACCURACY
  bottom: "$last_name"
  bottom: "label"
  top: "accuracy"
  include: { phase: TEST }
}

layers {
  name: "loss"
  type: SOFTMAX_LOSS
  bottom: "$last_name"
  bottom: "label"
  #bottom: "sample_weight"
  top: "loss"
}
    """).substitute(dict(last_name=last_layer))

    s_bare = Template("""
layers {
  name: "prob"
  type: SOFTMAX
  bottom: "$last_name"
  top: "prob"
}
    """).substitute(dict(last_name=last_layer))
    new_netsize = netsize
    return 'loss', 'loss', 0, new_netsize, s, s_bare


PATTERNS = [
    (re.compile(r'^d-([a-zA-Z0-9-]+)'), f_data),
    (re.compile(r'^p(?:ool)?(\d+)-(\d+)-(\w+)'), f_pool),
    (re.compile(r'^c(?:onv)?(\d+)-(\d+)'), f_conv),
    (re.compile(r'^(?:ip|fc)(\d+)'), f_ip),
    (re.compile(r'^relu'), f_relu),
    (re.compile(r'^dropout([\d.]+)'), f_dropout),
    (re.compile(r'^lrn(?:(\d+))?'), f_lrn),
    (re.compile(r'^softmax'), f_softmax),
]

def generate_network_files(path, parts, seed=0, device=0, lr=0.001, 
                           decay=0.001, iters=[10000], solver_params={}):
    ret = {}
    ret['main'] = []
    ret['bare'] = []
    cur_layer = 0
    last_layer = None
    cur_size = None
    last_size = None
    info = {}
    info['layers'] = {}
    for i, part in enumerate(parts):
        found = False
        params = {}
        if part.find('/') != -1:
            part, rest = part.split('/', 1)
            params = _parse_params(rest)

        for pattern, func in PATTERNS:
            m = pattern.match(part)
            if m:
                name, last_name, delta, cur_size, s, s_bare = func(last_layer, cur_layer, cur_size, *m.groups(), params=params, solver_params=solver_params)
                if seed == 0:
                    if last_size is None or last_size != cur_size:
                        cs = '{!s:15s} {}'.format(cur_size, np.prod(cur_size))
                    else:
                        cs = ''
                    print('{:7s} {}'.format(name, cs))
                ret['main'].append(s)
                ret['bare'].append(s_bare)
                cur_layer += delta
                last_layer = last_name
                last_size = cur_size
                found = True
                info['layers'][name] = dict(args=m.groups(), kwargs=params)
        if not found:
            raise Exception("Could not parse {}".format(part))

    snapshot = 10000
    s = solver(seed=seed,
               device=device,
               lr=lr,
               iters=[0],
               snapshot=0,
               decay=decay,
               base=True,
               params=solver_params)

    with open(os.path.join(path, 'solver0_s{}.prototxt'.format(seed)), 'w') as f:
        print(s, file=f)

    snapshot = 10000
    s = solver(seed=seed,
               device=device,
               lr=lr,
               iters=iters,
               snapshot=snapshot,
               decay=decay,
               base=(i == 0),
               params=solver_params)

    with open(os.path.join(path, 'solver1_s{}.prototxt'.format(seed)), 'w') as f:
        print(s, file=f)

    for m in 'main', 'bare':
        with open(os.path.join(path, '{}.prototxt'.format(m)), 'w') as f:
            print("".join(ret[m]), file=f)

    return info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', default=0, type=str)
    parser.add_argument('-n', '--number', default=1, type=int)
    parser.add_argument('-c', '--caption', default='A', type=str)
    parser.add_argument('-l', '--decay', default=0.001, type=float)
    parser.add_argument('-r', '--rate', default=0.001, type=float)
    parser.add_argument('--init', type=str)
    parser.add_argument('--iter', nargs='+', default=[10000], type=int)
    parser.add_argument('network', nargs='+', type=str)
    parser.add_argument('-p', '--params', default='{}', type=str)
    parser.add_argument('--calc-scores', action='store_true')
    parser.add_argument('--calc-responses', action='store_true')
    parser.add_argument('--plot-loss', action='store_true')

    args = parser.parse_args()

    path = 'maker{}'.format(args.caption)

    # Create folder
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.mkdir(path)
    for d in ['models', 'scores', 'responses']:
        os.mkdir(os.path.join(path, d))

    params = json.loads(args.params.replace("'", '"'))

    # Write the command line that was used to file
    with open(os.path.join(path, 'command'), 'w') as f:
        print("\n    ".join(sys.argv), file=f)

    for seed in range(args.number):
        print('generate', seed)
        network = generate_network_files(path, args.network,
                                         seed=seed, device=args.device, lr=args.rate,
                                         iters=args.iter, decay=args.decay,
                                         solver_params=params)

    with open(os.path.join(path, 'train.sh'), 'w') as f:
        s = trainer(network, args.caption, args.iter, args.number,
                    device=args.device, init=args.init,
                    test_scores=args.calc_scores,
                    test_responses=args.calc_responses,
                    plot_loss=args.plot_loss,
                    solver_params=params)
        print(s, file=f)


if __name__ == '__main__':
    main()
