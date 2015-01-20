from __future__ import division, print_function, absolute_import
import argparse
import re
import os
import sys
import numpy as np
from string import Template
import shutil

__version__ = 3

DATA_DIR = os.environ['MAKER_DATA_DIR']

DATASETS = {
    'cifar10w40': ([3, 32, 32], ('cifar10_w_tr40k.txt', 'cifar10_w_val.txt')),
    'cifar100w40': ([3, 32, 32], ('cifar100_w_tr40k.txt', 'cifar100_w_val.txt')),
    'cifar10': ([3, 32, 32], ('cifar10_train.txt', 'cifar10_test.txt')),
    'cifar100': ([3, 32, 32], ('cifar100_train.txt', 'cifar100_test.txt')),
}

def trainer(name, iters, seeds):
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
    s += """    echo "----- seed $SEED --------------------\n"""
    cur_iter = 0
    caffemodels = []
    for i, it in enumerate(iters):
        cur_iter += it
        caffemodels.append('models/v{version}_model_s${{SEED}}_iter_{iter}'.format(iter=cur_iter, version=__version__))

    cur_iter = 0
    for i, it in enumerate(iters):
        s += "    if [ ! -f {fn}.caffemodel ]; then\n".format(fn=caffemodels[i])
        s += "         $BIN train --solver=solver{i}_s${{SEED}}.prototxt ".format(i=i)
        if i != 0:
            s += """ --snapshot={fn}.solverstate""".format(fn=caffemodels[i-1])

        s += " > >(tee -a log.out) 2> >(tee -a log.err >&2) \n"
        s += "    fi\n"
        cur_iter += it

    s += "done"
    return s


def solver(seed=0, device=0, lr=0.001, decay=0.001, it=10000, snapshot=10000):
    d = {}
    d['version'] = __version__
    d['seed'] = seed
    d['lr'] = lr
    d['iter'] = it
    d['device_id'] = device
    d['momentum'] = 0.9
    d['decay'] = decay
    d['snapshot'] = snapshot
    s = Template("""# version: $version
net: "main.prototxt"
test_iter: 100
test_interval: 1000
base_lr: $lr
momentum: $momentum
weight_decay: $decay
lr_policy: "fixed"
display: 500
max_iter: $iter
snapshot: $snapshot
snapshot_prefix: "models/v${version}_model_s${seed}"
solver_mode: GPU
random_seed: ${seed}
device_id: ${device_id}
""").substitute(d)
    return s

def f_data(last_layer, layer_no, netsize, dataset, batch, g={}):
    if batch is None: batch = 100
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
  top: "sample_weight"
  hdf5_data_param {
    source: "$train"
    batch_size: $batch
  }
  include: { phase: TRAIN }
}

layers {
  name: "cifar"
  type: HDF5_DATA
  top: "data"
  top: "label"
  top: "sample_weight"
  hdf5_data_param {
    source: "$test"
    batch_size: $batch
  }
  include: { phase: TEST }
}
""").substitute(dict(train=train, test=test, batch=batch, version=__version__))

    s_bare = Template("""# version: $version
name: "bare"
input: "data"
input_dim: 1
input_dim: $dim0
input_dim: $dim1
input_dim: $dim2
""").substitute(dict(dim0=dim[0], dim1=dim[1], dim2=dim[2], version=__version__))

    return 'data', 'data', 0, tuple(dim), s, s_bare

def f_conv(last_layer, layer_no, netsize, size, num, std, g={}):
    if std is None:
        std = 0.01
    name = 'conv{}'.format(layer_no+1)
    pad = int(size) // 2
    s = Template("""
layers {
  name: "$name"
  type: CONVOLUTION
  bottom: "$last_name"
  top: "$name"
  blobs_lr: 1
  blobs_lr: 2
  convolution_param {
    num_output: $num
    pad: $pad
    kernel_size: $size
    stride: 1
    weight_filler {
      type: "gaussian"
      std: $std
    }
    bias_filler {
      type: "constant"
    }
  }
}
    """).substitute(dict(name=name, last_name=last_layer, size=size, num=num, pad=pad, std=std))

    new_netsize = (int(num),) + tuple([(ns + 2*pad) - (int(size) - 1) for ns in netsize[1:]])

    return name, name, 1, new_netsize, s, s


def f_pool(last_layer, layer_no, netsize, size, stride, operation, g={}):
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


def f_relu(last_layer, layer_no, netsize, g=None):
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


def f_dropout(last_layer, layer_no, netsize, rate, g=None):
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


def f_lrn(last_layer, layer_no, netsize, size, g={}):
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


def f_ip(last_layer, layer_no, netsize, num, std, decay, g={}):
    if decay is None: decay = 1
    name = 'ip{}'.format(layer_no+1)
    s = Template("""
layers {
  name: "$name"
  type: INNER_PRODUCT
  bottom: "$last_name"
  top: "$name"
  blobs_lr: 1
  blobs_lr: 2
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
    """).substitute(dict(name=name, last_name=last_layer, num=num, decay=decay, std=std))
    new_netsize = (int(num), 1, 1)
    return name, name, 1, new_netsize, s, s

def f_softmax(last_layer, layer_no, netsize):
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
  bottom: "sample_weight"
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
    (re.compile(r'd-([a-z0-9]+)(?:/(\d+))?'), f_data),
    (re.compile(r'p(\d+)-(\d+)-(\w+)'), f_pool),
    (re.compile(r'c(\d+)-(\d+)(?:-([\d.]+))?'), f_conv),
    (re.compile(r'ip(\d+)-([\d.]+)(?:\[(\d+)\])?'), f_ip),
    (re.compile(r'relu'), f_relu),
    (re.compile(r'dropout([\d.]+)'), f_dropout),
    (re.compile(r'lrn(?:(\d+))?'), f_lrn),
    (re.compile(r'softmax'), f_softmax),
]

def generate_network_files(path, parts, seed=0, device=0, lr=0.001, 
                           decay=0.001, iters=[10000]):
    ret = {}
    ret['main'] = []
    ret['bare'] = []
    cur_layer = 0
    last_layer = None
    cur_size = None
    last_size = None
    for i, part in enumerate(parts):
        found = False
        for pattern, func in PATTERNS:
            m = pattern.match(part)
            if m:
                name, last_name, delta, cur_size, s, s_bare = func(last_layer, cur_layer, cur_size, *m.groups())
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
        if not found:
            raise Exception("Could not parse {}".format(part))

    cur_iter = 0
    cur_lr = lr
    for i, it in enumerate(iters):
        snapshot = cur_iter
        cur_iter += it
        s = solver(seed=seed, device=device, lr=cur_lr, it=cur_iter, snapshot=cur_iter-snapshot, decay=decay)

        with open(os.path.join(path, 'solver{}_s{}.prototxt'.format(i, seed)), 'w') as f:
            print(s, file=f)

        cur_lr /= 10

    for m in 'main', 'bare':
        with open(os.path.join(path, '{}.prototxt'.format(m)), 'w') as f:
            print("".join(ret[m]), file=f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', default=0, type=str)
    parser.add_argument('-n', '--number', default=1, type=int)
    parser.add_argument('-c', '--caption', default='A', type=str)
    parser.add_argument('-l', '--decay', default=0.001, type=float)
    parser.add_argument('-r', '--rate', default=0.001, type=float)
    parser.add_argument('--iter', nargs='+', default=[10000], type=int)
    parser.add_argument('network', nargs='+', type=str)

    args = parser.parse_args()

    path = 'maker{}'.format(args.caption)

    # Create folder
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.mkdir(path)
    os.mkdir(os.path.join(path, 'models'))

    # Write the command line that was used to file
    with open(os.path.join(path, 'command'), 'w') as f:
        print("\n    ".join(sys.argv), file=f)

    for seed in range(args.number):
        print('generate', seed)
        network = generate_network_files(path, args.network,
                                         seed=seed, device=args.device, lr=args.rate,
                                         iters=args.iter, decay=args.decay)

    with open(os.path.join(path, 'train.sh'), 'w') as f:
        s = trainer(args.caption, args.iter, args.number)
        print(s, file=f)


if __name__ == '__main__':
    main()
