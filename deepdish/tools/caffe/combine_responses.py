from __future__ import division, print_function, absolute_import

import numpy as np
import deepdish as dd
from collections import OrderedDict
import re

def concat(x, y):
    if x is None:
        return np.asarray([y])
    else:
        try:
            return np.concatenate([x, [y]])
        except:
            import pdb; pdb.set_trace()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('responses', nargs='+', type=str)
    parser.add_argument('-o', '--output', default='responses.h5', type=str)

    args = parser.parse_args()

    all_responses = None
    responses = {}

    labels = None
    layers = None

    the_name = None

    for s in args.responses:
        data = dd.io.load(s)
        name = str(data.get('name', '(noname)'))

        for ll in data['responses']:
            responses[ll] = concat(responses.get(ll),
                                   data['responses'][ll])

        if layers is None:
            layers = [str(s) for s in data['responses']]


        if the_name is None:
            the_name = name
        else:
            assert name == the_name

    data = {}
    layers = list(sorted(layers, key=lambda x: re.sub(r'\D', '', x)))

    data['name'] = the_name
    data['responses'] = responses
    data['layers'] = layers

    dd.io.save(args.output, data)
