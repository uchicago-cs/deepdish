from __future__ import division, print_function, absolute_import 
import deepdish as dd
from collections import OrderedDict
import numpy as np
import argparse
import re

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('caffelog', type=str)
    parser.add_argument('-o', '--output', default='loss.h5', type=str)

    args = parser.parse_args()

    regex_iter = re.compile(r'Iteration (\d+)')
    regex_loss = re.compile(r'(\w+) net output #(\d+): (\w+) = ([\d.]+)')
    #regex_accu = re.compile(r'(\w+) net output #\d+

    infos = [{}]
    output_names = OrderedDict()

    cur_info = 0

    with open(args.caffelog) as f:
        info = infos[cur_info]
        cur_iter = None
        cur_accuracy = None
        for line in f:
            m = regex_iter.search(line)
            if m:
                iter = int(m.group(1))

                if iter < cur_iter:
                    cur_info += 1
                    infos.append({})
                    info = infos[cur_info]
                    break

                #print('Iteration: {}'.format(m.group(1)))
                cur_iter = iter
                continue

            m = regex_loss.search(line)
            if m:
                dataset = m.group(1)
                output = int(m.group(2))
                loss = float(m.group(4))
                #print('LOSS: {} {}'.format(m.group(1), m.group(2)))
                if dataset not in info:
                    info[dataset] = [[], [], []]

                if output == 0:
                    info[dataset][0].append(cur_iter)
                info[dataset][1+output].append(loss)
                if dataset == 'Test' and output == 0:
                    print(cur_iter, loss)

                output_names[m.group(3)] = None
                continue

    output_names = list(output_names.keys())

    new_infos = []
    for info in infos:
        new_info = {}
        for key in info.keys():
            #if key != 'Train':
                #info[key] = np.asarray(info[key])
            #else:
                #del info[key]

            if key == 'Test':
                new_info[key] = np.asarray(info[key])
        new_infos.append(new_info)

    dd.io.save(args.output, new_infos)

if __name__ == '__main__':
    main()
