from __future__ import division, print_function, absolute_import 

import numpy as np
import deepdish as dd
import itertools as itr

def main():
    import argparse
    parser = argparse.ArgumentParser('Concats HDF5 files')
    parser.add_argument('input', nargs='+', type=str, help='Input HDF5 files')
    parser.add_argument('-o', '--output', default='output.h5',
                        type=str, help='Output HDF5 file')
    parser.add_argument('-a', '--axis', default=0, type=int,
                        help='Concat over axis')
    parser.add_argument('-k', '--key', nargs='+', type=str, 
                        help='Specify keys to concat')
    parser.add_argument('--uncompressed', action='store_true',
                        help='Do not use compression')
    parser.add_argument('--copy-rest', action='store_true',
                        help='Copy the rest of the key entries across')
    args = parser.parse_args()

    all_keys = []
    all_data = []
    one = False
    for fn in args.input:
        data = dd.io.load(fn, unpack=False)
        all_data.append(data)
        all_keys.append(set(data.keys()))

    # Number of keys
    num_keys = np.asarray([len(k) for k in all_keys])

    if np.all(num_keys == 1) and args.key is None:
        # Ignore keys
        x = np.concatenate([data0.values()[0] for data0 in all_data], axis=args.axis)

        # Use the key in the first file
        key = next(iter(all_keys[0]))
        final = {key: x}
    else:
        keys = None
        if args.key is not None:
            keys = args.key
        else:
            # Take intersection of all keys
            keys = set.intersection(*all_keys)

        # Keys that should be copied without change
        final = {}
        for key in keys:
            final[key] = np.concatenate([data0[key] for data0 in all_data], axis=args.axis)

        # Now copy across the rest (basically the union minus keys)
        # If a key shows up multiple times, it will be overwritten.
        if args.copy_rest:
            for data0 in all_data:
                for key in data0:
                    if key not in keys:
                        final[key] = data0[key]

    dd.io.save(args.output, final, compress=not args.uncompressed)


if __name__ == '__main__':
    main()
