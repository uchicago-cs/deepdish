from __future__ import division
import numpy as np
import h5py, argparse
from deepdish.src.feat import frame_contexts
import struct
import os
from PIL import Image
from scipy.misc import imsave

parser = argparse.ArgumentParser("""Construct the database of window data to be fed into the SVM network
""")

parser.add_argument('--dset',type=str,help='dataset to use: train, dev, coreTest')
parser.add_argument('--seed',type=int,default=0,help='random seed for performing the shuffling')
parser.add_argument('--n_context_frames',type=int,default=17,help='number of context frames to use in the frame windows')

args = parser.parse_args()

n_context_frames = args.n_context_frames
dset = args.dset
seed = args.seed


F = h5py.File('work/timit_CDlabel_fbank_{0}.hdf5'.format(dset),'r')

n_frames = F['utt_start_end_inds'][-1,-1]
X = frame_contexts.utterance_to_contexts(F['data'][:2],n_context_frames)

feature_shape = X.shape
F_out = h5py.File('work/timit_CDlabel_fbank_window_{0}.hdf5'.format(dset),'w')
out_data = F_out.create_dataset('data',shape=(n_frames,) + feature_shape,dtype=np.float32)
np.random.seed(0)
data_indices = np.random.permutation(np.arange(n_frames,dtype=np.int32))
out_indices = F_out.create_dataset('indices',data=data_indices)

out_labels = F_out.create_dataset('labels',shape=(n_frames,),dtype=np.int32)

datum_id = 0
for utt_id, (utt_start_ind,utt_end_ind) in enumerate(F['utt_start_end_inds']):
    print utt_id
    use_frames = F['data'][utt_start_ind:utt_end_ind]
    use_labels = F['labels'][utt_start_ind:utt_end_ind]
    X = frame_contexts.utterance_to_contexts(use_frames,n_context_frames)
    for x_id, x in enumerate(X):
        shuffled_id = out_indices[datum_id]
        out_data[shuffled_id] = x
        out_labels[shuffled_id] = use_labels[x_id]
        

F.close()
F_out.close()
