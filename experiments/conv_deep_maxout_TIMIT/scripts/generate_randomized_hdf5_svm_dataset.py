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

feature_shape = X.shape[1:]
F_out = h5py.File('work/timit_CDlabel_fbank_window_{0}.hdf5'.format(dset),'w')
out_data = F_out.create_dataset('data',shape=(n_frames,) + feature_shape,dtype=np.float32)
np.random.seed(0)
data_indices = np.random.permutation(np.arange(n_frames,dtype=np.int32))
out_indices = F_out.create_dataset('indices',data=data_indices)

out_labels = F_out.create_dataset('labels',shape=(n_frames,),dtype=np.int32)

Xmin = np.inf
Xmax = -np.inf

batch_size=10000
n_batches = (F['data'].shape[0] -1)//batch_size + 1
for batch_id in xrange(n_batches):
    X = F['data'][batch_id*batch_size:min(F['data'].shape[0],
                                      (batch_id+1)*batch_size)]
    Xmin = min(Xmin,X.min())
    Xmax = max(Xmax,X.max())



datum_id = 0
for utt_id, (utt_start_ind,utt_end_ind) in enumerate(F['utt_start_end_inds']):
    print utt_id
    use_frames = F['data'][utt_start_ind:utt_end_ind]
    use_labels = F['labels'][utt_start_ind:utt_end_ind]
    use_frames -= Xmin
    use_frames /= (Xmax - Xmin)/255.5
    X = frame_contexts.utterance_to_contexts(use_frames,n_context_frames)
    if utt_id % 10 == 0: print X.shape
    # X1= frame_contexts.utterance_to_contexts(use_frames[:,:40],n_context_frames)
    # X2= frame_contexts.utterance_to_contexts(use_frames[:,41:81],n_context_frames)
    # X3= frame_contexts.utterance_to_contexts(use_frames[:,82:122],n_context_frames)
    # X1 = X1.reshape(X1.shape[0],40,17,1)
    # X2 = X2.reshape(X2.shape[0],40,17,1)
    # X3 = X3.reshape(X3.shape[0],40,17,1)
    # X = np.concatenate((X1,X2,X3),axis=3).astype(np.uint8)

    for x_id, x in enumerate(X):
        shuffled_id = out_indices[datum_id]
        out_data[shuffled_id] = x
        out_labels[shuffled_id] = use_labels[x_id]
        datum_id += 1
        

F.close()
F_out.close()
