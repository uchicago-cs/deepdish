# frame_contexts.py
# functions to construct frame contexts from hdf5 saved utterances
# Mark Stoehr
from __future__ import division
import numpy as np

def frame_id_to_contexts(frame_id, utt_start_inds, utt_end_inds,context_radius):
    """
    return the context frames
    """
    utt_id = np.where((utt_start_inds <= frame_id ) * (utt_end_inds > frame_id))[0][0]
    utt_start = utt_start_inds[utt_id]
    utt_end = utt_end_inds[utt_id]
    context_start = max(utt_start, frame_id - context_radius)
    context_relative_start = context_start - (frame_id - context_radius)
    context_end  = min(utt_end, frame_id + context_radius + 1)
    context_relative_end = context_relative_start +context_end -context_start
    return context_start, context_end, context_relative_start, context_relative_end

def utterance_to_contexts(utt_frames,n_context_frames):
    """
    return a series of vectors with the context radius
    """
    context_radius = (n_context_frames - 1)//2
    n_frames, n_frame_features = utt_frames.shape
    n_context_features = n_frame_features * n_context_frames
    
    X = np.zeros((n_frames,n_context_features),dtype=utt_frames.dtype)

    feature_window = np.zeros((n_context_frames,n_frame_features),dtype=utt_frames.dtype)


    for frame_id, frame in enumerate(utt_frames):
        # print "n_frames={0}, frame_id+context_radius+1={1}".format(n_frames,frame_id+context_radius+1)
        # print "n_context_frames + n_frames - frame_id + context_radius+1={0}".format(n_frames -frame_id+context_radius+1)
        # print frame_id, "template indices:", max(0,
        #                                          context_radius-frame_id), min(n_context_frames, 
        #                                                                        n_context_frames + n_frames - 
        #                                                                          (frame_id + context_radius+1)), "utt indices:", max(0,frame_id- context_radius), min(n_frames,
        #                    frame_id + context_radius+1)
        # print ""
        if frame_id == 0:
            feature_window[:] = 0.
            feature_window[max(0,
                           context_radius-frame_id):
                       min(n_context_frames, 
                           n_context_frames + n_frames - 
                           (frame_id + context_radius+1))
                       ] = utt_frames[max(0,frame_id- context_radius):
                                      min(n_frames,
                                          frame_id + context_radius+1)]
        else: 
            feature_window[:-1] = feature_window[1:]
            if frame_id + context_radius + 1 > n_frames:
                feature_window[-1] = 0.
            else:
                feature_window[-1] = utt_frames[frame_id + context_radius]

            
        X[frame_id] = feature_window.ravel()
        
    return X


        

def fbank_frame_select_idx(data_dset,labels_dset,utt_start_end_inds,n_context_frames,use_frames):
    """
    Construct an array with n_samples from the data_dset and get
    the associated labels.  This is useful for passing the data
    to machine learning algorithms that will only use a subset

    Parameters
    -----------
    utt_start_end_inds: array_like
    
    
    """
    
    context_radius = (n_context_frames - 1)//2
    
    n_total_samples, n_frame_features = data_dset.shape
    n_context_features = n_frame_features * n_context_frames
    
    

    X = np.zeros((len(use_frames),n_context_features),dtype=np.float32)
    y = np.zeros(len(use_frames),dtype=int)

    utt_start_inds = utt_start_end_inds[:,0]
    utt_end_inds = utt_start_end_inds[:,1]

    for i, frame_id in enumerate(use_frames):
        # figure out where the frame is with respect to the utterances
        context_start, context_end, context_relative_start, context_relative_end = frame_id_to_contexts(frame_id, utt_start_inds, utt_end_inds,context_radius)
        x_start = context_relative_start * n_frame_features
        x_end = context_relative_end * n_frame_features
        
        X[i,x_start:x_end] = data_dset[context_start:context_end,:].ravel()
        y[i] = labels_dset[frame_id]

    return X,y
        
    

def random_fbank_frame_select(data_dset,labels_dset,utt_start_end_inds,n_samples,n_context_frames,random_seed=None):
    """
    Construct an array with n_samples from the data_dset and get
    the associated labels.  This is useful for passing the data
    to machine learning algorithms that will only use a subset

    Parameters
    -----------
    utt_start_end_inds: array_like
    
    
    """
    if random_seed is not None: np.random.seed(random_seed)
    
    context_radius = (n_context_frames - 1)//2
    
    n_total_samples, n_frame_features = data_dset.shape
    n_context_features = n_frame_features * n_context_frames
    
    

    X = np.zeros((n_samples,n_context_features),dtype=np.float32)
    y = np.zeros(n_samples,dtype=int)
    use_frames = np.random.permutation(n_total_samples)[:n_samples]

    utt_start_inds = utt_start_end_inds[:,0]
    utt_end_inds = utt_start_end_inds[:,1]

    for i, frame_id in enumerate(use_frames):
        # figure out where the frame is with respect to the utterances
        context_start, context_end, context_relative_start, context_relative_end = frame_id_to_contexts(frame_id, utt_start_inds, utt_end_inds,context_radius)
        x_start = context_relative_start * n_frame_features
        x_end = context_relative_end * n_frame_features
        
        X[i,x_start:x_end] = data_dset[context_start:context_end,:].ravel()
        y[i] = labels_dset[frame_id]

    return X,y
        
