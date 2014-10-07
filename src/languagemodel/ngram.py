from __future__ import division
import numpy as np

def hdf5_bigram_counts_by_utt(words,utt_start_end_inds,n_words):
    """
    Assumption is that words are coded by numbers for numbers

    The output is an array such that for each utterance we have
    bigram counts.  Doing this per utterance is useful for LOO
    estimators
    """
    n_utts = utt_start_end_inds.shape[0]
    bigram_counts = np.zeros((n_words,n_words),dtype=np.int)
    for utt_id, (utt_start,utt_end) in enumerate(utt_start_end_inds):
        get_bigram_counts(words[utt_start:utt_end],bigram_counts)
 
    return bigram_counts
    

def get_bigram_counts(words,bigram_count_matrix):
    n_words = len(words)
    for word_id, word in enumerate(words):
        if word_id +1 < n_words:
            bigram_count_matrix[word,words[word_id+1]] += 1

def kfold_bigram_estimation(data,n_data,max_val,n):
    pass

def taper_bigram_counts(bigram_counts):
    pass
