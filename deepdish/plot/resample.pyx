#!python:
# cython: boundscheck=False
# cython: wraparound=False
# cython: embedsignature=True
# cython: cdivision=True

import numpy as np
cimport numpy as np


def resample_and_arrange_image(np.ndarray[dtype=np.uint8_t, ndim=2] image,
                               np.ndarray[dtype=np.uint8_t, ndim=2] nan_mask,
                               target_size,
                               np.ndarray[dtype=np.float64_t, ndim=2] lut):
    cdef:
        int dim0 = image.shape[0]
        int dim1 = image.shape[1]
        int output_dim0 = target_size[0]
        int output_dim1 = target_size[1]
        np.ndarray[np.float64_t, ndim=3] output = np.empty(target_size + (3,),
                                                          dtype=np.float64)
        np.uint8_t[:, :] image_mv = image
        np.float64_t[:, :, :] output_mv = output
        np.float64_t[:, :] lut_mv = lut
        np.uint8_t[:, :] nan_mask_mv = nan_mask
        double mn = image.min()
        int i, j, c
        np.uint8_t v
        np.float64_t NaN = np.nan

    with nogil:
        for i in range(output_dim0):
            for j in range(output_dim1):
                if nan_mask_mv[i, j]:
                    for c in range(3):
                        output_mv[i, j, c] = NaN
                else:
                    for c in range(3):
                        v = image[dim0*i/output_dim0, dim1*j/output_dim1]
                        output_mv[i, j, c] = lut_mv[v, c]

    return output
