import numpy as np

"""
decompose affine matrix into start, step and direction cosines
"""
def decompose(aff):
    (u,s,vh) = np.linalg.svd(aff[0:3,0:3])
    # remove scaling
    dir_cos = u @ vh
    step  = np.diag(aff[0:3,0:3] @ np.linalg.inv(dir_cos))
    start = (aff[0:3,3].T @ np.linalg.inv(dir_cos)).T
    return start, step, dir_cos

"""
create voxel to pytorch matrix
"""
def create_v2p_matrix(shape):
    v2p = np.diag( [2/shape[2],   2/shape[1],   2/shape[0], 1])
    v2p[0:3,3] = (  1/shape[2]-1, 1/shape[1]-1, 1/shape[0]-1  ) # adjust for half a voxel shift
    return v2p

