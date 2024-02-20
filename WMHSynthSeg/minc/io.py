#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from minc2_simple import minc2_file
from minc2_simple import minc2_xfm
from minc2_simple import minc2_dim
from minc2_simple import minc2_error

from time import gmtime, strftime
import numpy as np

from .geo import decompose


__minc2_to_numpy = {
        minc2_file.MINC2_BYTE:   np.int8,
        minc2_file.MINC2_UBYTE:  np.uint8,
        minc2_file.MINC2_SHORT:  np.int16,
        minc2_file.MINC2_USHORT: np.uint16,
        minc2_file.MINC2_INT:    np.int32,
        minc2_file.MINC2_UINT:   np.uint32,
        minc2_file.MINC2_FLOAT:  np.float32,
        minc2_file.MINC2_DOUBLE: np.float64,
    }
__numpy_to_minc2 = {y: x for x, y in __minc2_to_numpy.items()}


def format_history(argv):
    stamp=strftime("%a %b %d %T %Y>>>", gmtime())
    return stamp+(' '.join(argv))

""" Convert minc file header int voxel to world affine matrix"""
def hdr_to_affine(hdr):
    rot=np.zeros((3,3))
    scales=np.zeros((3,3))
    start=np.zeros(3)

    ax = np.array([h.id for h in hdr])

    for i in range(3):
        aa=np.where(ax == (i+1))[0][0] # HACK, assumes DIM_X=1,DIM_Y=2 etc
        if hdr[aa].have_dir_cos:
            rot[i,:] = hdr[aa].dir_cos
        else:
            rot[i,i] = 1
        scales[i,i] = hdr[aa].step
        start[i] = hdr[aa].start
    origin = start@rot
    out=np.eye(4)
    out[0:3,0:3] = scales@rot
    out[0:3,3] = origin
    return out


"""Convert affine matrix into minc file dimension description"""
def affine_to_dims(aff, shape):
    # convert to minc2 sampling format
    start, step, dir_cos = decompose(aff)
    if len(shape) == 3: # this is a 3D volume
        dims=[
                minc2_dim(id=i+1, length=shape[2-i], start=start[i], step=step[i], have_dir_cos=True, dir_cos=dir_cos[i,0:3]) for i in range(3)
            ]
    elif len(shape) == 4: # this is a 3D grid volume, vector space is the last one
        dims=[
                minc2_dim(id=i+1, length=shape[2-i], start=start[i], step=step[i], have_dir_cos=True, dir_cos=dir_cos[i,0:3]) for i in range(3)
            ] + [ minc2_dim(id=minc2_file.MINC2_DIM_VEC, length=shape[3], start=0, step=1, have_dir_cos=False, dir_cos=[0,0,0])]
    else:
        assert(False)
    return dims


""" 
    Load minc volume into tensor and return voxel2wordl matrix too
"""
def load_minc_volume(fname, dtype=np.float64):
    mm=minc2_file(fname)
    mm.setup_standard_order()

    _dtype=__numpy_to_minc2[dtype]
    d = mm.load_complete_volume(_dtype)
    aff = np.asmatrix(hdr_to_affine(mm.representation_dims()))

    mm.close()
    return d, aff

"""
    Save tensor into minc fille
"""
def save_minc_volume(fn, data, aff, ref_fname=None, history=None, dtype=np.float64):
    dims=affine_to_dims(aff, data.shape)
    out=minc2_file()
    if dtype in __numpy_to_minc2:
        _dtype=__numpy_to_minc2[dtype]
    elif dtype=='int64' or dtype=='uint64':
        _dtype=minc2_file.MINC2_USHORT
        data=data.astype(np.uint16)

    if _dtype==minc2_file.MINC2_DOUBLE :
        out.define(dims, minc2_file.MINC2_USHORT, minc2_file.MINC2_DOUBLE)
    elif _dtype==minc2_file.MINC2_FLOAT:
        out.define(dims, minc2_file.MINC2_USHORT, minc2_file.MINC2_FLOAT)
    else:
        out.define(dims, _dtype, _dtype)
    out.create(fn)
    
    if ref_fname is not None:
        ref=minc2_file(ref_fname)
        out.copy_metadata(ref)

    if history is not None:
        try:
            old_history=out.read_attribute("","history")
            new_history=old_history+"\n"+history
        except minc2_error : 
            new_history=history
        out.write_attribute("","history",new_history)

    out.setup_standard_order()
    out.save_complete_volume(data)
    out.close()


"""
    WIP: load a nonlinear only transform
"""
def load_nl_xfm(fn):
    x=minc2_xfm(fn)
    if x.get_n_concat()==1 and x.get_n_type(0)==minc2_xfm.MINC2_XFM_LINEAR:
        assert(False)
    else:
        _identity=np.asmatrix(np.identity(4))
        _eps=1e-6
        if x.get_n_type(0)==minc2_xfm.MINC2_XFM_LINEAR and x.get_n_type(1)==minc2_xfm.MINC2_XFM_GRID_TRANSFORM:
            assert(np.linalg.norm(_identity-np.asmatrix(x.get_linear_transform(0)) )<=_eps)
            grid_file, grid_invert=x.get_grid_transform(1)
        elif x.get_n_type(0)==minc2_xfm.MINC2_XFM_GRID_TRANSFORM:
            # TODO: if grid have to be inverted!
            grid_file, grid_invert =x.get_grid_transform(0)
        else:
            # probably unsupported type
            assert(False)

        # load grid file into 4D memory
        grid, v2w = load_minc_volume(grid_file, as_byte=False)
        return grid, v2w, grid_invert

"""
    WIP: load a linear only transform
"""
def load_lin_xfm(fn):
    _identity=np.asmatrix(np.identity(4))
    _eps=1e-6
    x=minc2_xfm(fn)

    if x.get_n_concat()==1 and x.get_n_type(0)==minc2_xfm.MINC2_XFM_LINEAR:
        # this is a linear matrix
        lin_xfm=np.asmatrix(x.get_linear_transform())
        return lin_xfm
    else:
        if x.get_n_type(0)==minc2_xfm.MINC2_XFM_LINEAR and x.get_n_type(1)==minc2_xfm.MINC2_XFM_GRID_TRANSFORM:
            # is this identity matrix
            assert(np.linalg.norm(_identity-np.asmatrix(x.get_linear_transform(0)) )<=_eps)
            # TODO: if grid have to be inverted!
            grid_file, grid_invert=x.get_grid_transform(1)
        elif x.get_n_type(1)==minc2_xfm.MINC2_XFM_GRID_TRANSFORM:
            # TODO: if grid have to be inverted!
            (grid_file, grid_invert)=x.get_grid_transform(0)
        assert(False) # TODO
        return None

