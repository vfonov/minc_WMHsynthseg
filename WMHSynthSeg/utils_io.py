import numpy as np
import os
from scipy.ndimage import gaussian_filter

try:
    import minc.io
    have_minc_io=True

except ImportError:
    # minc2_simple not available :(
    have_minc_io=False

try:
    import nibabel as nib
    have_nibabel=True
except ImportError:
    # nibabel not available :(
    have_nibabel=False


###############################3

def MRIwrite(volume, aff, filename, dtype=None, history=None):

    if dtype is not None:
        volume = volume.astype(dtype=dtype)
    else:
        dtype = volume.dtype
    if aff is None:
        aff = np.eye(4)
    if filename.endswith('.mnc'):
        assert have_minc_io, 'Need minc2_simple to write .mnc files'
        _volume=volume.transpose([2,1,0]).copy()
        minc.io.save_minc_volume(filename, _volume,  aff=aff, history=history, dtype=dtype)
    else:
        assert have_nibabel, 'Need nibabel to write .nii files'
        header = nib.Nifti1Header()
        nifty = nib.Nifti1Image(volume, aff, header)

        nib.save(nifty, filename)

###############################

def MRIread(filename, dtype=None, im_only=False):

    assert filename.endswith(('.nii', '.nii.gz', '.mgz', '.mnc')), 'Unknown data file: %s' % filename

    if filename.endswith(('.mnc')):
        assert have_minc_io, 'Need minc2_simple to read .mnc files'
        _volume,aff = minc.io.load_minc_volume(filename, np.float32 if dtype is None else dtype)
        # TODO: should I change RAS to LPS?
        # LPS to RAS
        #LPS_to_RAS = np.array([[-1, 0, 0, 0],[0, -1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])
        #aff=np.asarray(LPS_to_RAS@_aff)
        # convert to nibabel convention (x is the last)
        #volume=np.flip(np.flip(_volume,axis=0),axis=1).transpose([2,1,0]).copy()
        volume=_volume.transpose([2,1,0]).copy()
        minc_volume=True

    else:
        assert have_nibabel, 'Need nibabel to read .nii files'
        x = nib.load(filename)
        volume = x.get_fdata()
        aff = x.affine

        if dtype is not None:
            volume = volume.astype(dtype=dtype)
        minc_volume=False

    if im_only:
        return volume
    else:
        return volume, aff, minc_volume

