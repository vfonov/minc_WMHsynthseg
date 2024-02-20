import argparse
import os
import sys

import torch
from unet3d.model import UNet3D

from utils import align_volume_to_ref, myzoom_torch
from utils_io import MRIread
import numpy as np
from torch.nn import Softmax
# for voxel size
from minc.geo import decompose


def main():
    
    parser = argparse.ArgumentParser(description="WMH-SynthSeg: joint segmentation of anatomy and white matter hyperintensities ",
        epilog="""
If you use this method in a publication, please cite the following article:
Quantifying white matter hyperintensity and brain volumes in heterogeneous clinical and low-field portable MRI
Laso P, Cerri S, Sorby-Adams A, Guo J, Matteen F, Goebl P, Wu J, Liu P, Li H, Young SI, Billot B, Puonti O, Sze G, Payabavash S
DeHavenon A, Sheth KN, Rosen MS, Kirsch J, Strisciuglio N, Wolterink JM, Eshaghi A, Barkhof F, Kimberly WT, and Iglesias JE.
Under review. Preprint available at: https://arxiv.org/abs/2312.05119
""" )
    
    parser.add_argument("m", help="Model path")
    parser.add_argument("i", help="Input sample image for tracing.")
    parser.add_argument("o", help="Output model name in onnx format")
    
    args = parser.parse_args()

    model_file = args.m
    input_path = args.i
    output_path = args.o

    device='cpu'

    # Set up threads and device
    device = torch.device(device)

    # Constants;  TODO:replace by FS paths
    #model_file = os.path.join(os.environ.get('FREESURFER_HOME'), 'models', 'WMH-SynthSeg_v10_231110.pth')
    label_list_segmentation = [0, 14, 15, 16, 24, 77, 85, 2, 3, 4, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28, 41, 42, 43, 46,
                               47, 49, 50, 51, 52, 53, 54, 58, 60]
    label_list_segmentation_torch = torch.tensor(label_list_segmentation, device=device)
    label_names = ['background', '3rd-ventricle', '4th-ventricle', 'brainstem', 'extracerebral_CSF', 'WMH', 'optic-chiasm',
                   'left-white-matter', 'left-cortex', 'left-lateral-ventricle', 'left-cerebellum-white-matter', 'left-cerebellum-cortex',
                   'left-thalamus', 'left-caudate', 'left-putamen', 'left-pallidum', 'left-hippocampus', 'left-amygdala', 'left-accumbens', 'left-ventral-DC',
                   'right-white-matter', 'right-cortex', 'right-lateral-ventricle', 'right-cerebellum-white-matter', 'right-cerebellum-cortex',
                   'right-thalamus', 'right-caudate', 'right-putamen', 'right-pallidum', 'right-hippocampus', 'right-amygdala', 'right-accumbens', 'right-ventral-DC']
    n_neutral_labels = 7
    n_labels = len(label_list_segmentation)
    in_channels = 1
    out_channels = n_labels + 4 + 1 + 1
    f_maps = 64
    layer_order = 'gcl'
    num_groups = 8
    num_levels = 5
    ref_res = np.array([1.0, 1.0, 1.0])
    voxelsize = np.prod(ref_res)

    with torch.no_grad():

        # Model
        model = UNet3D(in_channels, out_channels, final_sigmoid=False, f_maps=f_maps, layer_order=layer_order,
                       num_groups=num_groups, num_levels=num_levels, is_segmentation=False, is3d=True).to(device)
        checkpoint = torch.load(model_file, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        input_file = input_path
        image, aff, minc_volume = MRIread(input_file)
        image_torch = torch.tensor(np.squeeze(image).astype(float), device=device)

        while len(image_torch.shape)>3:
            image_torch = image_torch.mean(image, dim=-1)

        if not minc_volume: # no need in MINC
            image_torch, aff2 = align_volume_to_ref(image_torch, aff, aff_ref=np.eye(4), return_aff=True, n_dims=3)
        else:
            # no need to do any of this with MINC
            image_torch, aff2 = image_torch, aff

        # intensity normalization
        image_torch = image_torch / torch.max(image_torch)

        if not minc_volume: # original broken logik
            voxsize = np.sqrt(np.sum(aff2 ** 2, axis=0))[:-1]
            factors = voxsize / ref_res
            upscaled = myzoom_torch(image_torch, factors, device=device)
            aff_upscaled = aff2.copy()
            for j in range(3):
                aff_upscaled[:-1, j] = aff_upscaled[:-1, j] / factors[j]
            aff_upscaled[:-1, -1] = aff_upscaled[:-1, -1] - np.matmul(aff_upscaled[:-1, :-1], 0.5 * (factors - 1))
            upscaled_padded = torch.zeros(tuple((np.ceil(np.array(upscaled.shape) / 32.0) * 32).astype(int)), device=device)
            upscaled_padded[:upscaled.shape[0], :upscaled.shape[1], :upscaled.shape[2]] = upscaled
        else:
            _,voxsize,_ = decompose(aff2)
            factors = voxsize / ref_res # do we need this?
            # upscale/downscale to 1mm ?
            print(f"{image_torch.shape=}")
            upscaled_padded = torch.zeros(tuple((np.ceil(np.array(image_torch.shape) / 32.0) * 32).astype(int)), device=device)
            upscaled_padded[:image_torch.shape[0], :image_torch.shape[1], :image_torch.shape[2]] = image_torch
            aff_upscaled = aff2.copy()
            print(f"{upscaled_padded.shape=}")

            onnx_path = output_path #+'.onnx'

            if True: # export using torch.onnx.export and then convert to OpenVINO
                torch.onnx.export(
                    model,
                    upscaled_padded[None, None, ...],
                    onnx_path,
                    #opset_version=11,
                    do_constant_folding=True,
                    verbose=False,
                    input_names=['scan'], 
                    output_names=['seg'],
                    dynamic_axes={'scan':{0:'batch', 2:'x', 3:'y', 4:'z'}, 
                                  'seg': {0:'batch', 2:'x', 3:'y', 4:'z'}},
                    )
            else: # export using torch.onnx.dynamo_export
                export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
                model_export=torch.onnx.dynamo_export(
                    model,
                    upscaled_padded,
                    export_options=export_options)
                
                print(model_export)
                model_export.save(onnx_path)
            
            print(f"ONNX model exported to {onnx_path}.")

# execute script
if __name__ == '__main__':
    main()
