import argparse
import os
import sys

import torch
from unet3d.model import UNet3D
from utils_io import MRIread, MRIwrite
from utils import myzoom_torch, align_volume_to_ref
import numpy as np
from torch.nn import Softmax

# for voxel size
from minc.geo import decompose

from lora import add_lora, load_lora_state_dict

def main():
    
    parser = argparse.ArgumentParser(description="WMH-SynthSeg: joint segmentation of anatomy and white matter hyperintensities ",
        epilog="""
If you use this method in a publication, please cite the following article:
Quantifying white matter hyperintensity and brain volumes in heterogeneous clinical and low-field portable MRI
Laso P, Cerri S, Sorby-Adams A, Guo J, Matteen F, Goebl P, Wu J, Liu P, Li H, Young SI, Billot B, Puonti O, Sze G, Payabavash S
DeHavenon A, Sheth KN, Rosen MS, Kirsch J, Strisciuglio N, Wolterink JM, Eshaghi A, Barkhof F, Kimberly WT, and Iglesias JE.
Under review. Preprint available at: https://arxiv.org/abs/2312.05119
""" )
    
    parser.add_argument("i", help="Input image or directory or .txt list")
    parser.add_argument("o", help="Output segmentation (or directory, if the input is a directory) or output list .txt file")

    parser.add_argument("--model", help="Model path", required=True)
    parser.add_argument("--csv_vols", help="(optional) CSV file with volumes of ROIs")
    parser.add_argument("--device", default='cpu', help="device (cpu or cuda; optional)")
    parser.add_argument("--threads", type=int, default=-1,
                        help="(optional) Number of CPU cores to be used. Default is 1. You can use -1 to use all available cores")
    parser.add_argument('-v', '--verbose',default=False, action='store_true') 
    parser.add_argument('--progress',default=False, action='store_true') 
    parser.add_argument('--trim',default=False, action='store_true',help='Trim instead of expand') 
    ### LORA
    parser.add_argument("--lora", help="Lora Model path")
    parser.add_argument("--lora-r", type=int, default=0, help="Lora rank")
    parser.add_argument("--lora-scale", type=float, default=1.0, help="Lora scale")
    

    args = parser.parse_args()

    input_path = args.i
    output_path = args.o
    output_csv_path = args.csv_vols
    model_file = args.model
    device = args.device
    threads = args.threads
    verbose = args.verbose
    progress = args.progress
    trim = args.trim

    # Prepare list of images to segment and leave before loading packages if nothing to do
    if os.path.exists(input_path) is False:
        raise Exception('Input does not exist')

    if output_csv_path is not None:
        head, tail = os.path.split(output_csv_path)
        if ((len(head)>0) and (os.path.isdir(head) is False)):
            raise Exception('Parent directory of CSV file does not exist')
        if tail.endswith('.csv') is False:
            raise Exception('CSV output must be a CSV file')

    if input_path.endswith('.txt') and output_path.endswith('.txt'):
        # reading a list and writing a list 
        with open(input_path,"r") as f:
            images_to_segment=[i.rstrip('\n') for i in f.readlines()]
        with open(output_path,"r") as f:
            segmentations_to_write=[i.rstrip('\n') for i in f.readlines()]
    else:
        if os.path.isfile(input_path): # file
            if not input_path.endswith(('.nii', '.nii.gz', '.mgz', '.mnc', '.txt')):
                raise Exception('Input image is not of a supported type (.nii, .nii.gz,.mgz or .mnc )')
            head, tail = os.path.split(output_path)
            if len(tail) == 0:
                raise Exception('If input is a file, output must be a file')
            if not tail.endswith(('.nii', '.nii.gz', '.mgz', '.mnc')):
                raise Exception('Output image is not of a supported type (.nii, .nii.gz, or .mgz)')
            if ((len(head) > 0) and (os.path.isdir(head) is False)):
                raise Exception('Parent directory of output image does not exist')
            images_to_segment = [input_path]
            segmentations_to_write = [output_path]

    # Set up threads and device
    device = torch.device(device)
    if verbose: print('Using ' + args.device)
    if threads < 0:
        threads = os.cpu_count()
        if verbose: print('Using all available threads ( %s )' % threads)
    torch.set_num_threads(threads)

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

    # We enter PyTorch territory
    with torch.no_grad():

        # Model
        if verbose: print('Preparing model and loading weights')

        model = UNet3D(in_channels, out_channels, final_sigmoid=False, f_maps=f_maps, layer_order=layer_order,
                       num_groups=num_groups, num_levels=num_levels, is_segmentation=False, is3d=True).to(device)
        checkpoint = torch.load(model_file, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        if args.lora is not None:
            add_lora(model, device=device, r=args.lora_r, scale=args.lora_scale, dropout_p=0.1)
            lora_dict = torch.load(args.lora, map_location=device)
            load_lora_state_dict(model, lora_dict)

        model.eval()

        n_ims = len(images_to_segment)
        if output_csv_path is not None:
            csv = open(output_csv_path, 'w')
            csv.write('Input-file,Intracranial-volume')
            for l in range(len(label_names)):
                lab = label_list_segmentation[l]
                if lab>0:
                    name = label_names[l]
                    csv.write(',' + name + '(' + str(lab) + ')')
            csv.write('\n')

        if progress:
            from tqdm import tqdm
            pbar = tqdm(total=n_ims)

        for nim in range(n_ims):
            input_file = images_to_segment[nim]
            output_file = segmentations_to_write[nim]
            if verbose: print('Working on image ' + str(1+nim) + ' of ' + str(+n_ims) + ': ' + input_file)

            # try:

            if verbose: print('     Loading input volume and normalizing to [0,1]')
            image, aff, minc_volume = MRIread(input_file)
            image_torch = torch.tensor(np.squeeze(image).astype(np.float32), device=device)

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
                if verbose: print('     Upscaling to target resolution') # TODO this this awful resampler
                voxsize = np.sqrt(np.sum(aff2 ** 2, axis=0))[:-1]
                factors = voxsize / ref_res
                upscaled = myzoom_torch(image_torch, factors, device=device)
                aff_upscaled = aff2.copy()
                for j in range(3):
                    aff_upscaled[:-1, j] = aff_upscaled[:-1, j] / factors[j]
                aff_upscaled[:-1, -1] = aff_upscaled[:-1, -1] - np.matmul(aff_upscaled[:-1, :-1], 0.5 * (factors - 1))
            else:
                _,voxsize,_ = decompose(aff2)
                factors = voxsize / ref_res # do we need this?
                # upscale/downscale to 1mm ?
                upscaled = image_torch
                aff_upscaled = aff2.copy()
            
            if trim:
                trim_sz = (np.floor(np.array(upscaled.shape) / 32.0) * 32).astype(int)
                upscaled_shift = ((upscaled.shape - trim_sz) // 2).astype(int)
                upscaled_trim = upscaled[upscaled_shift[0]: upscaled_shift[0]+trim_sz[0], upscaled_shift[1]:upscaled_shift[1]+trim_sz[1], upscaled_shift[2]:upscaled_shift[2]+trim_sz[2]].contiguous()

                pred1 = model(upscaled_trim[None, None, ...]).detach()
                pred2 = torch.flip(model(torch.flip(upscaled_trim,[0])[None, None, ...]), [2]).detach()

            else:
                upscaled_padded = torch.zeros(tuple((np.ceil(np.array(upscaled.shape) / 32.0) * 32).astype(int)), device=device)
                upscaled_padded[:upscaled.shape[0], :upscaled.shape[1], :upscaled.shape[2]] = upscaled

                pred1 = model(upscaled_padded[None, None, ...])[:, :, :image_torch.shape[0], :image_torch.shape[1], :image_torch.shape[2]].detach()
                pred2 = torch.flip(model(torch.flip(upscaled_padded,[0])[None, None, ...]), [2])[:, :, :image_torch.shape[0], :image_torch.shape[1], :image_torch.shape[2]].detach()

            softmax = Softmax(dim=0)

            nlat = int((n_labels - n_neutral_labels) / 2.0)
            vflip = np.concatenate([np.array(range(n_neutral_labels)),
                                    np.array(range(n_neutral_labels + nlat, n_labels)),
                                    np.array(range(n_neutral_labels, n_neutral_labels + nlat))])
            
            pred_seg_p = 0.5 * softmax(pred1[0, 0:n_labels, ...]) + 0.5 * softmax(pred2[0, vflip, ...])
            pred_seg = label_list_segmentation_torch[torch.argmax(pred_seg_p, 0)]
            pred_seg = np.squeeze(pred_seg.detach().cpu().numpy())
            # Write volumes from soft segmentation, if needed
            vols = voxelsize * torch.sum(pred_seg_p, dim=[1,2,3]).detach().cpu().numpy()

            if output_csv_path is not None:
                # Subject name and ICV
                csv.write(output_file + ',' + str(np.sum(vols[1:])))
                # volumes of structures
                for l in range(len(label_list_segmentation)):
                    if label_list_segmentation[l] > 0:
                        csv.write(',' + str(vols[l]) )
                csv.write('\n')
                
            if trim: # need to pad with zeros
                pred_seg_ = np.zeros(upscaled.shape, dtype=np.uint8)
                pred_seg_[upscaled_shift[0]:upscaled_shift[0]+trim_sz[0],upscaled_shift[1]:upscaled_shift[1]+trim_sz[1],upscaled_shift[2]:upscaled_shift[2]+trim_sz[2]] = pred_seg
                pred_seg = pred_seg_

            MRIwrite(pred_seg, aff_upscaled, output_file,dtype=np.uint8)

            if progress:
                pbar.update(1)
        #####
        if progress:
            pbar.close()
            
        # We are done!
        if output_csv_path is not None:
            csv.close()
            if verbose: 
                print(' ')
                print('Written volumes to ' + output_csv_path)
                print(' ')
        if verbose:  print('All done!')


# execute script
if __name__ == '__main__':
    main()
