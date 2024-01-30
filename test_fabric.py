import sys
from collections import defaultdict
import argparse

import pathlib
from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import lightning as L

import MRI_data
#from evaluate import METRIC_FUNCS, Metrics
from models import dAUTOMAP, UnetModelParallelEncoder, dAUTOMAPDualEncoderUnet

def create_datasets(args):
    if args.dataset == 'Cal':
        test_data   = MRI_data.SliceData(args.test_path,args.center_frac,args.acc_factor,args.sample_rate)
    elif args.dataset == 'Fast':
        test_data   = MRI_data.SliceDataFastTest(args.test_path,args.center_frac,args.acc_factor,args.sample_rate,args.patch_size)
    return test_data

def create_data_loaders(args):
    test_data = create_datasets(args)
    
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    
    return test_loader

def build_dualencoderunet(args):
    model = UnetModelParallelEncoder(
        in_chans=1,
        out_chans=1,
        chans=args.num_chans,
        num_pool_layers=args.num_pools,
        drop_prob=args.drop_prob
    )#.to(args.device)
    
    return model

def build_dautomap(args):
    model_params = {
      'input_shape': (2, args.patch_size, args.patch_size),
      'output_shape': (1, args.patch_size, args.patch_size),
      'tfx_params': {
        'nrow': args.patch_size,
        'ncol': args.patch_size,
        'nch_in': 2,
        'kernel_size': 1,
        'nl': None,
        'init_fourier': True,
        'init': None,
        'bias': False, 
        'share_tfxs': False,
        'learnable': True,
        'shift': False
      },
      'tfx_params2': {
        'nrow': args.patch_size,
        'ncol': args.patch_size,
        'nch_in': 2,
        'kernel_size': 1,
        'nl': 'relu',
        'init_fourier': False,
        'init': 'xavier_uniform_',
        'bias':True,
        'share_tfxs': False,
        'learnable': True,
        'shift': False
      },
      'depth': 2,
      'nl':'relu'
    }

    model = dAUTOMAP(model_params['input_shape'],model_params['output_shape'],model_params['tfx_params'],model_params['tfx_params2'])#.to(args.device)

    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    # model.load_state_dict(checkpoint['model'])
    return model


def build_model(args):
    dautomap_model = build_dautomap(args)
    dualencoderunet_model = build_dualencoderunet(args)
    model = dAUTOMAPDualEncoderUnet(dautomap_model,dualencoderunet_model)#.to(args.device)

    return model

def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    print("dual-encoder loaded from : ",checkpoint_file)
    dautomap_model = build_dautomap(args)
    dualencoderunet_model = build_dualencoderunet(args)
    model = dAUTOMAPDualEncoderUnet(dautomap_model,dualencoderunet_model)#.to(args.device)
    
    model.load_state_dict(checkpoint['model'])
    return model

def run_model(args, model, data_loader):

    model.eval()
    reconstructions = defaultdict(list)

    with torch.no_grad():

        for (iter,data) in enumerate(tqdm(data_loader)):
            image,input_kspace,target, fnames, slices = data 

            # For jupyter
            # image = image.unsqueeze(1).to(args.device)
            # target = target.unsqueeze(1).to(args.device)
            # input_kspace = input_kspace.to(args.device)

            # For Fabric
            image = image.unsqueeze(1)
            target = target.unsqueeze(1)

            input_kspace = input_kspace.permute(0,3,1,2)
            
            image = image.float()
            input_kspace = input_kspace.float()
            target = target.float()
            
            recons, _ = model(input_kspace,image)
            recons = recons.to('cpu').squeeze(1)

            for i in range(recons.shape[0]):
                recons[i] = recons[i] 
                reconstructions[fnames[i]].append((slices[i].numpy(), recons[i].numpy()))

    reconstructions = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)])
        for fname, slice_preds in reconstructions.items()
    }
    return reconstructions

def save_reconstructions(reconstructions, out_dir):    
    print("in save reconsruction")
    
    out_dir.mkdir(exist_ok=True)
    print("out_dir",out_dir)
    for fname, recons in reconstructions.items():
        filename = str(out_dir) + '/' + str(fname)
        np.save(filename, recons)

def main(args):
    data_loader = create_data_loaders(args)
    model = load_model(args.checkpoint_path)#.to(args.device)
    reconstructions = run_model(args, model, data_loader)

    save_reconstructions(reconstructions, args.output_path)
    print("check reconstructions @ :",args.output_path)

def create_arg_parser():

    parser = argparse.ArgumentParser(description="Test setup for MR recon")
    parser.add_argument('--test_path',default = '/Users/taylorfroelich/Code/Python/dualencoder/Data/Test_Fast/',type=str, help='path to test dataset')
    parser.add_argument('--output_path',default = '/Users/taylorfroelich/Code/Python/dualencoder/Data/Test_Recon/20240124/', type=pathlib.Path, help='Path to save the reconstructions to')
    parser.add_argument('--checkpoint_path',default = './checkpoints/20240124/best_model.pt', type=pathlib.Path, help='Path to the model')

    # Stuff for Datset
    parser.add_argument('--dataset',default = 'Fast', type=str, help = 'Cal, Fast, FastNoise')

    # Stuff for Mask
    parser.add_argument('--acc_factor',default = 4, type = int, help = 'acceleration factor')
    parser.add_argument('--center_frac',default = 0.08, type = float, help = 'Center Fraction of k-space')
    parser.add_argument('--sample_rate',default = 0.95, type = float, help = ' Randomly Samples 60per of all avaiable data')
    parser.add_argument('--dataset_type',type=str,help='Cal,else')

    parser.add_argument('--batch_size', default=100, type=int, help='Mini-batch size')
    parser.add_argument('--patch_size', type=int, default=256, help='Sets the image size for recon')

    # Stuff for Dataloader
    parser.add_argument('--shuffle',default = False, type = bool, help = 'Shuffle Data in dataloader')
    parser.add_argument('--data_parallel',default = False, type = bool, help = 'Enable Data Parallel')
    parser.add_argument('--num_workers',default = 0, type = int, help = 'Number of workers')        
    parser.add_argument('--pin_memory',default = False, type = bool, help = 'Enable for GPUs')

    # Stuff for Fabric
    parser.add_argument('--device', type=str, default = 'cpu', help = 'cpu or cuda or mps')
    parser.add_argument('--num_devices', type=int, default = 1, help = 'number of devices to train on')
    
    
    return parser

if __name__ == '__main__':
    #args = create_arg_parser().parse_args(sys.argv[1:])
    args = create_arg_parser().parse_args()
    print(args)
    
    fabric = L.Fabric(accelerator=args.device, devices=args.num_devices)
    fabric.launch()
    
    main(args)