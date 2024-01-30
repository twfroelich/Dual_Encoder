import time
import shutil
from pathlib import Path
import random
import pathlib
import numpy as np
from matplotlib import pyplot as plt
import logging
from tqdm import tqdm
import argparse

#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader

import lightning as L

import MRI_data
from models import dAUTOMAP, UnetModelParallelEncoder, dAUTOMAPDualEncoderUnet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_datasets(args):

    if args.dataset == 'Cal':
        train_data = MRI_data.SliceData(args.train_path,args.center_frac,args.acc_factor,sample_rate=args.sample_rate)
        dev_data   = MRI_data.SliceData(args.validation_path,args.center_frac,args.acc_factor,sample_rate=args.sample_rate)
    elif args.dataset == 'Fast':
        train_data = MRI_data.SliceDataFast(args.train_path,args.center_frac,args.acc_factor,args.sample_rate,args.patch_size)
        dev_data   = MRI_data.SliceDataFast(args.validation_path,args.center_frac,args.acc_factor,args.sample_rate,args.patch_size)
    elif args.dataset == 'Fast_Noise':
        train_data = MRI_data.SliceDataFastNoise(args.train_path,args.center_frac,args.acc_factor,args.sample_rate,args.patch_size)
        dev_data   = MRI_data.SliceDataFastNoise(args.validation_path,args.center_frac,args.acc_factor,args.sample_rate,args.patch_size)

    return dev_data, train_data

def create_data_loaders(args):

    dev_data, train_data = create_datasets(args)
    
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle = args.shuffle,
        num_workers = args.num_workers,
        pin_memory = args.pin_memory,
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    
    display_data = [dev_data[i] for i in range(0, len(dev_data), len(dev_data) // 16)]
    display_loader = DataLoader(
        dataset=display_data,
        batch_size=1,
    )
    return train_loader, dev_loader, display_loader

def build_optim(args, params):
    optimizer = torch.optim.Adam(params, args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.RMSprop(params, args.lr, weight_decay=args.weight_decay)

    return optimizer
    
def train_epoch(args, epoch, model,data_loader, optimizer, writer):
    
    model.train()
    avg_loss = 0.0
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    #print ("Entering Train epoch")

    for iter, data in enumerate(tqdm(data_loader)):
        image, input_kspace, target,_ ,_ = data
        
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
        output,_ = model(input_kspace,image)
        loss = F.mse_loss(output,target)

        optimizer.zero_grad()
        
	    # For jupyter
	    # loss.backward()
        # For Fabric
        fabric.backward(loss)

        optimizer.step()

        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
        writer.add_scalar('TrainLoss',loss.item(),global_step + iter)

        if iter % args.report_interval == 10:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()

    return avg_loss, time.perf_counter() - start_epoch


def evaluate(epoch, model, data_loader, writer):

    model.eval()
    losses = []
    start = time.perf_counter()
    
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):
    
            image, input_kspace, target, _, _ = data # Return kspace also we can ignore that for train and test

            # For jupyter
            # image = image.unsqueeze(1).to(args.device)
            # target = target.unsqueeze(1).to(args.device)
            # input_kspace = input_kspace.to(args.device)
            
            #For Fabric
            image = image.unsqueeze(1)
            target = target.unsqueeze(1)

            input_kspace = input_kspace.permute(0,3,1,2)

            image = image.float()
            input_kspace = input_kspace.float()
            target = target.float()

            output, _ = model(input_kspace,image)
            loss = F.mse_loss(output,target)
            
            losses.append(loss.item())
            
        writer.add_scalar('Dev_Loss',np.mean(losses),epoch)
       
    return np.mean(losses), time.perf_counter() - start


def visualize(epoch, model, data_loader, writer):
    
    def save_image(image, tag):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):
            image,input_kspace,target, _, _ = data 

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
            
            output, _ = model(input_kspace,image)
            #print("input: ", torch.min(input), torch.max(input))
            #print("target: ", torch.min(target), torch.max(target))
            #print("predicted: ", torch.min(output), torch.max(output))
            #save_image(input_kspace, 'Input')
            
            image = image.cpu()
            target = target.cpu()
            output = output.cpu()
            save_image(target, 'Target')
            save_image(output, 'Reconstruction')
            save_image(torch.abs(target.float() - output.float()), 'Error')
            break

def build_dualencoderunet(args):
    #model = UnetModelParallelEncoder(
    #    in_chans=1,
    #    out_chans=1,
    #    chans=args.num_chans,
    #    num_pool_layers=args.num_pools,
    #    drop_prob=args.drop_prob
    #).to(args.device)

    model = UnetModelParallelEncoder(
        in_chans=1,
        out_chans=1,
        chans=args.num_chans,
        num_pool_layers=args.num_pools,
        drop_prob=args.drop_prob)
    
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
    
    #model = dAUTOMAP(model_params['input_shape'],model_params['output_shape'],model_params['tfx_params'],model_params['tfx_params2']).to(args.device)
    model = dAUTOMAP(model_params['input_shape'],model_params['output_shape'],model_params['tfx_params'],model_params['tfx_params2'])

    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    # model.load_state_dict(checkpoint['model'])
    return model

def build_model(args):
    dautomap_model = build_dautomap(args)
    dualencoderunet_model = build_dualencoderunet(args)
    #model = dAUTOMAPDualEncoderUnet(dautomap_model,dualencoderunet_model).to(args.device)
    model = dAUTOMAPDualEncoderUnet(dautomap_model,dualencoderunet_model)
    return model

def save_model(args, exp_dir, epoch, model, optimizer,best_dev_loss,is_new_best):
    out = torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir':exp_dir
        },
        f=exp_dir / 'model.pt'
    )

    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')

def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = build_model(args)
    
    if args.data_parallel:
        model = torch.nn.DataParallel(model)

    model.load_state_dict(checkpoint['model'])

    optimizer = build_optim(args,model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint, model, optimizer

def main(args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.exp_dir / 'summary'))
    
    if args.resume:
        print('resuming model , batch_size', args.batch_size)
        checkpoint,model,optimizer = load_model(args.checkpoint)
        args = checkpoint['args']
        args.batch_size = 28
        best_dev_mse = checkpoint['best_dev_mse']
        best_dev_ssim = checkpoint['best_dev_mse']
        start_epoch = checkpoint['epoch']
        del checkpoint
    else:
        model = build_model(args)
        print ("Model Built")
        if args.data_parallel:
            model = torch.nn.DataParallel(model)
            
        optimizer = build_optim(args,model.parameters())
    
        # For Fabric
        model, optimizer = fabric.setup(model, optimizer)
        
        print ("Optmizer initialized")
        best_dev_loss = 1e9
        start_epoch = 0
    
    logging.info(args)
    logging.info(model)
    
    train_loader, dev_loader, _ = create_data_loaders(args)
    # For Fabric
    train_loader = fabric.setup_dataloaders(train_loader)
    dev_loader = fabric.setup_dataloaders(dev_loader)
    print ("Dataloader initialized")
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)
    for epoch in range(start_epoch,args.num_epochs):
    
        print ("Entering into Training")
        train_loss,train_time = train_epoch(args, epoch, model, train_loader, optimizer, writer)
        print ("Entering into Evaluation")
        dev_loss,dev_time = evaluate(epoch, model, dev_loader, writer)
        #visualize(epoch, model, display_loader, writer)
        scheduler.step()
    
        is_new_best = dev_loss < best_dev_loss
        best_dev_loss = min(best_dev_loss,dev_loss)
        save_model(args, args.exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best)
        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}], TrainLoss = {train_loss:.4g}, '
            f'DevLoss= {dev_loss:.4g}, TrainTime = {train_time:.4f}s, DevTime = {dev_time:.4f}s',
        )
    writer.close()

def create_arg_parser():

    parser = argparse.ArgumentParser(description='Train setup for MR recon')
    # Stuff for Paths
    parser.add_argument('--train_path',default = '/Users/taylorfroelich/Code/Python/dualencoder/Data/Train_Fast/',type = pathlib.Path,help='Path to training data')
    parser.add_argument('--validation_path',default = '/Users/taylorfroelich/Code/Python/dualencoder/Data/Val_Fast',type = pathlib.Path,help='Path to validation data')
    parser.add_argument('--exp_dir',default = './checkpoints',type = pathlib.Path,help='Path to checkpoint dir')

    # Stuff for Datset
    parser.add_argument('--dataset',default = 'Fast', type=str, help = 'Cal, Fast, FastNoise')
    
    # Stuff to resume training
    parser.add_argument('--checkpoint',default = None, type=str, help = 'checkpoint location')
    parser.add_argument('--resume',default = False, type=bool,help = 'Boolan to control resume at checkpoint')
    
    # Stuff for Mask
    parser.add_argument('--acc_factor',default = 4, type = int, help = 'acceleration factor')
    parser.add_argument('--center_frac',default = 0.08, type = float, help = 'Center Fraction of k-space')
    parser.add_argument('--sample_rate',default = 0.95, type = float, help = ' Randomly Samples 60per of all avaiable data')
    
    # Stuff for NN
    parser.add_argument('--num_epochs',default = 100, type = int, help = 'number of epochs for training')
    parser.add_argument('--batch_size',default = 28, type = int, help = 'Batch size')

    parser.add_argument('--num_chans',default = 32, type = int, help = 'Number of U-Net channels')
    parser.add_argument('--num_pools',default = 4, type = int, help = 'Number of U-Net pooling layers')
    parser.add_argument('--drop_prob',default = 0.0, type = float, help = 'Dropout Probability, i.e. inference-time dropout rate')
    parser.add_argument('--patch_size',default = 256, type = int, help = 'Patch Size of NN (match image)')

    parser.add_argument('--lr',default = 0.001, type = float, help = 'learning rate')
    parser.add_argument('--lr_step_size',default = 40, type = int, help = 'Period of learning rate decay')
    parser.add_argument('--lr_gamma',default = 0.1, type = float, help = 'Multiplicative factor of learning rate decay')
    parser.add_argument('--weight_decay',default = 0.1, type = float, help = 'Strength of weight decay regularization')    
    parser.add_argument('--report_interval',default = 2, type = int, help = 'Period of loss reporting')    

    # Stuff for loaders and such
    parser.add_argument('--seed',default = 42, type = int, help = 'Random Seed')    
    
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
    args = create_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(args)
    
    fabric = L.Fabric(accelerator=args.device, devices=args.num_devices)
    fabric.launch()
    
    main(args)
