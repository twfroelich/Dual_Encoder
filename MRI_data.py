import pathlib
import random
from torch.utils.data import Dataset
import torch
import numpy as np
from pathlib import Path
import h5py
import cv2

def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.

    Args:
        data (np.array): Input numpy array

    Returns:
        torch.Tensor: PyTorch version of data
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)
    return torch.from_numpy(data)

def apply_mask(data, mask_func, seed=None):
    """
    Subsample given k-space by multiplying with a mask.

    Args:
        data (torch.Tensor): The input k-space data. This should have at least 3 dimensions, where
            dimensions -3 and -2 are the spatial dimensions, and the final dimension has size
            2 (for complex values).
        mask_func (callable): A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed (int or 1-d array_like, optional): Seed for the random number generator.

    Returns:
        (tuple): tuple containing:
            masked data (torch.Tensor): Subsampled k-space data
            mask (torch.Tensor): The generated mask
    """ 

    shape = np.array(data.shape)
    shape[:-3] = 1
    # print("mask_func",mask_func)
    mask = mask_func(shape, seed)
    return torch.where(mask == 0, torch.Tensor([0]), data), mask

INTER_MODE = {'NEAREST': cv2.INTER_NEAREST, 'BILINEAR': cv2.INTER_LINEAR, 'BICUBIC': cv2.INTER_CUBIC}

def resize(img, size, interpolation='BILINEAR'):
    """Resize the input CV Image to the given size.

    Args:
        img (np.ndarray): Image to be resized.
        size (tuple or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (str, optional): Desired interpolation. Default is ``BILINEAR``

    Returns:
        cv Image: Resized image.
    """
    if isinstance(size, int):
        h, w, c = img.shape
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return cv2.resize(img, dsize=(ow, oh), interpolation=INTER_MODE[interpolation])
        else:
            oh = size
            ow = int(size * w / h)
            return cv2.resize(img, dsize=(ow, oh), interpolation=INTER_MODE[interpolation])
    else:
        oh, ow = size
        return cv2.resize(img, dsize=(int(ow), int(oh)), interpolation=INTER_MODE[interpolation])


class MaskFunc:
    """
    MaskFunc creates a sub-sampling mask of a given shape.

    The mask selects a subset of columns from the input k-space data. If the k-space data has N
    columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center corresponding to
           low-frequencies
        2. The other columns are selected uniformly at random with a probability equal to:
           prob = (N / acceleration - N_low_freqs) / (N - N_low_freqs).
    This ensures that the expected number of columns selected is equal to (N / acceleration)

    It is possible to use multiple center_fractions and accelerations, in which case one possible
    (center_fraction, acceleration) is chosen uniformly at random each time the MaskFunc object is
    called.

    For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04], then there
    is a 50% probability that 4-fold acceleration with 8% center fraction is selected and a 50%
    probability that 8-fold acceleration with 4% center fraction is selected.
    """

    def __init__(self, center_fractions, accelerations):
        """
        Args:
            center_fractions (List[float]): Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is chosen uniformly
                each time.

            accelerations (List[int]): Amount of under-sampling. This should have the same length
                as center_fractions. If multiple values are provided, then one of these is chosen
                uniformly each time. An acceleration of 4 retains 25% of the columns, but they may
                not be spaced evenly.
        """
        if len(center_fractions) != len(accelerations):
            raise ValueError('Number of center fractions should match number of accelerations')

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.rng = np.random.RandomState()

    def __call__(self, shape, seed=None):
        """
        Args:
            shape (iterable[int]): The shape of the mask to be created. The shape should have
                at least 3 dimensions. Samples are drawn along the second last dimension.
            seed (int, optional): Seed for the random number generator. Setting the seed
                ensures the same mask is generated each time for the same shape.
        Returns:
            torch.Tensor: A mask of the specified shape.
        """
        
        if len(shape) < 3:
            raise ValueError('Shape should have 3 or more dimensions')
        
        # print("shape=",shape)
        self.rng.seed(seed)
        num_cols = shape[-2]

        choice = self.rng.randint(0, len(self.accelerations))
        center_fraction = self.center_fractions[choice]
        acceleration = self.accelerations[choice]

        # Create the mask
        num_low_freqs = int(round(num_cols * center_fraction))
        prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)
        mask = self.rng.uniform(size=num_cols) < prob
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad:pad + num_low_freqs] = True

        # Reshape the mask
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols
        mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

        return mask

class SliceData(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """
    def __init__(self, root,center_frac, acc_factor, sample_rate): # acc_factor can be passed here and saved as self variable
        files = list(pathlib.Path(root).iterdir())
        self.examples = []
        self.acc_factor = acc_factor # 4
        self.center_frac = center_frac

        self.sample_rate = sample_rate #   0.6
        
        random.shuffle(files)
        num_files = round(len(files) * self.sample_rate)
        files = files[:num_files]

        for fname in sorted(files):
            kspace = np.load(fname)
            num_slices = kspace.shape[0]
            self.examples += [(fname, slice) for slice in range(0,num_slices)]   #20 20 middle slices 20 in from edges


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # Index the fname and slice using the list created in __init__
        
        fname, slice = self.examples[i] 

        data = np.load(fname)
        kspace = data[slice]
        kspace_cplx = kspace[:,:,0] + 1j*kspace[:,:,1]
        #kspace_cplx = torch.complex(kspace[:,:,0],kspace[:,:,1])

        kspace = np.fft.fftshift(kspace_cplx)
        target = np.fft.ifft2(kspace_cplx)
        target_abs = np.abs(target)
        target_abs = target_abs/np.max(target_abs)
        
        kspace_cmplx = np.fft.fftshift(np.fft.fft2(target_abs,norm='ortho'))
        kspace = to_tensor(kspace_cmplx)
        
        # center fraction, acceleration factor (0.08 i.e 8%, 4 x)
        mask_func = MaskFunc([self.center_frac], [self.acc_factor])
       
        seed =  tuple(map(ord, str(fname)))
        
        masked_kspace, mask = apply_mask(kspace.float(),mask_func,seed)
        
        masked_kspace_np = masked_kspace[:,:,0].numpy() + 1j*masked_kspace[:,:,1].numpy()
        us_img = np.abs(np.fft.ifft2(masked_kspace_np))            
       
        fname = Path(fname)    
        return us_img, masked_kspace , target_abs , str(fname.name) , slice

class SliceDataFast(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """
    def __init__(self, root,center_frac, acc_factor, sample_rate, patch_size): # acc_factor can be passed here and saved as self variable
        files = list(pathlib.Path(root).iterdir())
        self.examples = []
        self.acc_factor = acc_factor # 4
        self.center_frac = center_frac
        self.patch_size = patch_size

        self.sample_rate = sample_rate #   0.6
        
        random.shuffle(files)
        num_files = round(len(files) * self.sample_rate)
        files = files[:num_files]

        for fname in sorted(files):
            with h5py.File(fname, "r") as hf:
                kspace = np.fft.ifftn(hf["reconstruction_rss"][:],axes=(1,2))
            kspace = np.stack((kspace.real, kspace.imag), axis=-1)

            num_slices = kspace.shape[0]
            self.examples += [(fname, slice) for slice in range(0,num_slices)]   #20 20 middle slices 20 in from edges


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # Index the fname and slice using the list created in __init__
        fname, slice = self.examples[i] 
        
        with h5py.File(fname, "r") as hf:
            recon = hf["reconstruction_rss"][:]
        recon_shape1,_,_ = recon.shape

        new_shape = (recon_shape1,self.patch_size,self.patch_size)
        recon_resize = np.empty(new_shape)
        for slice_index in range(recon_shape1):
            recon_resize[slice_index,:,:] = resize(recon[slice_index,:,:],(self.patch_size,self.patch_size))

        data = np.fft.ifftn(recon_resize,axes=(1,2))
        data = np.stack((data.real, data.imag), axis=-1)

        kspace = data[slice]
        kspace_cplx = kspace[:,:,0] + 1j*kspace[:,:,1]
        
        kspace = np.fft.fftshift(kspace_cplx)
        target = np.fft.ifft2(kspace_cplx)
        target_abs = np.abs(target)
        target_abs = target_abs/np.max(target_abs)
        
        kspace_cmplx = np.fft.fftshift(np.fft.fft2(target_abs,norm='ortho'))
        kspace = to_tensor(kspace_cmplx)
        
        # center fraction, acceleration factor (0.08 i.e 8%, 4 x)
        mask_func = MaskFunc([self.center_frac], [self.acc_factor])
       
        seed =  tuple(map(ord, str(fname)))
        
        masked_kspace, mask = apply_mask(kspace.float(),mask_func,seed)
        
        masked_kspace_np = masked_kspace[:,:,0].numpy() + 1j*masked_kspace[:,:,1].numpy()
        us_img = np.abs(np.fft.ifft2(masked_kspace_np))            
       
        fname = Path(fname)    
        return np.float32(us_img), masked_kspace.to(torch.float32) , np.float32(target_abs) , str(fname.name) , slice
    
class SliceDataFastNoise(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """
    def __init__(self, root,center_frac, acc_factor, sample_rate, patch_size): # acc_factor can be passed here and saved as self variable
        files = list(pathlib.Path(root).iterdir())
        self.examples = []
        self.acc_factor = acc_factor # 4
        self.center_frac = center_frac
        self.patch_size = patch_size

        self.sample_rate = sample_rate #   0.6
        
        random.shuffle(files)
        num_files = round(len(files) * self.sample_rate)
        files = files[:num_files]

        for fname in sorted(files):
            with h5py.File(fname, "r") as hf:
                kspace = np.fft.ifftn(hf["reconstruction_rss"][:],axes=(1,2))
            kspace = np.stack((kspace.real, kspace.imag), axis=-1)

            num_slices = kspace.shape[0]
            self.examples += [(fname, slice) for slice in range(0,num_slices)]   #20 20 middle slices 20 in from edges


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # Index the fname and slice using the list created in __init__
        
        fname, slice = self.examples[i] 
        
        with h5py.File(fname, "r") as hf:
            recon = hf["reconstruction_rss"][:]
        recon_shape1,_,_ = recon.shape

        new_shape = (recon_shape1,self.patch_size,self.patch_size)
        recon_resize = np.empty(new_shape)
        for slice_index in range(recon_shape1):
            recon_resize[slice_index,:,:] = resize(recon[slice_index,:,:],(self.patch_size,self.patch_size))

        data = np.fft.ifftn(recon_resize,axes=(1,2))
        data = np.stack((data.real, data.imag), axis=-1)

        kspace = data[slice]
        kspace_cplx = kspace[:,:,0] + 1j*kspace[:,:,1]

        kspace = np.fft.fftshift(kspace_cplx)
        target = np.fft.ifft2(kspace_cplx)
        target_abs = np.abs(target)
        target_abs = target_abs/np.max(target_abs)
        
        
        kspace_cmplx = np.fft.fftshift(np.fft.fft2(target_abs,norm='ortho'))

        # Random Shift
                # The size of the matrix.
        N, M = kspace_cmplx.shape

        r = np.round(N + (-N-N) * np.random.rand(1,1))

        shift_dir = torch.randn(1)[0]
        if shift_dir > 0:
            delta = [r, 0]
        else:
            delta = [0, r]

        # The mathsy bit. The floors take care of odd-length signals.
        x_shift = np.exp(-1j * 2 * np.pi * delta[0] * np.concatenate((np.arange(0, np.floor(N/2)), np.arange(-np.floor(N/2), 0))) / N)
        y_shift = np.exp(-1j * 2 * np.pi * delta[1] * np.concatenate((np.arange(0, np.floor(M/2)), np.arange(-np.floor(M/2), 0))) / M)

        if shift_dir > 0:
            if N % 2 == 0:
                x_shift[:,N//2] = np.real(x_shift[:,N//2])
            if M % 2 == 0:
                y_shift[M//2] = np.real(y_shift[M//2])
        else:
            if N % 2 == 0:
                x_shift[N//2] = np.real(x_shift[N//2])
            if M % 2 == 0:
                y_shift[:,M//2] = np.real(y_shift[:,M//2])
        
        kspace_cmplx = (kspace_cmplx * np.outer(x_shift, y_shift)) + (np.random.rand(1)[0]/10 * (np.random.rand(N,M) + 1j*np.random.rand(N,M)))

        kspace = to_tensor(kspace_cmplx)
        # center fraction, acceleration factor (0.08 i.e 8%, 4 x)
        mask_func = MaskFunc([self.center_frac], [self.acc_factor])
       
        seed =  tuple(map(ord, str(fname)))
        
        masked_kspace, mask = apply_mask(kspace.float(),mask_func,seed)
        
        masked_kspace_np = masked_kspace[:,:,0].numpy() + 1j*masked_kspace[:,:,1].numpy()
        us_img = np.abs(np.fft.ifft2(masked_kspace_np))
       
        fname = Path(fname)    
        return np.float32(us_img), masked_kspace.to(torch.float32) , np.float32(target_abs) , str(fname.name) , slice
    
class SliceDataFastTest(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """
    def __init__(self, root, center_frac, acc_factor, sample_rate, patch_size): # acc_factor can be passed here and saved as self variable
        files = list(pathlib.Path(root).iterdir())
        self.examples = []
        self.acc_factor = acc_factor # 4
        self.center_frac = center_frac
        self.patch_size = patch_size
        self.sample_rate = sample_rate #   0.6
        
        random.shuffle(files)
        num_files = round(len(files) * self.sample_rate)
        files = files[:num_files]

        for fname in sorted(files):
            with h5py.File(fname, "r") as hf:
                kspace = hf["kspace"][:]
            kspace = np.transpose(kspace,[1,0,2,3])      # Convert from numpy array to pytorch tensor

            num_slices = kspace.shape[1]
            self.examples += [(fname, slice) for slice in range(0,num_slices)]   #20 20 middle slices 20 in from edges


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # Index the fname and slice using the list created in __init__
        
        fname, slice = self.examples[i] 

        with h5py.File(fname, "r") as hf:
            data = hf["kspace"][:]

        k_space_coils = np.transpose(data,[1,0,2,3])      # Convert from numpy array to pytorch tensor
        image_coils = np.abs(np.fft.ifftshift(np.fft.fft2(k_space_coils,axes=(2,3)),axes=(2,3)))

        coils,slices,_,_ = image_coils.shape
        
        new_shape = (coils,slices,self.patch_size,self.patch_size)
        recon_resize = np.empty(new_shape)
        for coil_index in range(coils):
            for  slice_index in range(slices):
                recon_resize[coil_index,slice_index,:,:] = resize(image_coils[coil_index,slice_index,:,:],(self.patch_size,self.patch_size))
                
        recon_resize = np.sqrt((recon_resize ** 2).sum(0))
        recon_resize = recon_resize[slice,:,:]

        image_out = (recon_resize/np.max(recon_resize))
        k_space_out = np.fft.fftshift(np.fft.ifft2(recon_resize,axes=(0,1)),axes=(0,1))
        k_space_out = np.stack((k_space_out.real, k_space_out.imag), axis=-1)
        k_space_out = torch.from_numpy(k_space_out)
       
        fname = Path(fname)    
        return recon_resize, k_space_out, image_out, str(fname.name) , slice 