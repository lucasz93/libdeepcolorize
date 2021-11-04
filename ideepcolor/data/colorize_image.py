import numpy as np
import cv2
import matplotlib.pyplot as plt
import os, time
import torch
from libtiff import TIFF
from scipy.ndimage.interpolation import zoom

def rgb2lab_transpose(img_rgb):
    ''' INPUTS
            img_rgb XxXx3
        OUTPUTS
            returned value is 3xXxX '''
    lab = cv2.cvtColor(img_rgb.astype(np.float32) / 255., cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = np.float64(l)
    a = np.float64(a)
    b = np.float64(b)    
    return np.stack((l, a, b))

def cuda_rgb2l(shape, gpu_img_rgb):
    ''' INPUTS
            img_rgb XxYx3   CV_8UC3
        OUTPUTS
            returned value is 3xXxY CV_32FC3'''
    # Scale into [0..1]
    scratch = cv2.cuda_GpuMat(shape[0], shape[1], cv2.CV_32FC3)
    gpu_img_rgb.convertTo(scratch.type(), 1. / 255., scratch)

    # Convert to LAB
    cv2.cuda.cvtColor(scratch, cv2.COLOR_RGB2LAB, scratch)

    # Extract the L component.
    gpu_l = cv2.cuda_GpuMat(shape[0], shape[1], cv2.CV_32FC1)
    cv2.cuda.split(scratch, (gpu_l, None, None))
    return gpu_l


class ColorizeImageBase():
    def __init__(self, Xd=256):
        self.Xd = Xd
        self.img_l_set = False
        self.net_set = False
        self.img_just_set = False  # this will be true whenever image is just loaded
        # net_forward can set this to False if they want

    def prep_net(self):
        raise Exception("Should be implemented by base class")

    # ***** Image prepping *****
    def load_image(self, input_path):
        # rgb image [CxXdxXd]
        tif = TIFF.open(input_path, mode='r')
        try:
            im = cv2.cvtColor(tif.read_image(), cv2.COLOR_GRAY2RGB)
            self.set_image(im)
        finally:
            tif.close()

    def set_image(self, im):
        gpu_im = cv2.cuda_GpuMat(im)
        self._set_img_lab_fullres_(im.shape, gpu_im)

        # convert into lab space
        im = cv2.resize(im, (self.Xd, self.Xd))
        self._set_img_lab_(im)

    def net_forward(self, input_ab, input_mask):
        # INPUTS
        #     ab       2xXxX    input color patches (non-normalized)
        #     mask     1xXxX    input mask, indicating which points have been provided
        # assumes self.img_l_mc has been set

        if(not self.img_l_set):
            print('I need to have an image!')
            return -1
        if(not self.net_set):
            print('I need to have a net!')
            return -1

        self.input_ab_mc = (input_ab - self.ab_mean) / self.ab_norm
        self.input_mask_mult = input_mask * self.mask_mult
        return 0

    def get_img_fullres(self):
        # This assumes self.img_l_fullres, self.output_ab are set.
        # Typically, this means that set_image() and net_forward()
        # have been called.
        # bilinear upsample
        
        a = self.output_ab[0, :, :]
        b = self.output_ab[1, :, :]
        
        full_shape = self.gpu_img_l_fullres_shape
        size = (int(self.output_ab.shape[1] * 1. * full_shape[0] / self.output_ab.shape[1]), int(self.output_ab.shape[2] * 1. * full_shape[1] / self.output_ab.shape[2]))
 
        gpu_a_fullsize = cv2.cuda_GpuMat(full_shape[0], full_shape[1], cv2.CV_32FC1)
        gpu_b_fullsize = cv2.cuda_GpuMat(full_shape[0], full_shape[1], cv2.CV_32FC1)
        
        cv2.cuda.resize(cv2.cuda_GpuMat(a), size, gpu_a_fullsize, interpolation=cv2.INTER_CUBIC)
        cv2.cuda.resize(cv2.cuda_GpuMat(b), size, gpu_b_fullsize, interpolation=cv2.INTER_CUBIC)

        # Allocate output memory for the merge...otherwise OpenCV downloads it to the CPU.
        # Merge components back together.
        scratch = cv2.cuda_GpuMat(full_shape[0], full_shape[1], cv2.CV_32FC3)
        cv2.cuda.merge((self.gpu_img_l_fullres, gpu_a_fullsize, gpu_b_fullsize), scratch)
        
        # Don't need these anymore. Keep memory usage down.
        del gpu_a_fullsize
        del gpu_b_fullsize

        # Convert to RGB.
        cv2.cuda.cvtColor(scratch, cv2.COLOR_LAB2RGB, scratch)

        # Scale into [0..255].
        gpu_byte_rgb = cv2.cuda_GpuMat(full_shape[0], full_shape[1], cv2.CV_8UC3)
        scratch.convertTo(gpu_byte_rgb.type(), 255., gpu_byte_rgb)

        # Download into a C array.
        return gpu_byte_rgb.download()

    # ***** Private functions *****
    def _set_img_lab_fullres_(self, shape, gpu_img_rgb_fullres):
        # INPUTS
        #     gpu_img_rgb_fullres    XxYx3    CV_U8C3
        self.gpu_img_l_fullres_shape = [shape[0], shape[1]]
        self.gpu_img_l_fullres = cuda_rgb2l(self.gpu_img_l_fullres_shape, gpu_img_rgb_fullres)

    def _set_img_lab_(self, img_rgb):
        img_lab = rgb2lab_transpose(img_rgb)

        # lab image, mean centered [XxYxX]
        img_lab_mc = img_lab / np.array((self.l_norm, self.ab_norm, self.ab_norm))[:, np.newaxis, np.newaxis] - np.array(
            (self.l_mean / self.l_norm, self.ab_mean / self.ab_norm, self.ab_mean / self.ab_norm))[:, np.newaxis, np.newaxis]

        self.img_l_mc = img_lab_mc[[0], :, :]
        self.img_l_set = True


class ColorizeImageTorch(ColorizeImageBase):
    def __init__(self, gpu_id, Xd=256, maskcent=False):
        print('ColorizeImageTorch instantiated')
        ColorizeImageBase.__init__(self, Xd)
        self.gpu_id = gpu_id
        self.l_norm = 1.
        self.ab_norm = 1.
        self.l_mean = 50.
        self.ab_mean = 0.
        self.mask_mult = 1.
        self.mask_cent = .5 if maskcent else 0
        
        cv2.cuda.setDevice(gpu_id)
        torch.cuda.set_device(gpu_id)

    # ***** Net preparation *****
    def prep_net(self, path='', dist=False):
        import torch
        import models.pytorch.model as model
        self.net = model.SIGGRAPHGenerator(self.gpu_id, dist)
        state_dict = torch.load(path)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        self.net.load_state_dict(state_dict)
        if self.gpu_id != None:
            print(f'Running on CUDA:{self.gpu_id}')
            self.net.cuda(self.gpu_id)
        else:
            print('Running on CPU')

        self.net.eval()
        self.net_set = True

    # ***** Call forward *****
    def net_forward(self, input_ab, input_mask):
        # INPUTS
        #     ab       2xXxX    input color patches (non-normalized)
        #     mask     1xXxX    input mask, indicating which points have been provided
        # assumes self.img_l_mc has been set

        if ColorizeImageBase.net_forward(self, input_ab, input_mask) == -1:
            return -1

        self.output_ab = self.net.forward(self.img_l_mc, self.input_ab_mc, self.input_mask_mult, self.mask_cent)[0, :, :, :].cpu().data.numpy()
        return 0


