import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from scipy.ndimage.interpolation import zoom

def cuda_lab2rgb_transpose(shape, gpu_img_l, gpu_img_a, gpu_img_b):
    ''' INPUTS
            img_l     1xXxX
            img_ab    2xXxX
        OUTPUTS
            returned value is XxXx3 '''
    # Allocate output memory for the merge...otherwise OpenCV downloads it to the CPU.
    gpu_pred_lab = cv2.cuda_GpuMat(shape[0], shape[1], cv2.CV_32FC3)
    cv2.cuda.merge((gpu_img_l, gpu_img_a, gpu_img_b), gpu_pred_lab)

    # Convert to RGB.
    gpu_float_rgb = cv2.cuda_GpuMat(shape[0], shape[1], cv2.CV_32FC3)
    cv2.cuda.cvtColor(gpu_pred_lab, cv2.COLOR_LAB2RGB, gpu_float_rgb)

    # Scale into [0..255].
    gpu_byte_rgb = cv2.cuda_GpuMat(shape[0], shape[1], cv2.CV_8UC3)
    gpu_float_rgb.convertTo(gpu_byte_rgb.type(), 255., gpu_byte_rgb)
    
    return gpu_byte_rgb.download()


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
    gpu_float_rgb = cv2.cuda_GpuMat(shape[0], shape[1], cv2.CV_32FC3)
    gpu_img_rgb.convertTo(gpu_float_rgb.type(), 1. / 255., gpu_float_rgb)

    # Convert to LAB
    gpu_lab = cv2.cuda_GpuMat(shape[0], shape[1], cv2.CV_32FC3)
    cv2.cuda.cvtColor(gpu_float_rgb, cv2.COLOR_RGB2LAB, gpu_lab)

    # Extract the L component.
    gpu_l = cv2.cuda_GpuMat(shape[0], shape[1], cv2.CV_32FC1)
    cv2.cuda.split(gpu_lab, (gpu_l, None, None))
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
        im = cv2.cvtColor(cv2.imread(input_path, 1), cv2.COLOR_BGR2RGB)
        self.set_image(im)

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
        
        size = (int(self.output_ab.shape[1] * 1. * self.gpu_img_l_fullres_shape[0] / self.output_ab.shape[1]), int(self.output_ab.shape[2] * 1. * self.gpu_img_l_fullres_shape[1] / self.output_ab.shape[2]))
        gpu_a_fullsize = cv2.cuda.resize(cv2.cuda_GpuMat(a), size, interpolation=cv2.INTER_CUBIC)
        gpu_b_fullsize = cv2.cuda.resize(cv2.cuda_GpuMat(b), size, interpolation=cv2.INTER_CUBIC)

        return cuda_lab2rgb_transpose(self.gpu_img_l_fullres_shape, self.gpu_img_l_fullres, gpu_a_fullsize, gpu_b_fullsize)

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
    def __init__(self, Xd=256, maskcent=False):
        print('ColorizeImageTorch instantiated')
        ColorizeImageBase.__init__(self, Xd)
        self.l_norm = 1.
        self.ab_norm = 1.
        self.l_mean = 50.
        self.ab_mean = 0.
        self.mask_mult = 1.
        self.mask_cent = .5 if maskcent else 0

    # ***** Net preparation *****
    def prep_net(self, gpu_id=None, path='', dist=False):
        import torch
        import models.pytorch.model as model
        self.net = model.SIGGRAPHGenerator(dist=dist)
        state_dict = torch.load(path)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        self.net.load_state_dict(state_dict)
        if gpu_id != None:
            self.net.cuda()
        self.net.eval()
        self.net_set = True

    # ***** Call forward *****
    def net_forward(self, input_ab, input_mask):
        # INPUTS
        #     ab         2xXxX     input color patches (non-normalized)
        #     mask     1xXxX    input mask, indicating which points have been provided
        # assumes self.img_l_mc has been set

        if ColorizeImageBase.net_forward(self, input_ab, input_mask) == -1:
            return -1
            
        self.output_ab = self.net.forward(self.img_l_mc, self.input_ab_mc, self.input_mask_mult, self.mask_cent)[0, :, :, :].cpu().data.numpy()
        return 0


