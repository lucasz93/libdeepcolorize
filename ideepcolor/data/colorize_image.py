import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from scipy.ndimage.interpolation import zoom

def lab2rgb_transpose(img_l, img_ab):
    ''' INPUTS
            img_l     1xXxX
            img_ab    2xXxX
        OUTPUTS
            returned value is XxXx3 '''
    l = np.float32(img_l[0,:,:])
    a = np.float32(img_ab[0,:,:])
    b = np.float32(img_ab[1,:,:])
    pred_lab = cv2.merge((l, a, b))
    
    return (cv2.cvtColor(pred_lab, cv2.COLOR_LAB2RGB) * 255.).astype('uint8')


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
        self._set_img_lab_fullres_(im)

        # convert into lab space
        im = cv2.resize(im, (self.Xd, self.Xd))
        self._set_img_lab_(im)
        self._set_img_lab_mc_()

    def set_image(self, input_image):
        self._set_img_lab_fullres_(input_image)

        # convert into lab space
        self._set_img_lab_(input_image)
        self._set_img_lab_mc_()

    def net_forward(self, input_ab, input_mask):
        # INPUTS
        #     ab         2xXxX     input color patches (non-normalized)
        #     mask     1xXxX    input mask, indicating which points have been provided
        # assumes self.img_l_mc has been set

        if(not self.img_l_set):
            print('I need to have an image!')
            return -1
        if(not self.net_set):
            print('I need to have a net!')
            return -1

        self.input_ab = input_ab
        self.input_ab_mc = (input_ab - self.ab_mean) / self.ab_norm
        self.input_mask = input_mask
        self.input_mask_mult = input_mask * self.mask_mult
        return 0

    def get_img_fullres(self):
        # This assumes self.img_l_fullres, self.output_ab are set.
        # Typically, this means that set_image() and net_forward()
        # have been called.
        # bilinear upsample

        print('START SLOW POINT A')
        #
        # TODO: Build OpenCV with CUDA support - https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7
        #
        zoom_factor = (1, 1. * self.img_l_fullres.shape[1] / self.output_ab.shape[1], 1. * self.img_l_fullres.shape[2] / self.output_ab.shape[2])
        output_ab_fullres = zoom(self.output_ab, zoom_factor, order=1)
        
        #size = (int(self.output_ab.shape[1] * 1. * self.img_l_fullres.shape[1] / self.output_ab.shape[1]), int(self.output_ab.shape[2] * 1. * self.img_l_fullres.shape[2] / self.output_ab.shape[2]))
        #gpu_ab, output_ab_fullres = cv2.cuda_GpuMat(), cv2.mat()
        #gpu_ab.upload(np.float32(self.output_ab))
        #cv2.cuda.resize(gpu_ab, size).download(output_ab_fullres)
        
        print('END SLOW POINT A')
        
        return lab2rgb_transpose(self.img_l_fullres, output_ab_fullres)

    def get_sup_img(self):
        return lab2rgb_transpose(50 * self.input_mask, self.input_ab)

    # ***** Private functions *****
    def _set_img_lab_fullres_(self, img_rgb_fullres):
        img_lab_fullres = rgb2lab_transpose(img_rgb_fullres)
        self.img_l_fullres = img_lab_fullres[[0], :, :]
        self.img_ab_fullres = img_lab_fullres[1:, :, :]

    def _set_img_lab_(self, img_rgb):
        # set self.img_lab from self.im_rgb
        self.img_lab = rgb2lab_transpose(img_rgb)
        self.img_l = self.img_lab[[0], :, :]
        self.img_ab = self.img_lab[1:, :, :]

    def _set_img_lab_mc_(self):
        # set self.img_lab_mc from self.img_lab
        # lab image, mean centered [XxYxX]
        self.img_lab_mc = self.img_lab / np.array((self.l_norm, self.ab_norm, self.ab_norm))[:, np.newaxis, np.newaxis] - np.array(
            (self.l_mean / self.l_norm, self.ab_mean / self.ab_norm, self.ab_mean / self.ab_norm))[:, np.newaxis, np.newaxis]
        self._set_img_l_()

    def _set_img_l_(self):
        self.img_l_mc = self.img_lab_mc[[0], :, :]
        self.img_l_set = True

    def _set_img_ab_(self):
        self.img_ab_mc = self.img_lab_mc[[1, 2], :, :]

    def _set_out_ab_(self):
        self.output_lab = rgb2lab_transpose(self.output_rgb)
        self.output_ab = self.output_lab[1:, :, :]


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

        # Load grid properties
        self.pts_in_hull = np.array(np.meshgrid(np.arange(-110, 120, 10), np.arange(-110, 120, 10))).reshape((2, 529)).T

    # ***** Net preparation *****
    def prep_net(self, gpu_id=None, path='', dist=False):
        import torch
        import models.pytorch.model as model
        print('path = %s' % path)
        print('Model set! dist mode? ', dist)
        self.net = model.SIGGRAPHGenerator(dist=dist)
        state_dict = torch.load(path)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        # patch InstanceNorm checkpoints prior to 0.4
        for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
            self.__patch_instance_norm_state_dict(state_dict, self.net, key.split('.'))
        self.net.load_state_dict(state_dict)
        if gpu_id != None:
            self.net.cuda()
        self.net.eval()
        self.net_set = True

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    # ***** Call forward *****
    def net_forward(self, input_ab, input_mask):
        # INPUTS
        #     ab         2xXxX     input color patches (non-normalized)
        #     mask     1xXxX    input mask, indicating which points have been provided
        # assumes self.img_l_mc has been set

        if ColorizeImageBase.net_forward(self, input_ab, input_mask) == -1:
            return -1
            
        # FIXME: This seems terribly wasteful. Converts AB -> RGB -> AB
        output_ab = self.net.forward(self.img_l_mc, self.input_ab_mc, self.input_mask_mult, self.mask_cent)[0, :, :, :].cpu().data.numpy()
        self.output_rgb = lab2rgb_transpose(self.img_l, output_ab)        
        self._set_out_ab_()
        return self.output_rgb


