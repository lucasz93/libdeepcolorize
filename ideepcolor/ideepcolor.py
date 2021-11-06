import numpy as np
import cv2
import os, sys, time, multiprocessing
from threading import Thread, Lock
import multiprocessing

from libtiff import TIFF, tiff_h_4_3_0 as tiff_h
from libtiff.tif_lzw import encode as lzw_encode
from data import colorize_image as CI
import fasttiff

color_model = './models/pytorch/caffemodel.pth'
rgb_image = './imgs/E-056_N-02/Mars_Viking_ClrMosaic_global_E-056_N-02.tif'
test_image = './imgs/E-056_N-02/Murray-Lab_CTX-Mosaic_beta01_E-056_N-02.tif'
gpu_id = 0

def tif_encode_main(start, end, arr, encoded, mutex):
    print(f'{start} -> {end}')

    for _n in range(start, end):
        # Extract the row.
        row = arr[_n, :, :]

        # Delta encode the row (horizontal predictor).
        # np.diff computes diffs only -> doesn't contain the first element.
        # Need to prepend the first element to turn it into a delta encoding.
        diff = np.diff(row, axis=0)
        delta = np.insert(diff, 0, row[0, :], axis=0)

        e = lzw_encode(delta)
        with mutex:
            encoded[_n] = e

class Draw:
    def __init__(self, color_model, load_size):
        self.model = color_model
        self.load_size = load_size
        self.points = []

        self.suffix = 0

    def add_point(self, point):
        self.points += [point]
        
    def load_image(self, image_file):
        self.image_loaded = True
        self.image_file = image_file

        self.model.load_image(image_file)

    def set_image(self, image):
        self.model.set_image(image)
        self.image_loaded = True

    def get_input(self):
        h = self.load_size
        w = self.load_size
        im = np.zeros((h, w, 3), np.uint8)
        mask = np.zeros((h, w, 1), np.uint8)

        for p in self.points:
            p.render(im, mask)

        return im, mask

    def compute_result(self, rgb):
        if rgb.shape[0] != self.load_size or rgb.shape[1] != self.load_size:
            raise Exception(f'Input not expected size - is {rgb.shape}, expected {(self.load_size, self.load_size, 3)}!')
    
        #fasttiff.write_image_contig('/home/mechsoft/Documents/rgb.tif', rgb, rgb.shape[1], rgb.shape[0], rgb.shape[2], rgb.itemsize, tiff_h.SAMPLEFORMAT_UINT)

        # RGB has all pixels filled.
        im_mask0 = np.ones((1, self.load_size, self.load_size))
        start = time.perf_counter()
        im_lab = CI.rgb2lab_transpose(rgb)
        im_ab0 = im_lab[1:3, :, :]
        self.model.net_forward(im_ab0, im_mask0)

    def render(self):
        self.result = self.model.get_img_fullres()[:, :, ::-1]

    def save(self, filename):
        filename = f'/home/mechsoft/Documents/{filename}'
    
        arr = self.result
        shape = self.result.shape

        if arr.dtype in np.sctypes['float']:
            sample_format = tiff_h.SAMPLEFORMAT_IEEEFP
        elif arr.dtype in np.sctypes['uint'] + [bool]:
            sample_format = tiff_h.SAMPLEFORMAT_UINT
        elif arr.dtype in np.sctypes['int']:
            sample_format = tiff_h.SAMPLEFORMAT_INT
        elif arr.dtype in np.sctypes['complex']:
            sample_format = tiff_h.SAMPLEFORMAT_COMPLEXIEEEFP
        else:
            raise NotImplementedError(repr(arr.dtype))

        fasttiff.write_image_contig(filename, arr, shape[1], shape[0], shape[2], arr.itemsize, sample_format)

#####################################

print(f'Loading RGB...')
rgb0 = fasttiff.read_two_quarters_contig(rgb_image, 0)
rgb1 = fasttiff.read_two_quarters_contig(rgb_image, 1)
load_size = rgb0.shape[1]

print(f'Creating networks on GPU {gpu_id} and buffer size {load_size}...')
start = time.perf_counter()
colorModel = CI.ColorizeImageTorch(gpu_id, Xd=load_size)
colorModel.prep_net(path=color_model)
draw = Draw(colorModel, load_size)

print(f'Loading grayscale...')
g0 = fasttiff.read_two_quarters_contig(test_image, 0)
g1 = fasttiff.read_two_quarters_contig(test_image, 1)

####################################

print(f'Preprocessing UL...')
draw.set_image(g0[0, :, :, :])

print(f'Rendering UL...')
draw.compute_result(rgb0[0, :, :, :])
draw.render()

print(f'Saving...')
draw.save('ul.tif')

#---------------
print(f'Preprocessing UR...')
draw.set_image(g0[1, :, :, :])

print(f'Rendering UR...')
draw.compute_result(rgb0[1, :, :, :])
draw.render()

print(f'Saving...')
draw.save('ur.tif')

#---------------
print(f'Preprocessing LL...')
draw.set_image(g1[0, :, :, :])

print(f'Rendering LL...')
draw.compute_result(rgb1[0, :, :, :])
draw.render()

print(f'Saving...')
draw.save('ll.tif')

#---------------
print(f'Preprocessing LR...')
draw.set_image(g1[1, :, :, :])

print(f'Rendering LR...')
draw.compute_result(rgb1[1, :, :, :])
draw.render()

print(f'Saving...')
draw.save('lr.tif')

#---------------

