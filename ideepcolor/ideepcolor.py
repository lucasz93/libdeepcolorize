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
rgb_image = './imgs/E-056_N-02/Mars_Viking_ClrMosaic_global_925m-E-056_N-02.tif'
test_image = './imgs/E-056_N-02/Murray-Lab_CTX-Mosaic_beta01_E-056_N-02-medium.tif'
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

    def read_image(self, image_file):
        self.image_loaded = True
        self.image_file = image_file

        self.model.load_image(image_file)

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
            raise Exception('Input not expected size!')

        # RGB has all pixels filled.
        im_mask0 = np.ones((1, self.load_size, self.load_size))
        start = time.perf_counter()
        im_lab = CI.rgb2lab_transpose(rgb)
        im_ab0 = im_lab[1:3, :, :]
        self.model.net_forward(im_ab0, im_mask0)

    def render(self):
        self.result = self.model.get_img_fullres()[:, :, ::-1]

    def save_result(self):
        path = os.path.abspath(self.image_file)
        path, ext = os.path.splitext(path)

        save_path = "_".join([path, str(self.suffix)])
        self.suffix += 1

        #print('  saving result to <%s>\n' % save_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

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

        fasttiff.write_image_contig(os.path.join(save_path, 'ours_fullres.tif'), arr, shape[1], shape[0], shape[2], arr.itemsize, sample_format)

############### SETUP ###############

print(f'Loading top half')
start = time.perf_counter()
g0 = fasttiff.read_two_quarters_contig('./imgs/E-056_N-02/Murray-Lab_CTX-Mosaic_beta01_E-056_N-02.tif', 0)
print(f'Took {time.perf_counter() - start} seconds\n')

print(f'Loading bottom half')
start = time.perf_counter()
g1 = fasttiff.read_two_quarters_contig('./imgs/E-056_N-02/Murray-Lab_CTX-Mosaic_beta01_E-056_N-02.tif', 1)
print(f'Took {time.perf_counter() - start} seconds\n')

#fasttiff.write_image_contig('test_ul.tif', g0[0, :, :, :], g0.shape[1], g0.shape[2], g0.shape[3], g0.itemsize, tiff_h.SAMPLEFORMAT_UINT)
#fasttiff.write_image_contig('test_ur.tif', g0[1, :, :, :], g0.shape[1], g0.shape[2], g0.shape[3], g0.itemsize, tiff_h.SAMPLEFORMAT_UINT)
#fasttiff.write_image_contig('test_ll.tif', g1[0, :, :, :], g1.shape[1], g1.shape[2], g1.shape[3], g1.itemsize, tiff_h.SAMPLEFORMAT_UINT)
#fasttiff.write_image_contig('test_lr.tif', g1[1, :, :, :], g1.shape[1], g1.shape[2], g1.shape[3], g1.itemsize, tiff_h.SAMPLEFORMAT_UINT)

print(f'Writing whole thing')
start = time.perf_counter()
fasttiff.stitch_and_write_quarters_contig('test.tif', g0, g1, g0.shape[1] * 2, g0.shape[2] * 2, g0.shape[3], g0.itemsize, tiff_h.SAMPLEFORMAT_UINT)
print(f'Took {time.perf_counter() - start} seconds\n')

exit()

print(f'Loading RGB {test_image}')
start = time.perf_counter()
tif = TIFF.open(rgb_image, mode='r')
rgb = tif.read_image().astype('uint8')
h, w, c = rgb.shape
if w != h:
    raise Exception('w != h')
load_size = h
print(f'Took {time.perf_counter() - start} seconds\n')

print('Creating networks...')
start = time.perf_counter()
colorModel = CI.ColorizeImageTorch(gpu_id, Xd=load_size)
colorModel.prep_net(path=color_model)
print(f'Took {time.perf_counter() - start} seconds\n')

print(f'Loading grayscale {test_image}')
start = time.perf_counter()
draw = Draw(colorModel, load_size)
draw.read_image(test_image)
print(f'Took {time.perf_counter() - start} seconds\n')

print('Computing (cold)...')
start = time.perf_counter()
draw.compute_result(rgb)
print(f'Took {time.perf_counter() - start} seconds\n')

print('Computing (warm)...')
start = time.perf_counter()
draw.compute_result(rgb)
print(f'Took {time.perf_counter() - start} seconds\n')

print('Rendering...')
start = time.perf_counter()
draw.render()
print(f'Took {time.perf_counter() - start} seconds\n')

print('Saving...')
start = time.perf_counter()
draw.save_result()
print(f'Took {time.perf_counter() - start} seconds\n')

