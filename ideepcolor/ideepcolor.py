import numpy as np
import cv2
import os, sys, time, multiprocessing
from libtiff import TIFF, tiff_h_4_3_0 as tiff_h
from libtiff.tif_lzw import encode as lzw_encode
from data import colorize_image as CI

color_model = './models/pytorch/caffemodel.pth'
rgb_image = './imgs/E-056_N-02/Mars_Viking_ClrMosaic_global_925m-E-056_N-02.tif'
test_image = './imgs/E-056_N-02/Murray-Lab_CTX-Mosaic_beta01_E-056_N-02.tif'
gpu_id = 0

def tif_encode_main(_n, strips, encoded):
     encoded[_n] = lzw_encode(strips[_n, :, :].ctypes.data)

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
        
        #
        # The below is basically ripped from 'TIFF.write_image', but extended to include horizontal prediction.
        #
        arr = np.ascontiguousarray(self.result)
        shape = arr.shape
        bits = arr.itemsize * 8
        
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
        
        tif = TIFF.open(os.path.join(save_path, 'ours_fullres.tif'), mode='w')
        tif.SetField(tiff_h.TIFFTAG_COMPRESSION, tiff_h.COMPRESSION_LZW)
        tif.SetField(tiff_h.TIFFTAG_BITSPERSAMPLE, bits)
        tif.SetField(tiff_h.TIFFTAG_SAMPLEFORMAT, sample_format)
        tif.SetField(tiff_h.TIFFTAG_ORIENTATION, tiff_h.ORIENTATION_TOPLEFT)
        
        # Guess the planar config, with preference for separate planes
        if shape[2] == 3 or shape[2] == 4:
            planar_config = tiff_h.PLANARCONFIG_CONTIG
            height, width, depth = shape
            size = width * depth * arr.itemsize
        else:
            planar_config = tiff_h.PLANARCONFIG_SEPARATE
            depth, height, width = shape
            size = width * height * arr.itemsize

        tif.SetField(tiff_h.TIFFTAG_PHOTOMETRIC, tiff_h.PHOTOMETRIC_RGB)
        tif.SetField(tiff_h.TIFFTAG_IMAGEWIDTH, width)
        tif.SetField(tiff_h.TIFFTAG_IMAGELENGTH, height)
        tif.SetField(tiff_h.TIFFTAG_SAMPLESPERPIXEL, depth)
        tif.SetField(tiff_h.TIFFTAG_PLANARCONFIG, planar_config)
        if depth == 4:  # RGBA
            tif.SetField(tiff_h.TIFFTAG_EXTRASAMPLES,
                          [tiff_h.EXTRASAMPLE_UNASSALPHA])
        elif depth > 4:  # No idea...
            tif.SetField(tiff_h.TIFFTAG_EXTRASAMPLES,
                          [tiff_h.EXTRASAMPLE_UNSPECIFIED] * (depth - 3))

        if planar_config == tiff_h.PLANARCONFIG_CONTIG:
            # This field can only be set after compression and before
            # writing data. Horizontal predictor often improves compression,
            # but some rare readers might support LZW only without predictor.
            tif.SetField(tiff_h.TIFFTAG_PREDICTOR, tiff_h.PREDICTOR_HORIZONTAL)
        
            tif.SetField(tiff_h.TIFFTAG_ROWSPERSTRIP, 1)
            
            from operator import sub
            from itertools import starmap, islice
        
            for _n in range(height):
                # Extract the row.
                row = arr[_n, :, :]
                
                # Delta encode the row (horizontal predictor).
                # np.diff computes diffs only -> doesn't contain the first element.
                # Need to prepend the first element to turn it into a delta encoding.
                diff = np.diff(row, axis=0)
                delta = np.insert(diff, 0, row[0, :], axis=0)
                
                e = lzw_encode(delta)
                tif.WriteRawStrip(_n, e.ctypes.data, e.nbytes)
        else:
            for _n in range(depth):
                tif.WriteEncodedStrip(_n, arr[_n, :, :].ctypes.data, size)
        tif.WriteDirectory()
        tif.close()

############### SETUP ###############

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

