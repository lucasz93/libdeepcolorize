import numpy as np
import cv2
import os, sys, time
from data import colorize_image as CI

color_model = './models/pytorch/caffemodel.pth'
rgb_image = './imgs/E-056_N-02/Mars_Viking_ClrMosaic_global_925m-E-056_N-02.tif'
test_image = './imgs/E-056_N-02/Murray-Lab_CTX-Mosaic_beta01_E-056_N-02.tif'
gpu_id = 0

class ColorPoint:
    def __init__(self, pnt, color, width):
        self.pnt = pnt
        self.color = color
        self.width = width

    def render(self, im, mask):
        w = self.width
        pnt = self.pnt
        x1, y1 = self.pnt[0], self.pnt[1]
        tl = (x1, y1)
        x2, y2 = self.pnt[0]+w, self.pnt[1]+w
        br = (x2, y2)
        cv2.rectangle(mask, tl, br, self.color, -1)
        cv2.rectangle(im, tl, br, self.color, -1)

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

        self.im_ab0 = np.zeros((2, self.load_size, self.load_size))
        self.im_mask0 = np.zeros((1, self.load_size, self.load_size))

        self.model.load_image(image_file)

    def get_input(self):
        h = self.load_size
        w = self.load_size
        im = np.zeros((h, w, 3), np.uint8)
        mask = np.zeros((h, w, 1), np.uint8)

        for p in self.points:
            p.render(im, mask)

        return im, mask

    def compute_result(self):
        im, mask = self.get_input()
        im_mask0 = mask > 0.0
        self.im_mask0 = im_mask0.transpose((2, 0, 1))
        im_lab = CI.rgb2lab_transpose(im)
        self.im_ab0 = im_lab[1:3, :, :]

        self.model.net_forward(self.im_ab0, self.im_mask0)

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
        cv2.imwrite(os.path.join(save_path, 'ours_fullres.png'), self.result)

############### SETUP ###############

cv2.cuda.setDevice(gpu_id)

print(f'Loading RGB {test_image}')
start = time.perf_counter()
rgb = cv2.imread(rgb_image).astype(int)
h, w, c = rgb.shape
if w != h:
    raise Exception('w != h')
load_size = h
half_load_size = int(load_size / 2)
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

# TODO: Replace this with 'draw.setRGB', or something like that.
print('Overlaying RGB')
start = time.perf_counter()
for y in range(h):
    for x in range(w):
        draw.add_point(ColorPoint([x, y], (int(rgb[y, x, 2]), int(rgb[y, x, 1]), int(rgb[y, x, 0])), 1))
print(f'Took {time.perf_counter() - start} seconds\n')

print('Computing (cold)...')
start = time.perf_counter()
draw.compute_result()
print(f'Took {time.perf_counter() - start} seconds\n')

print('Computing (warm)...')
start = time.perf_counter()
draw.compute_result()
print(f'Took {time.perf_counter() - start} seconds\n')

print('Rendering...')
start = time.perf_counter()
draw.render()
print(f'Took {time.perf_counter() - start} seconds\n')

print('Saving...')
start = time.perf_counter()
draw.save_result()
print(f'Took {time.perf_counter() - start} seconds\n')

