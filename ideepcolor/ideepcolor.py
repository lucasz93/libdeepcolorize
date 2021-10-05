import numpy as np
import cv2
import os, sys
from skimage import color
from data import colorize_image as CI

color_model = './models/pytorch/caffemodel.pth'
rgb_image = './imgs/E-056_N-02/Mars_Viking_ClrMosaic_global_925m-E-056_N-02.tif'
test_image = './imgs/E-056_N-02/Murray-Lab_CTX-Mosaic_beta01_E-056_N-02.tif'
gpu_id = None

class ColorPoint:
    def __init__(self, pnt, color, width):
        self.pnt = pnt
        self.color = color
        self.width = width

#        if self.pnt[0] - width < 0:
#            self.pnt[0] -= self.pnt[0] - width
#        if self.pnt[1] - width < 0:
#            self.pnt[1] -= self.pnt[1] - width

#        if self.pnt[0] + width >= load_size:
#            self.pnt[0] += self.pnt[0] - load_size
#        if self.pnt[1] + width >= load_size:
#            self.pnt[1] += self.pnt[1] - load_size

        self.scale = 1.

    def render(self, im, mask):
        w = int(self.width / self.scale)
        pnt = self.pnt
        x1, y1 = self.pnt[0], self.pnt[1]
        tl = (x1, y1)
        x2, y2 = self.pnt[0]+w, self.pnt[1]+w
        br = (x2, y2)
        cv2.rectangle(mask, tl, br, self.color, -1)
        cv2.rectangle(im, tl, br, self.color, -1)

class Draw:
    def __init__(self, color_model, dist_model, load_size):
        self.model = color_model
        self.dist_model = dist_model
        self.load_size = load_size
        self.points = []

        self.scale = 1.
        self.suffix = 0

    def add_point(self, point):
        self.points += [point]

    def read_image(self, image_file):
        self.image_loaded = True
        self.image_file = image_file
        print(image_file)
        im_bgr = cv2.imread(image_file)
        self.im_full = im_bgr.copy()
        # get image for display
        h, w, c = self.im_full.shape

        im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)
        self.im_gray3 = cv2.cvtColor(im_gray, cv2.COLOR_GRAY2BGR)

        im_bgr = cv2.resize(im_bgr, (self.load_size, self.load_size), interpolation=cv2.INTER_CUBIC)
        self.im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)

        self.im_lab = color.rgb2lab(im_bgr[:, :, ::-1])
        self.im_l = self.im_lab[:, :, 0]
        self.im_ab = self.im_lab[:, :, 1:]
        self.im_size = self.im_rgb.shape[0:2]

        self.im_ab0 = np.zeros((2, self.load_size, self.load_size))
        self.im_mask0 = np.zeros((1, self.load_size, self.load_size))
        self.brushWidth = 2 * self.scale

        self.model.load_image(image_file)

        if (self.dist_model is not None):
            self.dist_model.set_image(self.im_rgb)

    def predict_color(self):
        if self.dist_model is not None and self.image_loaded:
            im, mask = self.get_input()
            im_mask0 = mask > 0.0
            self.im_mask0 = im_mask0.transpose((2, 0, 1))
            im_lab = color.rgb2lab(im).transpose((2, 0, 1))
            self.im_ab0 = im_lab[1:3, :, :]

            self.dist_model.net_forward(self.im_ab0, self.im_mask0)

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
        im_lab = color.rgb2lab(im).transpose((2, 0, 1))
        self.im_ab0 = im_lab[1:3, :, :]

        self.model.net_forward(self.im_ab0, self.im_mask0)
        ab = self.model.output_ab.transpose((1, 2, 0))
        pred_lab = np.concatenate((self.im_l[..., np.newaxis], ab), axis=2)
        pred_rgb = (np.clip(color.lab2rgb(pred_lab), 0, 1) * 255).astype('uint8')
        self.result = pred_rgb

    def save_result(self):
        self.compute_result()

        path = os.path.abspath(self.image_file)
        path, ext = os.path.splitext(path)

        save_path = "_".join([path, str(self.suffix)])
        self.suffix += 1

        print('saving result to <%s>\n' % save_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        #np.save(os.path.join(save_path, 'im_l.npy'), self.model.img_l)
        #np.save(os.path.join(save_path, 'im_ab.npy'), self.im_ab0)
        #np.save(os.path.join(save_path, 'im_mask.npy'), self.im_mask0)

        #result_bgr = cv2.cvtColor(self.result, cv2.COLOR_RGB2BGR)
        #mask = self.im_mask0.transpose((1, 2, 0)).astype(np.uint8) * 255
        #cv2.imwrite(os.path.join(save_path, 'input_mask.png'), mask)
        #cv2.imwrite(os.path.join(save_path, 'ours.png'), result_bgr)
        cv2.imwrite(os.path.join(save_path, 'ours_fullres.png'), self.model.get_img_fullres()[:, :, ::-1])
        #cv2.imwrite(os.path.join(save_path, 'input_fullres.png'), self.model.get_input_img_fullres()[:, :, ::-1])
        #cv2.imwrite(os.path.join(save_path, 'input.png'), self.model.get_input_img()[:, :, ::-1])
        cv2.imwrite(os.path.join(save_path, 'input_ab.png'), self.model.get_sup_img()[:, :, ::-1])

############### SETUP ###############

print(f'Loading RGB {test_image}')
rgb = cv2.imread(rgb_image)
h, w, c = rgb.shape
if w != h:
	raise Exception('w != h')
load_size = h
half_load_size = int(load_size / 2)
print(f'File is {rgb.shape}')
print('')

print('Creating networks...')
colorModel = CI.ColorizeImageTorch(Xd=load_size)
colorModel.prep_net(path=color_model, gpu_id=gpu_id)
print('')

distModel = CI.ColorizeImageTorchDist(Xd=load_size)
distModel.prep_net(path=color_model, gpu_id=gpu_id, dist=True)
print('')

print(f'Loading grayscale {test_image}')
draw = Draw(colorModel, distModel, load_size)
draw.read_image(test_image)
print('')

print('Overlaying RGB')
for y in range(h):
	for x in range(w):
		draw.add_point(ColorPoint([x, y], (int(rgb[y, x, 2]), int(rgb[y, x, 1]), int(rgb[y, x, 0])), 1))
draw.save_result()

