import os, argparse, time
from ideepcolor import Colorize
import fasttiff

grayscale_root = '/mnt/data/maps/Murray-Lab_CTX-Mosaic_beta01/'
rgb_root = '/mnt/data/maps/Mars_Viking_ClrMosaic_global/Tiled/'
out_root = '/mnt/data/maps/Murray-Lab_CTX-ClrMosaic_beta01'

#
# E.g: turns '/some/dir/Mars_Viking_ClrMosaic_global_E000_N00.tif' into
# 'E000_N00' -> '/some/dir/Mars_Viking_ClrMosaic_global_E000_N00.tif'
#
def build_lut(root, prefix):
	lut = dict()

	print(f'Walking \'{root}\'')
	for root, dirs, files in os.walk(root):
		for f in files:
			if f.endswith('.tif'):
				key = f.replace(prefix, '').replace('.tif', '')
				lut[key] = os.path.join(root, f)

	return lut

#
# Gather all files.
#
grayscale_files = build_lut(grayscale_root, 'Murray-Lab_CTX-Mosaic_beta01_')
rgb_files = build_lut(rgb_root, 'Mars_Viking_ClrMosaic_global_')

#
# Ensure output location exists.
#
if not os.path.exists(out_root):
	os.makedirs(out_root)

#
# 
#
colorize = None
def colorize_main(gray_path, rgb_path, gpu_id):
	global colorize

	rgb0 = fasttiff.read_two_quarters_contig(rgb_path, 0)
	rgb1 = fasttiff.read_two_quarters_contig(rgb_path, 1)

	if colorize == None:
		colorize = Colorize(rgb0.shape[1], gpu_id)

	gray0 = fasttiff.read_two_quarters_contig(gray_path, 0)
	gray1 = fasttiff.read_two_quarters_contig(gray_path, 1)
	
	ul = colorize.compute(gray0[0, :, :, :], rgb0[0, :, :, :])
	ur = colorize.compute(gray0[1, :, :, :], rgb0[1, :, :, :])
	ll = colorize.compute(gray1[0, :, :, :], rgb1[0, :, :, :])
	lr = colorize.compute(gray1[1, :, :, :], rgb1[1, :, :, :])
	
	fasttiff.stitch_and_write_quarters_contig(
	    os.path.join(out_root, f'{key}.tif'), 
	    ul, ur, ll, lr, 
	    gray0.shape[2] + gray1.shape[2], 
	    gray0.shape[1] + gray1.shape[1],
	    3
	)

#
# Colorise all the things.
#
i = 1
for key in grayscale_files:
	gray_path = grayscale_files[key]
	rgb_path = rgb_files[key]
	
	print(f'{i}/{len(grayscale_files)}: {key}... ', end='')
	start = time.perf_counter()
	
	colorize_main(gray_path, rgb_path, gpu_id=0)
	
	print(f'{time.perf_counter() - start}s')
	i += 1
	exit()

