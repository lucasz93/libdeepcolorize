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
rgb_files = build_lut(rgb_root, 'Mars_Viking_ClrMosaic_global_925m_')

#
# Ensure output location exists.
#
if not os.path.exists(out_root):
	os.makedirs(out_root)

#
# 
#
colorizer = None
def colorize_main(key, gray_path, rgb_path, gpu_id):
	global colorizer
	
	filename = os.path.join(out_root, f'{key}.tif')
	if os.path.exists(filename):
		return False

	rgb0 = fasttiff.read_two_quarters_contig(rgb_path, 0)
	rgb1 = fasttiff.read_two_quarters_contig(rgb_path, 1)

	# Make 1 coloriser per GPU in the system.
	# Balance the load between those poor single slot 1070s.
	if colorizer == None:
		colorizer = Colorize(rgb0.shape[1], gpu_id)
	
	gray0 = fasttiff.read_two_quarters_contig(gray_path, 0)
	gray1 = fasttiff.read_two_quarters_contig(gray_path, 1)
	
	ul = colorizer.compute(gray0[0, :, :, :], rgb0[0, :, :, :])
	ur = colorizer.compute(gray0[1, :, :, :], rgb0[1, :, :, :])
	ll = colorizer.compute(gray1[0, :, :, :], rgb1[0, :, :, :])
	lr = colorizer.compute(gray1[1, :, :, :], rgb1[1, :, :, :])
	
	# Write to a temp file first. If we crash (or quit) in here
	# we don't want a corrupted file to prevent regenerating the 
	# correct version.
	fasttiff.stitch_and_write_quarters_contig(
	    f'{filename}_', 
	    ul, ur, ll, lr, 
	    gray0.shape[2] + gray1.shape[2], 
	    gray0.shape[1] + gray1.shape[1],
	    3
	)
	
	os.rename(f'{filename}_', filename)
	return True

# Adapted from https://stackoverflow.com/a/24542445	
def predict_remaining_time(times, remaining_items):
	intervals = (3600, 60, 1)
	
	seconds = int(sum(times) / len(times)) * remaining_items
	result = []

	for count in intervals:
		value = seconds // count
		if value:
			seconds -= value * count
		result.append(f'{value:02}')
		
	return ':'.join(result)

#
# Colorise all the things.
#
i = 1
recent_times = []
for key in sorted(grayscale_files.keys()):
	gray_path = grayscale_files[key]
	rgb_path = rgb_files[key]
	
	print(f'[{i}/{len(grayscale_files)}] {key}... ', end='')
	
	start = time.perf_counter()
	
	if colorize_main(key, gray_path, rgb_path, gpu_id=0):
		# Only update the performance if we processed the tile.
		took = time.perf_counter() - start
		if len(recent_times) == 20:
			recent_times.pop(0);
		recent_times.append(took)
		
		print(f'{took:.1f}s -> {predict_remaining_time(recent_times, len(grayscale_files) - i + 1)} remaining')
	else:
		print('already rendered')
		
	i += 1

