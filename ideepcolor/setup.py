from distutils.core import setup, Extension 
import numpy, os

os.environ['CFLAGS'] = '-mavx2'
setup(name='fasttiff', 
    ext_modules=[ 
        Extension('fasttiff', 
            sources=['modules/fasttiff/fasttiff.c', 'modules/fasttiff/tif_lzw.c'],
            include_dirs=[numpy.get_include()],
            libraries=['tiff']
        ) 
    ] 
)
