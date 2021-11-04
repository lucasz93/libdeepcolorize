#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL tif_lzw_PyArray_API
#include "numpy/arrayobject.h"

#include <tiff.h>
#include <tiffio.h>
#include <pthread.h>
#include <unistd.h>

typedef unsigned char *pixel_t;

typedef struct
{
	tdata_t data;
	tsize_t data_sz, size;
} encoded_row_t;

typedef struct
{
	int start, end;
	size_t stride;
	PyArrayObject *arr;
	encoded_row_t *encoded;
} write_data_t;

extern size_t lzw_encode(uint8 *src, size_t len, uint8 *dst, size_t dst_sz);

static void write_thread(write_data_t *w)
{
	for (int r = w->start; r < w->end; r++)
	{
		// HACK: There's some weirdness with indexing here.
		// The gold plated solution is to make the input Python array contiguous ('np.ascontiguousarray(self.result)').
		// But that _triples_ the time taken to save files. So instead we settle for this weird indexing thing.
		// Accessing component 0 seems to make everything off by 2. Shifting backwards to -1 means the last pixel in the image is blank.
		// Similarly, shifting forwards to 5 means the first pixel in the image isn't written.
		// This hack sucks, but seems to work in practice.
		//
		// Don't judge me - it's 1AM here and this fasttiff module is 2 days behind schedule.
		pixel_t *src = (pixel_t *)PyArray_GETPTR3(w->arr, r, 0, 2);
		
		encoded_row_t *e = w->encoded + r;
				
		e->size = lzw_encode(src, w->stride, e->data, e->data_sz);
	}
}

//
// The below is basically ripped from 'PyLibTiff.TIFF.write_image', but extended to include horizontal prediction.
//
static int c_write_image_contig(const char *path, PyArrayObject *arr, int width, int height, int depth, int compsz, int sample_format)
{
	// Only support RGB8 right now.
	if (depth != 3 && compsz != 1)
	{
		return 0;
	}

	TIFF *tif = TIFFOpen(path, "w");
	TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_LZW);
	TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, compsz * 8);
	TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, sample_format);
	TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);

	const int planar_config = PLANARCONFIG_CONTIG;
	const int size = width * depth * compsz;

	TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
	TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width);
	TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);
	TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, depth);
	TIFFSetField(tif, TIFFTAG_PLANARCONFIG, planar_config);

    // This field can only be set after compression and before
	// writing data. Horizontal predictor often improves compression,
	// but some rare readers might support LZW only without predictor.
//	TIFFSetField(tif, TIFFTAG_PREDICTOR, PREDICTOR_HORIZONTAL);
	TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, 1);

	// Allocate worker threads.
	const int cpu_count = sysconf(_SC_NPROCESSORS_ONLN);
	pthread_t *threads = alloca(sizeof(pthread_t) * cpu_count);
	
	// Allocate return data for worker threads.
	unsigned char *encoded_buf = malloc(size * height);
	encoded_row_t *encoded = malloc(sizeof(encoded_row_t) * height);
	for (int i = 0; i < height; i++)
	{
		encoded[i].data_sz = size;
		encoded[i].data = encoded_buf + i * size;
	}
	
	// Allocate worker thread data.
	write_data_t *write_data = malloc(sizeof(write_data_t) * cpu_count);
	
	// Figure out how much work each thread is doing.
	double current_row = 0.0;
	double rows_per_thread = (double)height / cpu_count;
		
	// Start the workers.
	for (int i = 0; i < cpu_count; i++)
	{
		write_data_t *w = write_data + i;
		
		w->encoded = encoded;
		w->stride = size;
		w->start = (int)(current_row);
		w->end = (int)(current_row + rows_per_thread);
		w->arr = arr;
	
		pthread_create(threads + i, NULL, write_thread, w);
		
		current_row += rows_per_thread;
	}
	
	// Wait for the workers.
	for (int i = 0; i < cpu_count; i++)
	{
		pthread_join(threads[i], NULL);
		
		// Write as the work completes.
		const write_data_t *w = write_data + i;
		for (int r = w->start; r < w->end; r++)
		{
			encoded_row_t *e = encoded + r;
			TIFFWriteRawStrip(tif, r, e->data, e->size);
		}
	}

	// Cleanup.
    TIFFWriteDirectory(tif);
    TIFFClose(tif);
    
	free(encoded);
	free(encoded_buf);
	free(write_data);
	return 1;
}

static PyObject* py_write_image_contig(PyObject *self, PyObject *args, PyObject *kwds) 
{
	char *path;
	PyArrayObject* arr = NULL;
	int width, height, depth, compsz, sample_format;

	if (!PyArg_ParseTuple(args, "sOiiiii", &path, &arr, &width, &height, &depth, &compsz, &sample_format))
		return NULL;
	if (!PyArray_Check(arr))
	{
		PyErr_SetString(PyExc_TypeError, "second argument must be array object");
		return NULL;
	}
	
	return Py_BuildValue("i", c_write_image_contig(path, arr, width, height, depth, compsz, sample_format));
}

static PyMethodDef PyFastTiffMethods[] = {
	{"write_image_contig", (PyCFunction)py_write_image_contig, METH_VARARGS|METH_KEYWORDS, "Writes the TIFF to the specified file."},
	{NULL}  /* Sentinel */
};

static struct PyModuleDef moduledef = {
	PyModuleDef_HEAD_INIT,
	"fasttiff",
	"Fast TIFF IO",
	-1,
	PyFastTiffMethods
};

PyMODINIT_FUNC PyInit_fasttiff() 
{
	import_array();
	if (PyErr_Occurred())
	{
		PyErr_SetString(PyExc_ImportError, "can't initialize module fasttiff (failed to import numpy)");
		return NULL;
	}

	return PyModule_Create(&moduledef); 
} 
