#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL tif_lzw_PyArray_API
#include "numpy/arrayobject.h"

#include <tiff.h>
#include <tiffio.h>
#include <pthread.h>
#include <unistd.h>

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))

typedef unsigned char comp_t;

void shutup_libtiff(const char* module, const char* fmt, ...) {}

/*
================================================================================

	THREADED FILE READING

================================================================================
*/

typedef struct
{
	const char *path;
	PyArrayObject *dst;
	
	int start, end;
	
	int half_image_width, half_image_length;
	int dst_row_sz;
} read_data_t;

// Split the input into 4 quarters - assume that'll be small enough to fit into the GPU. 
// All good since we're rocking at least 8GB VRAM per card.
static void read_thread_main(read_data_t *r)
{
	size_t strip = 0;
	
	TIFF *tif = TIFFOpen(r->path, "rM");

	comp_t * restrict buf = _TIFFmalloc(TIFFStripSize(tif));
	const size_t strip_count = TIFFNumberOfStrips(tif);

	const size_t u_offset = r->start * r->dst_row_sz;
	const size_t l_offset = (r->start - r->half_image_length) * r->dst_row_sz;

	char * restrict ul = PyArray_GETPTR4(r->dst, 0, 0, 0, 0) + u_offset;
	char * restrict ur = PyArray_GETPTR4(r->dst, 1, 0, 0, 0) + u_offset;
	char * restrict ll = PyArray_GETPTR4(r->dst, 2, 0, 0, 0) + l_offset;
	char * restrict lr = PyArray_GETPTR4(r->dst, 3, 0, 0, 0) + l_offset;
	
	for (strip = r->start; strip < MIN(r->end, strip_count / 2); strip++)
	{
		TIFFReadEncodedStrip(tif, strip, buf, (tsize_t) -1);
	
		memcpy(ul, buf, r->dst_row_sz);
		memcpy(ur, buf + r->dst_row_sz, r->dst_row_sz);
		
		ul += r->dst_row_sz;
		ur += r->dst_row_sz;
	}
	
	for (; strip < MIN(r->end, strip_count); strip++)
	{
		TIFFReadEncodedStrip(tif, strip, buf, (tsize_t) -1);
		
		memcpy(ll, buf, r->dst_row_sz);
		memcpy(lr, buf + r->dst_row_sz, r->dst_row_sz);
		
		ll += r->dst_row_sz;
		lr += r->dst_row_sz;
	}

	_TIFFfree(buf);
	TIFFClose(tif);
}

static PyArrayObject *c_read_image_contig(const char *path)
{
	uint32 width, height;
	int32 rows_per_strip;
	int16 samples_per_pixel;
	short s;

	TIFF *tif = TIFFOpen(path, "rM");
	
	//
	// Ensure this TIF is compatible.
	//
	TIFFGetField(tif, TIFFTAG_COMPRESSION, &s);
	assert(s == COMPRESSION_LZW);
	
	TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &s);
	assert(s == sizeof(comp_t) * 8);
	
	TIFFGetField(tif, TIFFTAG_SAMPLEFORMAT, &s);
	assert(s == SAMPLEFORMAT_UINT);
	
	TIFFGetField(tif, TIFFTAG_ORIENTATION, &s);
	assert(s == ORIENTATION_TOPLEFT);

	TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &s);
	assert(s == PHOTOMETRIC_RGB);
	
	TIFFGetField(tif, TIFFTAG_PLANARCONFIG, &s);
	assert(s == PLANARCONFIG_CONTIG);
	
	TIFFGetField(tif, TIFFTAG_PREDICTOR, &s);
	assert(s == PREDICTOR_HORIZONTAL);
	
	TIFFGetField(tif, TIFFTAG_ROWSPERSTRIP, &rows_per_strip);
	assert(rows_per_strip == 1);
	
	TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samples_per_pixel);
	TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
	TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
	
	//
	// Allocate memory.
	//
	const npy_intp dims[4] = { 4, height / 2, width / 2, samples_per_pixel };
	PyArrayObject *dst = PyArray_New(&PyArray_Type, 
		4, dims,	// Dimensions
		NPY_UINT8, 	// Data type
		NULL, NULL, // Stride + data
		0,
		0, 
		NULL);

	//
	// Spawn the read threads.
	//
	const int cpu_count = sysconf(_SC_NPROCESSORS_ONLN);
	pthread_t *threads = alloca(sizeof(pthread_t) * cpu_count);
	double current_row = 0.0;
	double rows_per_thread = (double)height / cpu_count;
	read_data_t *read_data = alloca(sizeof(read_data_t) * cpu_count);
	for (int i = 0; i < cpu_count; i++)
	{
		read_data_t *r = read_data + i;
		
		r->path = path;
		r->dst = dst;
		r->half_image_width = width / 2;
		r->half_image_length = height / 2;
		r->dst_row_sz = r->half_image_width * sizeof(comp_t) * samples_per_pixel;
		
		r->start = (int)current_row;
		r->end = (int)(current_row + rows_per_thread);
		current_row += rows_per_thread;
		
		pthread_create(threads + i, NULL, read_thread_main, r);
	}
	
	//
	// Wait for the threads.
	//
	for (int i = 0; i < cpu_count; i++)
	{
		pthread_join(threads[i], NULL);
	}
	
	TIFFClose(tif);
	return dst;
}

static PyObject* py_read_image_contig(PyObject *self, PyObject *args, PyObject *kwds) 
{
	char *path;

	if (!PyArg_ParseTuple(args, "s", &path))
		return NULL;
	
	return c_read_image_contig(path);
}

/*
================================================================================

	THREADED FILE WRITING

================================================================================
*/

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
	int depth;
} write_data_t;

static void write_thread_main(write_data_t *w)
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
		comp_t *src = w->depth == 3
			? (comp_t *)PyArray_GETPTR3(w->arr, r, 0, 2)
			: (comp_t *)PyArray_GETPTR2(w->arr, r, 0);
		
		encoded_row_t *e = w->encoded + r;
		
		extern size_t lzw_encode(int depth, uint8 *src, size_t len, uint8 *dst, size_t dst_sz);
		e->size = lzw_encode(w->depth, src, w->stride, e->data, e->data_sz);
	}
}

//
// The below is basically ripped from 'PyLibTiff.TIFF.write_image', but extended to include horizontal prediction.
//
static int c_write_image_contig(const char *path, PyArrayObject *arr, int width, int height, int depth, int compsz, int sample_format)
{
	// Only support RGB8 right now.
	if (compsz != 1)
	{
		return 0;
	}

	TIFF *tif = TIFFOpen(path, "wM");
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
	TIFFSetField(tif, TIFFTAG_PREDICTOR, PREDICTOR_HORIZONTAL);
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
	write_data_t *write_data = alloca(sizeof(write_data_t) * cpu_count);
	
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
		w->depth = depth;
	
		pthread_create(threads + i, NULL, write_thread_main, w);
		
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


/*
================================================================================

	PYTHON STUFF

================================================================================
*/

static PyMethodDef PyFastTiffMethods[] = {
	{"read_image_contig", (PyCFunction)py_read_image_contig, METH_VARARGS|METH_KEYWORDS, "Reads the TIFF from the specified file into the given array."},
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
	TIFFSetErrorHandler(shutup_libtiff);
	TIFFSetWarningHandler(shutup_libtiff);
	
	import_array();
	if (PyErr_Occurred())
	{
		PyErr_SetString(PyExc_ImportError, "can't initialize module fasttiff (failed to import numpy)");
		return NULL;
	}

	return PyModule_Create(&moduledef); 
} 
