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
	
	int row;	// 0 for the top 2 quarters, 1 for the bottom 2 quarters.
	
	int start, end;
	
	int half_image_width, half_image_length;
	int dst_row_sz;
} read_data_t;

// Split the input into 4 quarters - assume that'll be small enough to fit into the GPU. 
// All good since we're rocking at least 8GB VRAM per card.
static void read_thread_main(read_data_t *r)
{
	TIFF *tif = TIFFOpen(r->path, "rM");

	comp_t * restrict buf = _TIFFmalloc(TIFFStripSize(tif));
	const size_t strip_count = TIFFNumberOfStrips(tif);

	const size_t offset = r->start * r->dst_row_sz;
	char * restrict left_quarter = PyArray_GETPTR4(r->dst, 0, 0, 0, 0) + offset;
	char * restrict right_quarter = PyArray_GETPTR4(r->dst, 1, 0, 0, 0) + offset;
	
	for (size_t strip = r->start + (r->row * r->half_image_length); strip < r->end + (r->row * r->half_image_length); strip++)
	{
		TIFFReadEncodedStrip(tif, strip, buf, (tsize_t) -1);
	
		memcpy(left_quarter, buf, r->dst_row_sz);
		memcpy(right_quarter, buf + r->dst_row_sz, r->dst_row_sz);
		
		left_quarter += r->dst_row_sz;
		right_quarter += r->dst_row_sz;
	}

	_TIFFfree(buf);
	TIFFClose(tif);
}

static PyArrayObject *c_read_two_quarters_contig(const char *path, int row)
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
	const npy_intp dims[4] = { 2, height / 2, width / 2, samples_per_pixel };
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
	double rows_per_thread = (double)(height / 2) / cpu_count;	// Remember: we're only decoding the top or bottom half of the image.
	read_data_t *read_data = alloca(sizeof(read_data_t) * cpu_count);
	for (int i = 0; i < cpu_count; i++)
	{
		read_data_t *r = read_data + i;
		
		r->row = row;
		
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

static PyObject* py_read_two_quarters_contig(PyObject *self, PyObject *args, PyObject *kwds) 
{
	char *path;
	int row;
	
	if (!PyArg_ParseTuple(args, "si", &path, &row))
		return NULL;
	
	return c_read_two_quarters_contig(path, row);
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
	int image_height, depth;
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
static int c_write_image_contig(const char *path, PyArrayObject *arr, int width, int height, int depth)
{
	TIFF *tif = TIFFOpen(path, "wM");
	TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_LZW);
	TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, sizeof(comp_t) * 8);
	TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
	TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);

	const int planar_config = PLANARCONFIG_CONTIG;
	const int size = width * depth * sizeof(comp_t);

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
		w->image_height = height;
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
	int width, height, depth;

	if (!PyArg_ParseTuple(args, "sOiii", &path, &arr, &width, &height, &depth))
		return NULL;
	if (!PyArray_Check(arr))
	{
		PyErr_SetString(PyExc_TypeError, "second argument must be array object");
		return NULL;
	}
	
	return Py_BuildValue("i", c_write_image_contig(path, arr, width, height, depth));
}

/*
================================================================================

	STITCH AND WRITE 2 HALVES TOGETHER

================================================================================
*/

typedef struct
{
	int start, end;
	size_t stride;
	PyArrayObject *left, *right;
	encoded_row_t *encoded;
	int image_height, depth;
} stitch_write_data_t;

static void stitch_and_write_thread_main(stitch_write_data_t *w)
{
	comp_t *left_src, *right_src;
	char *buf = malloc(w->stride);
	
	int half_stride = w->stride / 2;
	
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
		if (w->depth == 3)
		{
			left_src  = (comp_t *)PyArray_GETPTR3(w->left, r, 0, 2);
			right_src = (comp_t *)PyArray_GETPTR3(w->right, r, 0, 2);
		}
		else
		{
			left_src  = (comp_t *)PyArray_GETPTR2(w->left, r, 0);
			right_src = (comp_t *)PyArray_GETPTR2(w->right, r, 0);
		}
		
		memcpy(buf, left_src, half_stride);
		memcpy(buf + half_stride, right_src, half_stride);
		
		encoded_row_t *e = w->encoded + r;
		
		extern size_t lzw_encode(int depth, uint8 *src, size_t len, uint8 *dst, size_t dst_sz);
		e->size = lzw_encode(w->depth, buf, w->stride, e->data, e->data_sz);
	}
	
	free(buf);
}

//
// The below is basically ripped from 'PyLibTiff.TIFF.write_image', but extended to include horizontal prediction.
//
static int c_stitch_and_write_quarters_contig(const char *path, PyArrayObject *ul, PyArrayObject *ur, PyArrayObject *ll, PyArrayObject *lr, int width, int height, int depth)
{
	TIFF *tif = TIFFOpen(path, "wM");
	TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_LZW);
	TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, sizeof(comp_t) * 8);
	TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
	TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);

	const int planar_config = PLANARCONFIG_CONTIG;
	const int size = width * depth * sizeof(comp_t);

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
	stitch_write_data_t *write_data0 = alloca(sizeof(stitch_write_data_t) * cpu_count);
	stitch_write_data_t *write_data1 = alloca(sizeof(stitch_write_data_t) * cpu_count);

	// Figure out how much work each thread is doing.
	double rows_per_thread = (double)(height / 2) / cpu_count;	// Remember: we're only encoding the top or bottom half of the image.

	//
	// Start the workers for the top row.
	//
	double current_row = 0.0;
	for (int i = 0; i < cpu_count; i++)
	{
		stitch_write_data_t *w = write_data0 + i;
		
		w->encoded = encoded;
		w->stride = size;
		w->start = (int)(current_row);
		w->end = (int)(current_row + rows_per_thread);
		w->left = ul;
		w->right = ur;
		w->depth = depth;
	
		pthread_create(threads + i, NULL, stitch_and_write_thread_main, w);
		
		current_row += rows_per_thread;
	}

	//
	// Wait for the top-row workers.
	//
	for (int i = 0; i < cpu_count; i++)
	{
		pthread_join(threads[i], NULL);
	}

	//
	// Kick off the second row workers.
	//
	current_row = 0.0;
	for (int i = 0; i < cpu_count; i++)
	{
		stitch_write_data_t *w = write_data1 + i;

		w->encoded = encoded + (height / 2);
		w->stride = size;
		w->start = (int)(current_row);
		w->end = (int)(current_row + rows_per_thread);
		w->left = ll;
		w->right = lr;
		w->depth = depth;

		pthread_create(threads + i, NULL, stitch_and_write_thread_main, w);
		
		current_row += rows_per_thread;
	}

	//
	// Write the top row while the second row is encoding.
	//
	for (int i = 0; i < cpu_count; i++)
	{
		const stitch_write_data_t *w = write_data0 + i;
		for (int r = w->start; r < w->end; r++)
		{
			encoded_row_t *e = w->encoded + r;
			TIFFWriteRawStrip(tif, r, e->data, e->size);
		}
	}

	//
	// Wait for the workers.
	//
	for (int i = 0; i < cpu_count; i++)
	{
		pthread_join(threads[i], NULL);
		
		// Write as the work completes.
		const stitch_write_data_t *w = write_data1 + i;
		for (int r = w->start; r < w->end; r++)
		{
			encoded_row_t *e = w->encoded + r;
			TIFFWriteRawStrip(tif, r + (height / 2), e->data, e->size);
		}
	}

	// Cleanup.
    TIFFWriteDirectory(tif);
    TIFFClose(tif);
    
	free(encoded);
	free(encoded_buf);
	return 1;
}

static PyObject* py_stitch_and_write_quarters_contig(PyObject *self, PyObject *args, PyObject *kwds) 
{
	char *path;
	PyArrayObject* ul, *ur, *ll, *lr;
	int width, height, depth;

	if (!PyArg_ParseTuple(args, "sOOOOiii", &path, &ul, &ur, &ll, &lr, &width, &height, &depth))
		return NULL;
	if (!PyArray_Check(ul) || !PyArray_Check(ur) || !PyArray_Check(ll) || !PyArray_Check(lr))
	{
		PyErr_SetString(PyExc_TypeError, "const char *path, PyArrayObject *ul, PyArrayObject *ur, PyArrayObject *ll, PyArrayObject *lr, int width, int height, int depth");
		return NULL;
	}
	
	return Py_BuildValue("i", c_stitch_and_write_quarters_contig(path, ul, ur, ll, lr, width, height, depth));
}

/*
================================================================================

	PYTHON STUFF

================================================================================
*/

static PyMethodDef PyFastTiffMethods[] = {
	{"read_two_quarters_contig", (PyCFunction)py_read_two_quarters_contig, METH_VARARGS|METH_KEYWORDS, "Reads the TIFF from the specified file into the given array."},
	{"write_image_contig", (PyCFunction)py_write_image_contig, METH_VARARGS|METH_KEYWORDS, "Writes the TIFF to the specified file."},
	{"stitch_and_write_quarters_contig", (PyCFunction)py_stitch_and_write_quarters_contig, METH_VARARGS|METH_KEYWORDS, "Writes the TIFF to the specified file."},
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
