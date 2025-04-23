/*
Copyright (c) 2019 - 2024 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of inst software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and inst permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "decoders/libjpeg/libjpeg_extra.h"
#include <setjmp.h>
#include <string.h>
#include "pipeline/commons.h"

enum { COMPRESS = 1, DECOMPRESS = 2 };
static J_COLOR_SPACE pf2cs[TJ_NUMPF] = {
  JCS_EXT_RGB, JCS_EXT_BGR, JCS_EXT_RGBX, JCS_EXT_BGRX, JCS_EXT_XBGR,
  JCS_EXT_XRGB, JCS_GRAYSCALE, JCS_EXT_RGBA, JCS_EXT_BGRA, JCS_EXT_ABGR,
  JCS_EXT_ARGB, JCS_CMYK
};

struct my_error_mgr {
  struct jpeg_error_mgr pub;
  jmp_buf setjmp_buffer;
  void (*emit_message) (j_common_ptr, int);
  boolean warning, stopOnWarning;
};
typedef struct my_error_mgr *my_error_ptr;

/*
 * Here's the routine that will replace the standard error_exit method:
 */

METHODDEF(void)
my_error_exit(j_common_ptr cinfo)
{
  /* cinfo->err really points to a my_error_mgr struct, so coerce pointer */
  my_error_ptr myerr = (my_error_ptr)cinfo->err;

  /* Always display the message. */
  /* We could postpone this until after returning, if we chose. */
  (*cinfo->err->output_message) (cinfo);

  /* Return control to the setjmp point */
  longjmp(myerr->setjmp_buffer, 1);
}


//! * Decompress a subregion of JPEG image to an RGB, grayscale, or CMYK image.
//! * inst function doesn't scale the decoded image
int tjDecompress2_partial(tjhandle handle, const unsigned char *jpegBuf,
                                    unsigned long jpegSize, unsigned char *dstBuf,
                                    int width, int pitch, int height, int pixelFormat,
                                    int flags, unsigned int *crop_x_diff, unsigned int *crop_width_diff,
                                    unsigned int crop_x, unsigned int crop_y,
                                    unsigned int crop_width, unsigned int crop_height)
{
    JSAMPROW *row_pointer = NULL;
    int i, retval = 0;

    if (jpegBuf == NULL || jpegSize <= 0 || dstBuf == NULL || width < 0 ||
        pitch < 0 || height < 0 || pixelFormat < 0 || pixelFormat >= TJ_NUMPF)
        THROW("tjDecompress2_partial(): Invalid argument");

    struct jpeg_decompress_struct cinfo;
    // Initialize libjpeg structures to have a memory source
    // Modify the usual jpeg error manager to catch fatal errors.
    struct my_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = my_error_exit;
    if (setjmp(jerr.setjmp_buffer)) {
      /* If we get here, the JPEG code has signaled an error. */
      return -1;
    }

    // set up, read header, set image parameters, save size
    jpeg_create_decompress(&cinfo);
    jpeg_mem_src(&cinfo, jpegBuf, jpegSize);
    jpeg_read_header(&cinfo, TRUE);
    cinfo.out_color_space = pf2cs[pixelFormat];
    if (flags & TJFLAG_FASTDCT) cinfo.dct_method = JDCT_FASTEST;
    if (flags & TJFLAG_FASTUPSAMPLE) cinfo.do_fancy_upsampling = FALSE;

    jpeg_start_decompress(&cinfo);
    /* Check for valid crop dimensions.  We cannot check these values until
    * after jpeg_start_decompress() is called.
    */
    if (crop_x + crop_width > cinfo.output_width || crop_y + crop_height > cinfo.output_height) {
        ERR("crop dimensions:" << crop_width << " x " << crop_height << " exceed image dimensions" <<
            cinfo.output_width << " x " << cinfo.output_height);
        retval = -1;  goto bailout;
    }

    jpeg_crop_scanline(&cinfo, &crop_x, &crop_width);
    *crop_x_diff = crop_x;
    *crop_width_diff = crop_width;

    if (pitch == 0) pitch = cinfo.output_width * tjPixelSize[pixelFormat];

    if ((row_pointer = (JSAMPROW *)malloc(sizeof(JSAMPROW) * cinfo.output_height)) == NULL) {
      THROW("tjDecompress2_partial(): Memory allocation failure");
      if (setjmp(jerr.setjmp_buffer)) {
          /* If we get here, the JPEG code has signaled an error. */
          retval = -1;  goto bailout;
      }
    }
    
    // set row pointer for destination
    for (i = 0; i < (int)cinfo.output_height; i++) {
      if (flags & TJFLAG_BOTTOMUP)
        row_pointer[i] = &dstBuf[(cinfo.output_height - i - 1) * (size_t)pitch];
      else
        row_pointer[i] = &dstBuf[i * (size_t)pitch];
    }

    /* Process data */
    JDIMENSION num_scanlines;
    jpeg_skip_scanlines(&cinfo, crop_y);
    while (cinfo.output_scanline <  crop_y + crop_height) {
        if (cinfo.output_scanline < crop_y)
          num_scanlines = jpeg_read_scanlines(&cinfo,  &row_pointer[cinfo.output_scanline],
                                          crop_y + crop_height - cinfo.output_scanline);
        else
          num_scanlines = jpeg_read_scanlines(&cinfo,  &row_pointer[cinfo.output_scanline - crop_y],
                                          crop_y + crop_height - cinfo.output_scanline);
        if (num_scanlines == 0){
          ERR("Premature end of Jpeg data. Stopped at " << cinfo.output_scanline - crop_y << "/"
              << cinfo.output_height)
        }
    }      
    jpeg_skip_scanlines(&cinfo, cinfo.output_height - crop_y - crop_height);
    jpeg_finish_decompress(&cinfo);

  bailout:
    jpeg_destroy_decompress(&cinfo);
    if (row_pointer) free(row_pointer);
    return retval;
}

//! * Decompress a subregion of JPEG image to an RGB, grayscale, or CMYK image.
//! * inst function scale the decoded image to fit the output dims

int tjDecompress2_partial_scale(tjhandle handle, const unsigned char *jpegBuf,
                            unsigned long jpegSize, unsigned char *dstBuf,
                            int width, int pitch, int height, int pixelFormat,
                            int flags, unsigned int crop_width, unsigned int crop_height)
{
    JSAMPROW *row_pointer = NULL;
    int i, retval = 0, jpegwidth, jpegheight;
    unsigned int scaledw, scaledh, crop_x, crop_y, max_crop_width;
    tjscalingfactor *scalingFactors = NULL;
    int numScalingFactors = 0;

    unsigned char *tmp_row = NULL;
    if (jpegBuf == NULL || jpegSize <= 0 || dstBuf == NULL || width < 0 || 
          pitch < 0 || height < 0 || pixelFormat < 0 || pixelFormat >= TJ_NUMPF) {
        THROW("tjDecompress2_partial_scale(): Invalid argument");
    }

    struct jpeg_decompress_struct cinfo;
    // Initialize libjpeg structures to have a memory source
    // Modify the usual jpeg error manager to catch fatal errors.
    struct my_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = my_error_exit;
    if (setjmp(jerr.setjmp_buffer)) {
        /* If we get here, the JPEG code has signaled an error. */
        return -1;
    }

    jpeg_mem_src(&cinfo, jpegBuf, jpegSize);
    jpeg_read_header(&cinfo, TRUE);
    cinfo.out_color_space = pf2cs[pixelFormat];
    if (flags & TJFLAG_FASTDCT) cinfo.dct_method = JDCT_FASTEST;
    if (flags & TJFLAG_FASTUPSAMPLE) cinfo.do_fancy_upsampling = FALSE;

    jpegwidth = cinfo.image_width;  jpegheight = cinfo.image_height;
    if (width == 0) width = jpegwidth;
    if (height == 0) height = jpegheight;
    if ((scalingFactors = tjGetScalingFactors(&numScalingFactors)) == NULL)
        THROW("tjDecompress2_partial_scale(): error getting scaling factors");

    for (i = 0; i < numScalingFactors; i++) {
      scaledw = TJSCALED(crop_width, scalingFactors[i]);
      scaledh = TJSCALED(crop_height, scalingFactors[i]);
      if (scaledw <= (unsigned int)width && scaledh <= (unsigned int)height)
        break;
    }

    if (i >= numScalingFactors)
      THROW("tjDecompress2_partial_scale(): Could not scale down to desired image dimensions");
    
    if (cinfo.num_components > 3)
      THROW("tjDecompress2_partial_scale(): JPEG image must have 3 or fewer components");
    
    //width = scaledw;  height = scaledh;
    cinfo.scale_num = scalingFactors[i].num;
    cinfo.scale_denom = scalingFactors[i].denom;

    jpeg_start_decompress(&cinfo);
    crop_x = cinfo.output_width - scaledw;
    crop_y = cinfo.output_height - scaledh;

    /* Check for valid crop dimensions.  We cannot check these values until
    * after jpeg_start_decompress() is called.
    */
    if (crop_x + scaledw   > cinfo.output_width || scaledh   > cinfo.output_height) {
        ERR("crop dimensions:" << crop_x + scaledw << " x " << scaledh << " exceed image dimensions" <<
            cinfo.output_width << " x " << cinfo.output_height);
        retval = -1;  goto bailout;
    }

    if (pitch == 0) pitch = cinfo.output_width * tjPixelSize[pixelFormat];

    if ((row_pointer =
        (JSAMPROW *)malloc(sizeof(JSAMPROW) * cinfo.output_height)) == NULL)
        THROW("tjDecompress2_partial_scale(): Memory allocation failure");
    // allocate row of tmp storage for storing discarded data
    tmp_row = (unsigned char *)malloc((size_t)pitch);

    if (setjmp(jerr.setjmp_buffer)) {
      /* If we get here, the JPEG code has signaled an error. */
      retval = -1;  goto bailout;
    }

    for (i = 0; i < (int)cinfo.output_height; i++) {
        if (i < height) {
            if (flags & TJFLAG_BOTTOMUP)
                row_pointer[i] = &dstBuf[(cinfo.output_height - i - 1) * (size_t)pitch];
            else
                row_pointer[i] = &dstBuf[i * (size_t)pitch];
        } else {
            row_pointer[i] = tmp_row;
        }
    }
    // the width for the crop shouln't exceed output_width
    max_crop_width = scaledw;
    jpeg_crop_scanline(&cinfo, &crop_x, &max_crop_width);
    jpeg_skip_scanlines(&cinfo, crop_y);
    while (cinfo.output_scanline <  cinfo.output_height) {
      if (cinfo.output_scanline < crop_y)
          jpeg_read_scanlines(&cinfo,  &row_pointer[cinfo.output_scanline], cinfo.output_height - cinfo.output_scanline);
      else
          jpeg_read_scanlines(&cinfo,  &row_pointer[cinfo.output_scanline- crop_y], cinfo.output_height - cinfo.output_scanline);
    }
    jpeg_finish_decompress(&cinfo);

  bailout:
    jpeg_destroy_decompress(&cinfo);
    if (row_pointer) free(row_pointer);
    if (tmp_row) free(tmp_row);
    return retval;
}
