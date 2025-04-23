/*
Copyright (c) 2015 - 2024 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#define _CRT_SECURE_NO_WARNINGS
#include "profiler.h"
#include "kernels.h"
#include "lens_distortion_remap.h"
#include "warp.h"
#include "merge.h"
#include "seam_find.h"
#include "exposure_compensation.h"
#include "multiband_blender.h"
#include <sstream>
#include <stdarg.h>
#include <map>
#include <string>

// Version
#define LS_VERSION             "0.9.9"

//////////////////////////////////////////////////////////////////////
//! \brief The magic number for validation
#define LIVE_STITCH_MAGIC      0x600df00d

//////////////////////////////////////////////////////////////////////
//! \brief The stitching modes
enum {
	stitching_mode_normal          = 0, // normal mode
	stitching_mode_quick_and_dirty = 1, // quick and dirty mode
};

//////////////////////////////////////////////////////////////////////
//! \brief The LoomIO module info
#define LOOMIO_MAX_LENGTH_MODULE_NAME        64
#define LOOMIO_MAX_LENGTH_KERNEL_NAME        VX_MAX_KERNEL_NAME
#define LOOMIO_MAX_LENGTH_KERNEL_ARGUMENTS   1024
#define LOOMIO_MIN_AUX_DATA_CAPACITY          256
#define LOOMIO_DEFAULT_AUX_DATA_CAPACITY     1024
struct ls_loomio_info {
	char module[LOOMIO_MAX_LENGTH_MODULE_NAME];
	char kernelName[LOOMIO_MAX_LENGTH_KERNEL_NAME];
	char kernelArguments[LOOMIO_MAX_LENGTH_KERNEL_ARGUMENTS];
};

//////////////////////////////////////////////////////////////////////
//! \brief The internal table buffer sizes
struct ls_internal_table_size_info {
	vx_size warpTableSize;
	vx_size expCompOverlapTableSize;
	vx_size expCompValidTableSize;
	vx_size blendOffsetTableSize;
	vx_size seamFindValidTableSize;
	vx_size seamFindWeightTableSize;
	vx_size seamFindAccumTableSize;
	vx_size seamFindPrefInfoTableSize;
	vx_size seamFindPathTableSize;
};

//////////////////////////////////////////////////////////////////////
//! \brief The stitch handle
struct ls_context_t {
	// magic word and status
	int  magic;                                 // should be LIVE_STITCH_MAGIC
	bool feature_enable_reinitialize;           // true if reinitialize feature is enabled
	bool initialized;                           // true if initialized
	bool scheduled;                             // true if scheduled
	bool reinitialize_required;                 // true if reinitialize required
	bool rig_params_updated;                    // true if rig parameters updated
	bool camera_params_updated;                 // true if camera parameters updated
	bool overlay_params_updated;                // true if overlay parameters updated
	// configuration parameters
	vx_int32    stitching_mode;                 // stitching mode
	vx_uint32   num_cameras;                    // number of cameras
	vx_uint32   num_camera_rows;				// camera buffer number of rows
	vx_uint32   num_camera_columns;				// camera buffer number of cols
	vx_df_image camera_buffer_format;           // camera buffer format (VX_DF_IMAGE_UYVY/YUYV/RGB)
	vx_uint32   camera_buffer_width;            // camera buffer width
	vx_uint32   camera_buffer_height;           // camera buffer height
	camera_params * camera_par;                 // individual camera parameters
	vx_uint32   camera_rgb_buffer_width;        // camera buffer width after color conversion
	vx_uint32   camera_rgb_buffer_height;       // camera buffer height after color conversion
	vx_uint32   num_overlays;                   // number of overlays
	vx_uint32   num_overlay_rows;				// overlay buffer width
	vx_uint32   num_overlay_columns;			// overlay buffer height
	vx_uint32   overlay_buffer_width;           // overlay buffer width
	vx_uint32   overlay_buffer_height;          // overlay buffer height
	camera_params * overlay_par;                // individual overlay parameters
	rig_params  rig_par;                        // rig parameters
	vx_uint32   output_buffer_width;            // output equirectangular image width
	vx_uint32   output_buffer_height;           // output equirectangular image height
	vx_df_image output_buffer_format;           // output image format (VX_DF_IMAGE_UYVY/YUYV/RGB/NV12/IYUV)
	vx_uint32   output_rgb_buffer_width;        // camera buffer width after color conversion
	vx_uint32   output_rgb_buffer_height;       // camera buffer height after color conversion
	cl_context  opencl_context;                 // OpenCL context for DGMA interop
	vx_uint32   camera_buffer_stride_in_bytes;  // stride of each row in input opencl buffer
	vx_uint32   overlay_buffer_stride_in_bytes; // stride of each row in overlay opencl buffer (optional)
	vx_uint32   output_buffer_stride_in_bytes;  // stride of each row in output opencl buffer
	// global options
	vx_uint32   EXPO_COMP, SEAM_FIND;           // exposure comp seam find flags from environment variable
	vx_uint32   SEAM_COST_SELECT;               // seam find cost generation flag from environment variable
	vx_uint32   SEAM_REFRESH, SEAM_FLAGS;       // seamfind seam refresh flag from environment variable
	vx_uint32   MULTIBAND_BLEND;                // multiband blend flag from environment variable
	vx_uint32   EXPO_COMP_GAINW, EXPO_COMP_GAINH;// exposure comp module gain image width and height
	vx_uint32   EXPO_COMP_GAINC;                // exposure comp gain array number of gain values per camera. For mode 4, this should be 12 which is default if not specified.
	// global OpenVX objects
	bool        context_is_external;            // To avoid releaseing external OpenVX context
	vx_context  context;                        // OpenVX context
	vx_graph    graphStitch;                    // OpenVX graph for stitching
	// internal buffer sizes
	ls_internal_table_size_info table_sizes;	// internal table sizes
	vx_image	rgb_input, rgb_output;			// internal images
	// data objects
	vx_remap    overlay_remap;                  // remap table for overlay
	vx_remap    camera_remap;                   // remap table for camera (in simple stitch mode)
	vx_image    Img_input, Img_output, Img_overlay;
	vx_image    Img_input_rgb, Img_output_rgb, Img_overlay_rgb, Img_overlay_rgba;
	vx_node	    InputColorConvertNode, SimpleStitchRemapNode, OutputColorConvertNode;
	vx_array    ValidPixelEntry, WarpRemapEntry, OverlapPixelEntry, valid_array, gain_array;
	vx_matrix   overlap_matrix, A_matrix;
	vx_image    RGBY1, RGBY2, weight_image, cam_id_image, group1_image, group2_image;
	vx_node     WarpNode, ExpcompComputeGainNode, ExpcompSolveGainNode, ExpcompApplyGainNode, MergeNode;
	vx_node     nodeOverlayRemap, nodeOverlayBlend;
	vx_float32  alpha, beta;                    // needed for expcomp
	vx_int32    * A_matrix_initial_value;       // needed for expcomp
	// seamfind data & node elements
	vx_array    overlap_rect_array, seamfind_valid_array, seamfind_weight_array, seamfind_accum_array, 
				seamfind_pref_array, seamfind_info_array, seamfind_path_array, seamfind_scene_array;
	vx_image    valid_mask_image, warp_luma_image, sobelx_image, sobely_image, sobel_magnitude_s16_image, 
				sobel_magnitude_image, sobel_phase_image, seamfind_weight_image;
	vx_node     SobelNode, MagnitudeNode, PhaseNode, ConvertDepthNode, SeamfindStep1Node, SeamfindStep2Node,
				SeamfindStep3Node, SeamfindStep4Node, SeamfindStep5Node, SeamfindAnalyzeNode;
	vx_scalar   current_frame, scene_threshold, seam_cost_enable;
	vx_int32    current_frame_value;
	vx_uint32   scene_threshold_value, SEAM_FIND_TARGET;
	// multiband data elements
	vx_int32    num_bands;
	vx_array    blend_offsets;
	vx_image    blend_mask_image;
	StitchMultibandData * pStitchMultiband;
	vx_size     * multibandBlendOffsetIntoBuffer;
	// LoomIO support
	vx_uint32   loomioOutputAuxSelection, loomioCameraAuxDataLength, loomioOverlayAuxDataLength, loomioOutputAuxDataLength;
	vx_scalar   cameraMediaConfig, overlayMediaConfig, outputMediaConfig, viewingMediaConfig;
	vx_array    loomioCameraAuxData, loomioOverlayAuxData, loomioOutputAuxData, loomioViewingAuxData;
	vx_node     nodeLoomIoCamera, nodeLoomIoOverlay, nodeLoomIoOutput, nodeLoomIoViewing;
	ls_loomio_info loomio_camera, loomio_output, loomio_overlay, loomio_viewing;
	FILE        * loomioAuxDumpFile;
	// internal buffers for input camera lens models
	vx_uint32   paddingPixelCount, overlapCount;
	StitchCoord2dFloat * camSrcMap;
	vx_float32  * camIndexTmpBuf;
	vx_uint8    * camIndexBuf;
	vx_uint32   * validPixelCamMap, *paddedPixelCamMap;
	vx_rectangle_t * overlapRectBuf;
	vx_rectangle_t * overlapValid[LIVE_STITCH_MAX_CAMERAS], *overlapPadded[LIVE_STITCH_MAX_CAMERAS];
	vx_uint32   validCamOverlapInfo[LIVE_STITCH_MAX_CAMERAS], paddedCamOverlapInfo[LIVE_STITCH_MAX_CAMERAS];
	vx_int32    * overlapMatrixBuf;
	// internal buffers for overlay models
	StitchCoord2dFloat * overlaySrcMap;
	vx_uint32   * validPixelOverlayMap;
	vx_float32  * overlayIndexTmpBuf;
	vx_uint8    * overlayIndexBuf;
	// internal buffers for frame encode
    #define MAX_TILE_IMG 16
	vx_uint32   output_encode_buffer_width;             // buffer width after encode conversion
	vx_uint32   output_encode_buffer_height;            // buffer height after encode conversion
	vx_uint32   output_encode_tiles;                    // total number of encode tiles
	vx_uint32   num_encode_sections;                    // total number of encode sectional images
	vx_image    encode_src_rgb_imgs[MAX_TILE_IMG];      // encode intermediate images
	vx_image    encode_dst_imgs[MAX_TILE_IMG];          // encode intermediate images
	vx_image    encodetileOutput[MAX_TILE_IMG];         // encode tile output NV12 images
	vx_rectangle_t src_encode_tile_rect[MAX_TILE_IMG];  // src encode rectangles 
	vx_rectangle_t dst_encode_tile_rect[MAX_TILE_IMG];  // dst encode rectangles 
	vx_node     encode_color_convert_nodes[MAX_TILE_IMG];// nodes to color convert each of the sectional ROI images
	// chroma key
	vx_uint32   CHROMA_KEY;                             // chroma key flag variable
	vx_uint32   CHROMA_KEY_EED;                         // chroma key flag variable
	vx_image    chroma_key_input_img;                   // chroma key input RGB intermediate images
	vx_image    chroma_key_mask_img;                    // chroma key U8 mask intermediate images
	vx_image    chroma_key_input_RGB_img;               // intermediate images
	vx_image    chroma_key_dilate_mask_img;             // chroma key U8 mask dilate intermediate images
	vx_image    chroma_key_erode_mask_img;              // chroma key U8 mask dilate intermediate images
	vx_node     chromaKey_mask_generation_node;         // nodes to generate chroma key mask
	vx_node     chromaKey_dilate_node;                  // nodes to dilate chroma key mask
	vx_node     chromaKey_erode_node;                   // nodes to erode chroma key mask
	vx_node     chromaKey_merge_node;                   // nodes to merge chroma input and stitch output
	// temporal filter
	vx_uint32   NOISE_FILTER;                           // temporal noise filter enable/disable environment variable
	vx_float32  noiseFilterLambda;                      // temporal noise filter variable
	vx_scalar   filterLambda;                           // temporal noise filter scalar lambda variable from user
	vx_delay    noiseFilterImageDelay;                  // temporal noise filter delay element
	vx_image    noiseFilterInput_image;                 // temporal noise filter delay input image
	vx_node     noiseFilterNode;                        // temporal noise filter node
	// quick setup load
	vx_uint32   SETUP_LOAD;                             // quick setup load flag variable
	vx_bool     SETUP_LOAD_FILES_FOUND;                 // quick setup load files found flag variable
	// data for Initialize tables
	vx_uint32   USE_CPU_INIT;
	StitchInitializeData *stitchInitData;
	// attributes
	vx_float32  live_stitch_attr[LIVE_STITCH_ATTR_MAX_COUNT];
};

//////////////////////////////////////////////////////////////////////
//! \brief The global attributes with default values
static bool g_live_stitch_attr_initialized = false;
static vx_float32 g_live_stitch_attr[LIVE_STITCH_ATTR_MAX_COUNT] = { 0 };
static stitch_log_callback_f g_live_stitch_log_message_callback = nullptr;

//////////////////////////////////////////////////////////////////////
//! \brief The macro for object creation error checking and reporting.
#define ERROR_CHECK_OBJECT_(call) { vx_reference obj = (vx_reference)(call); vx_status status = vxGetStatus(obj); if(status != VX_SUCCESS) { ls_printf("ERROR: OpenVX object creation failed at " __FILE__ "#%d\n", __LINE__); return status; } }
//! \brief The macro for status error checking and reporting.
#define ERROR_CHECK_STATUS_(call) {vx_status status = (call); if(status != VX_SUCCESS) { ls_printf("ERROR: OpenVX call failed with status = (%d) at " __FILE__ "#%d\n", status, __LINE__); return status; }  }
//! \brief The macro for type error checking and reporting.
#define ERROR_CHECK_TYPE_(call) { vx_enum type_ = (call); if(type_ == VX_TYPE_INVALID) { ls_printf("ERROR: OpenVX call failed with type = VX_TYPE_INVALID at " __FILE__ "#%d\n", __LINE__); return VX_ERROR_NOT_SUFFICIENT; }  }
//! \brief The macro for object creation error checking and reporting.
#define ERROR_CHECK_ALLOC_(call) { void * obj = (call); if(!obj) { ls_printf("ERROR: memory allocation failed at " __FILE__ "#%d\n", __LINE__); return VX_ERROR_NOT_ALLOCATED; } }
//! \brief The macro for fread error checking and reporting.
#define ERROR_CHECK_FREAD_(call,value) {size_t retVal = (call); if(retVal != (size_t)value) { ls_printf("ERROR: fread call expected to return [ %d elements ] but returned [ %d elements ] at " __FILE__ "#%d\n", (int)value, (int)retVal, __LINE__); return VX_FAILURE; }  }
//! \brief The log callback.
void ls_printf(const char * format, ...)
{
	char buffer[1024];
	va_list args;
	va_start(args, format);
	vsnprintf(buffer, sizeof(buffer) - 1, format, args);
	if (g_live_stitch_log_message_callback) {
		g_live_stitch_log_message_callback(buffer);
	}
	else {
		printf("%s", buffer);
		fflush(stdout);
	}
	va_end(args);
}
//! \brief The log callback.
static void VX_CALLBACK log_callback(vx_context context, vx_reference ref, vx_status status, const vx_char string[])
{
	if (g_live_stitch_log_message_callback) {
		g_live_stitch_log_message_callback(string);
		if (!(string[0] && string[strlen(string) - 1] == '\n')) g_live_stitch_log_message_callback("\n");
	}
	else {
		printf("LOG:[status=%d] %s", status, string);
		if (!(string[0] && string[strlen(string) - 1] == '\n')) printf("\n");
		fflush(stdout);
	}
}
//! \brief Dump utilities.
vx_status DumpBuffer(const vx_uint8 * buf, vx_size size, const char * fileName)
{
	FILE * fp = fopen(fileName, "wb"); 
	if (!fp) { 
		printf("ERROR: DumpBuffer: unable to create: %s\n", fileName); 
		if (fp != NULL)	fclose(fp);
		return VX_FAILURE;
	}
	fwrite(buf, size, 1, fp);
	fclose(fp);
	printf("OK: DumpBuffer: %d bytes into %s\n", (int)size, fileName);
	return VX_SUCCESS;
}
vx_status DumpImage(vx_image img, const char * fileName)
{
	FILE * fp = fopen(fileName, "wb"); 
	if (!fp) { 
		printf("ERROR: DumpImage: unable to create: %s\n", fileName);
		if (fp != NULL)	fclose(fp);
		return VX_FAILURE;
	}
	vx_df_image format = VX_DF_IMAGE_VIRT;
	vx_size num_planes = 0;
	vx_rectangle_t rectFull = { 0, 0, 0, 0 };
	int stride_y;
	ERROR_CHECK_STATUS(vxQueryImage(img, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));
	ERROR_CHECK_STATUS(vxQueryImage(img, VX_IMAGE_ATTRIBUTE_PLANES, &num_planes, sizeof(num_planes)));
	ERROR_CHECK_STATUS(vxQueryImage(img, VX_IMAGE_ATTRIBUTE_WIDTH, &rectFull.end_x, sizeof(rectFull.end_x)));
	ERROR_CHECK_STATUS(vxQueryImage(img, VX_IMAGE_ATTRIBUTE_HEIGHT, &rectFull.end_y, sizeof(rectFull.end_y)));
	// write all image planes from vx_image
	for (vx_uint32 plane = 0; plane < (vx_uint32)num_planes; plane++){
		vx_imagepatch_addressing_t addr = { 0 };
		vx_uint8 * src = NULL;
		ERROR_CHECK_STATUS(vxAccessImagePatch(img, &rectFull, plane, &addr, (void **)&src, VX_READ_ONLY));
		vx_size width = (addr.dim_x * addr.scale_x) / VX_SCALE_UNITY;
		vx_size width_in_bytes = (format == VX_DF_IMAGE_U1_AMD) ? ((width + 7) >> 3) : (width * addr.stride_x);
		stride_y = addr.stride_y;
		for (vx_uint32 y = 0; y < addr.dim_y; y += addr.step_y){
			vx_uint8 *srcp = (vx_uint8 *)vxFormatImagePatchAddress2d(src, 0, y, &addr);
			fwrite(srcp, 1, width_in_bytes, fp);
		}
		ERROR_CHECK_STATUS(vxCommitImagePatch(img, &rectFull, plane, &addr, src));
	}
	fclose(fp);
	printf("OK: Dump: Image %dx%d of stride %d %4.4s image into %s\n", rectFull.end_x, rectFull.end_y, stride_y, (const char *)&format, fileName);
	return VX_SUCCESS;
}
vx_status DumpArray(vx_array arr, const char * fileName)
{
	FILE * fp = fopen(fileName, "wb"); if (!fp) { 
		printf("ERROR: DumpArray: unable to create: %s\n", fileName); 
		if (fp != NULL)	fclose(fp);
		return VX_FAILURE;
	}
	vx_size numItems, itemSize;
	ERROR_CHECK_STATUS_(vxQueryArray(arr, VX_ARRAY_ITEMSIZE, &itemSize, sizeof(itemSize)));
	ERROR_CHECK_STATUS_(vxQueryArray(arr, VX_ARRAY_NUMITEMS, &numItems, sizeof(numItems)));
	vx_map_id map_id;
	vx_uint8 * ptr;
	vx_size stride;
	ERROR_CHECK_STATUS_(vxMapArrayRange(arr, 0, numItems, &map_id, &stride, (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
	fwrite(ptr, itemSize, numItems, fp);
	ERROR_CHECK_STATUS_(vxUnmapArrayRange(arr, map_id));
	fclose(fp);
	printf("OK: Dump: Array [%d][%d] into %s\n", (int)numItems, (int)itemSize, fileName);
	return VX_SUCCESS;
}
static vx_status DumpMatrix(vx_matrix mat, const char * fileName)
{
	FILE * fp = fopen(fileName, "wb"); 
	if (!fp) { 
		printf("ERROR: DumpMatrix: unable to create: %s\n", fileName); 
		if (fp != NULL)	fclose(fp);
		return VX_FAILURE;
	}
	vx_size size;
	ERROR_CHECK_STATUS_(vxQueryMatrix(mat, VX_MATRIX_SIZE, &size, sizeof(size)));
	vx_uint8 * buf = new vx_uint8[size];
	ERROR_CHECK_STATUS_(vxCopyMatrix(mat, buf, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
	fwrite(buf, size, 1, fp);
	delete[] buf;
	fclose(fp);
	vx_size rows, columns;
	ERROR_CHECK_STATUS_(vxQueryMatrix(mat, VX_MATRIX_ROWS, &rows, sizeof(rows)));
	ERROR_CHECK_STATUS_(vxQueryMatrix(mat, VX_MATRIX_COLUMNS, &columns, sizeof(columns)));
	printf("OK: Dump: Matrix %dx%d (%d bytes) into %s\n", (int)rows, (int)columns, (int)size, fileName);
	return VX_SUCCESS;
}
static vx_status DumpRemap(vx_remap remap, const char * fileName)
{
	FILE * fp = fopen(fileName, "wb"); 
	if (!fp) { 
		printf("ERROR: DumpRemap: unable to create: %s\n", fileName); 
		if (fp != NULL)	fclose(fp);
		return VX_FAILURE;
	}
	vx_uint32 dstWidth, dstHeight;
	ERROR_CHECK_STATUS_(vxQueryRemap(remap, VX_REMAP_DESTINATION_WIDTH, &dstWidth, sizeof(dstWidth)));
	ERROR_CHECK_STATUS_(vxQueryRemap(remap, VX_REMAP_DESTINATION_HEIGHT, &dstHeight, sizeof(dstHeight)));
	for (vx_uint32 y = 0; y < dstHeight; y++){
		for (vx_uint32 x = 0; x < dstWidth; x++){
			vx_float32 src_xy[2];
			ERROR_CHECK_STATUS_(vxGetRemapPoint(remap, x, y, &src_xy[0], &src_xy[1]));
			fwrite(src_xy, sizeof(src_xy), 1, fp);
		}
	}
	fclose(fp);
	printf("OK: Dump: Remap %dx%d into %s\n", dstWidth, dstHeight, fileName);
	return VX_SUCCESS;
}
static vx_status DumpReference(vx_reference ref, const char * fileName)
{
	vx_enum type;
	ERROR_CHECK_STATUS_(vxQueryReference(ref, VX_REFERENCE_TYPE, &type, sizeof(type)));
	if (type == VX_TYPE_IMAGE) return DumpImage((vx_image)ref, fileName);
	else if (type == VX_TYPE_ARRAY) return DumpArray((vx_array)ref, fileName);
	else if (type == VX_TYPE_MATRIX) return DumpMatrix((vx_matrix)ref, fileName);
	else if (type == VX_TYPE_REMAP) return DumpRemap((vx_remap)ref, fileName);
	else return VX_ERROR_NOT_SUPPORTED;
}

static vx_image CreateAlignedImage(ls_context stitch, vx_uint32 width, vx_uint32 height, vx_uint32 alignpixels, vx_df_image format, vx_enum mem_type)
{
	if (mem_type == VX_MEMORY_TYPE_OPENCL){
		cl_context opencl_context = nullptr;
		vx_imagepatch_addressing_t addr_in = { 0 };
		void *ptr[1] = { nullptr };
		addr_in.dim_x = width;
		addr_in.dim_y = height;
		addr_in.stride_x = ((format == VX_DF_IMAGE_RGBX) | (format == VX_DF_IMAGE_U32) | (format==VX_DF_IMAGE_S32)) ? 4 : (format == VX_DF_IMAGE_RGB4_AMD) ? 6 : 1;
		if (alignpixels == 0)
			addr_in.stride_y = addr_in.dim_x *addr_in.stride_x;
		else
			addr_in.stride_y = ((addr_in.dim_x + alignpixels - 1) & ~(alignpixels - 1))*addr_in.stride_x;

		// allocate opencl buffer with required dim
		vx_status status = vxQueryContext(stitch->context, VX_CONTEXT_ATTRIBUTE_AMD_OPENCL_CONTEXT, &opencl_context, sizeof(opencl_context));
		if (status != VX_SUCCESS){
			ls_printf("vxQueryContext of failed(%d)\n", status);
			return nullptr;
		}
		vx_size size = (addr_in.dim_y + 1) * addr_in.stride_y;
		cl_int err = CL_SUCCESS;
		cl_mem clImg = clCreateBuffer(opencl_context, CL_MEM_READ_WRITE, size, NULL, &err);
		if (!clImg || err){
			ls_printf("clCreateBuffer of size %d failed(%d)\n", (int)size, err);
			return nullptr;
		}
		ptr[0] = clImg;
		return vxCreateImageFromHandle(stitch->context, format, &addr_in, ptr, mem_type);
	}
	else
	{
		return vxCreateImage(stitch->context, width, height, format);
	}
}

//! \brief Function to set default values to global attributes
static void ResetLiveStitchGlobalAttributes()
{
	// set default settings only once
	if (!g_live_stitch_attr_initialized) {
		g_live_stitch_attr_initialized = true;
		memset(g_live_stitch_attr, 0, sizeof(g_live_stitch_attr));
		g_live_stitch_attr[LIVE_STITCH_ATTR_EXPCOMP] = 1;
		g_live_stitch_attr[LIVE_STITCH_ATTR_EXPCOMP_GAIN_IMG_W] = 1;
		g_live_stitch_attr[LIVE_STITCH_ATTR_EXPCOMP_GAIN_IMG_H] = 1;
		g_live_stitch_attr[LIVE_STITCH_ATTR_EXPCOMP_GAIN_IMG_C] = 1;
		g_live_stitch_attr[LIVE_STITCH_ATTR_EXPCOMP_ALPHA_VALUE] = 0.01f;
		g_live_stitch_attr[LIVE_STITCH_ATTR_EXPCOMP_BETA_VALUE] = 100.0f;
		g_live_stitch_attr[LIVE_STITCH_ATTR_SEAMFIND] = 1;
		g_live_stitch_attr[LIVE_STITCH_ATTR_SEAM_COST_SELECT] = 1;
		g_live_stitch_attr[LIVE_STITCH_ATTR_SEAM_REFRESH] = 1;
		g_live_stitch_attr[LIVE_STITCH_ATTR_SEAM_THRESHOLD] = 25;
		g_live_stitch_attr[LIVE_STITCH_ATTR_SEAM_VERT_PRIORITY] = 1;
		g_live_stitch_attr[LIVE_STITCH_ATTR_SEAM_HORT_PRIORITY] = -1;
		g_live_stitch_attr[LIVE_STITCH_ATTR_SEAM_FREQUENCY] = 6000;
		g_live_stitch_attr[LIVE_STITCH_ATTR_SEAM_QUALITY] = 1;
		g_live_stitch_attr[LIVE_STITCH_ATTR_SEAM_STAGGER] = 1;
		g_live_stitch_attr[LIVE_STITCH_ATTR_SEAM_COEQUSH_ENABLE] = 1;
		g_live_stitch_attr[LIVE_STITCH_ATTR_SEAM_COEQUSH_HFOV_MIN] = 120;
		g_live_stitch_attr[LIVE_STITCH_ATTR_SEAM_COEQUSH_PITCH_TOL] = 5;
		g_live_stitch_attr[LIVE_STITCH_ATTR_SEAM_COEQUSH_YAW_TOL] = 5;
		g_live_stitch_attr[LIVE_STITCH_ATTR_SEAM_COEQUSH_OVERLAP_HR] = 0.15f;
		g_live_stitch_attr[LIVE_STITCH_ATTR_SEAM_COEQUSH_OVERLAP_VD] = 20;
		g_live_stitch_attr[LIVE_STITCH_ATTR_SEAM_COEQUSH_TOPBOT_TOL] = 5;
		g_live_stitch_attr[LIVE_STITCH_ATTR_SEAM_COEQUSH_TOPBOT_VGD] = 46;
		g_live_stitch_attr[LIVE_STITCH_ATTR_MULTIBAND] = 1;
		g_live_stitch_attr[LIVE_STITCH_ATTR_MULTIBAND_NUMBANDS] = 4;
		g_live_stitch_attr[LIVE_STITCH_ATTR_STITCH_MODE] = (float)stitching_mode_normal;
		// frame encoding default attributes
		g_live_stitch_attr[LIVE_STITCH_ATTR_OUTPUT_TILE_NUM_X] = 1;
		g_live_stitch_attr[LIVE_STITCH_ATTR_OUTPUT_TILE_NUM_Y] = 1;
		g_live_stitch_attr[LIVE_STITCH_ATTR_OUTPUT_src_tile_overlap] = 0;
		g_live_stitch_attr[LIVE_STITCH_ATTR_OUTPUT_TILE_BUFFER_VALUE] = 0;
		g_live_stitch_attr[LIVE_STITCH_ATTR_OUTPUT_ENCODER_WIDTH] = 3840;
		g_live_stitch_attr[LIVE_STITCH_ATTR_OUTPUT_ENCODER_HEIGHT] = 2160;
		g_live_stitch_attr[LIVE_STITCH_ATTR_OUTPUT_ENCODER_STRIDE_Y] = 3840;
		// chroma key default
		g_live_stitch_attr[LIVE_STITCH_ATTR_CHROMA_KEY] = 0;
		g_live_stitch_attr[LIVE_STITCH_ATTR_CHROMA_KEY_VALUE] = 8454016;
		g_live_stitch_attr[LIVE_STITCH_ATTR_CHROMA_KEY_TOL] = 25;
		g_live_stitch_attr[LIVE_STITCH_ATTR_CHROMA_KEY_EED] = 0;
		// LoomIO specific attributes
		g_live_stitch_attr[LIVE_STITCH_ATTR_IO_CAMERA_AUX_DATA_SIZE] = (float)LOOMIO_DEFAULT_AUX_DATA_CAPACITY;
		g_live_stitch_attr[LIVE_STITCH_ATTR_IO_OVERLAY_AUX_DATA_SIZE] = (float)LOOMIO_DEFAULT_AUX_DATA_CAPACITY;
		g_live_stitch_attr[LIVE_STITCH_ATTR_IO_OUTPUT_AUX_DATA_SIZE] = (float)LOOMIO_DEFAULT_AUX_DATA_CAPACITY;
		// Temporal Filter
		g_live_stitch_attr[LIVE_STITCH_ATTR_NOISE_FILTER] = 0;
		g_live_stitch_attr[LIVE_STITCH_ATTR_NOISE_FILTER_LAMBDA] = 1;
		g_live_stitch_attr[LIVE_STITCH_ATTR_SAVE_AND_LOAD_INIT] = 0;
	}
}
static std::vector<std::string> split(std::string str, char delimiter) {
	std::stringstream ss(str);
	std::string tok;
	std::vector<std::string> internal;
	while (std::getline(ss, tok, delimiter)) {
		internal.push_back(tok);
	}
	return internal;
}
static vx_status IsValidContext(ls_context stitch)
{
	if (!stitch || stitch->magic != LIVE_STITCH_MAGIC) return VX_ERROR_INVALID_REFERENCE;
	return VX_SUCCESS;
}
static vx_status IsValidContextAndInitialized(ls_context stitch)
{
	if (!stitch || stitch->magic != LIVE_STITCH_MAGIC) return VX_ERROR_INVALID_REFERENCE;
	if (!stitch->initialized) return VX_ERROR_NOT_ALLOCATED;
	return VX_SUCCESS;
}
static vx_status IsValidContextAndNotInitialized(ls_context stitch)
{
	if (!stitch || stitch->magic != LIVE_STITCH_MAGIC) return VX_ERROR_INVALID_REFERENCE;
	if (stitch->initialized) return VX_ERROR_NOT_SUPPORTED;
	return VX_SUCCESS;
}
static const char * GetFileNameSuffix(ls_context stitch, vx_reference ref, bool& isIntermediateTmpData, bool& isForCpuUseOnly)
{
	struct {
		vx_reference ref;
		bool isIntermediateTmpData;
		bool isForCpuUseOnly;
		const char * fileNameSuffix;
	} refList[] = {
		// intermediate tables and data that needs initialization
		{ (vx_reference)stitch->ValidPixelEntry,       false, false, "warp-valid.bin" },
		{ (vx_reference)stitch->WarpRemapEntry,        false, false, "warp-remap.bin" },
		{ (vx_reference)stitch->RGBY1,                 false, false, "warp-rgby.raw" },
		{ (vx_reference)stitch->cam_id_image,          false, false, "merge-camid.raw" },
		{ (vx_reference)stitch->group1_image,          false, false, "merge-group1.raw" },
		{ (vx_reference)stitch->group2_image,          false, false, "merge-group2.raw" },
		{ (vx_reference)stitch->weight_image,          false, false, "merge-weight.raw" },
		{ (vx_reference)stitch->valid_array,           false, false, "exp-valid.bin" },
		{ (vx_reference)stitch->OverlapPixelEntry,     false, false, "exp-overlap.bin" },
		{ (vx_reference)stitch->overlap_matrix,        false, true,  "exp-count.bin" },
		{ (vx_reference)stitch->RGBY2,                 false, false, "exp-rgby.raw" },
		{ (vx_reference)stitch->valid_mask_image,      false, false, "valid-mask.raw" },
		{ (vx_reference)stitch->seamfind_valid_array,  false, false, "seam-valid.bin" },
		{ (vx_reference)stitch->seamfind_weight_array, false, false, "seam-weight.bin" },
		{ (vx_reference)stitch->seamfind_accum_array,  false, false, "seam-accum.bin" },
		{ (vx_reference)stitch->seamfind_pref_array,   false, false, "seam-pref.bin" },
		{ (vx_reference)stitch->seamfind_info_array,   false, false, "seam-info.bin" },
		{ (vx_reference)stitch->seamfind_path_array,   false, false, "seam-path.bin" },
		{ (vx_reference)stitch->seamfind_scene_array,  false, false, "seam-scene.bin" },
		{ (vx_reference)stitch->seamfind_weight_image, false, false, "seam-mask.raw" },
		{ (vx_reference)stitch->blend_mask_image,      false, false, "blend-mask.raw" },
		{ (vx_reference)stitch->blend_offsets,         false, false, "blend-offsets.bin" },
		{ (vx_reference)stitch->camera_remap,          false, false, "remap-input.raw" },
		{ (vx_reference)stitch->overlay_remap,         false, false, "remap-overlay.raw" },
		// intermediate temporary data
		{ (vx_reference)stitch->Img_input,             true,  false, "camera-input.raw" },
		{ (vx_reference)stitch->Img_overlay,           true,  false, "overlay-input.raw" },
		{ (vx_reference)stitch->Img_overlay_rgba,      true,  false, "overlay-warped.raw" },
		{ (vx_reference)stitch->Img_output,            true,  false, "stitch-output.raw" },
	};
	for (vx_size i = 0; i < dimof(refList); i++) {
		if (refList[i].ref && refList[i].ref == ref) {
			isIntermediateTmpData = refList[i].isIntermediateTmpData;
			isForCpuUseOnly = refList[i].isForCpuUseOnly;
			return refList[i].fileNameSuffix;
		}
	}
	return nullptr;
}
static vx_status DumpInternalTables(ls_context stitch, const char * fileNamePrefix, bool dumpIntermediateTmpDataToo)
{
	vx_reference refList[] = {
		// intermediate tables and data that needs initialization
		(vx_reference)stitch->ValidPixelEntry,
		(vx_reference)stitch->WarpRemapEntry,
		(vx_reference)stitch->RGBY1,
		(vx_reference)stitch->cam_id_image,
		(vx_reference)stitch->group1_image,
		(vx_reference)stitch->group2_image,
		(vx_reference)stitch->weight_image,
		(vx_reference)stitch->valid_array,
		(vx_reference)stitch->OverlapPixelEntry,
		(vx_reference)stitch->overlap_matrix,
		(vx_reference)stitch->RGBY2,
		(vx_reference)stitch->valid_mask_image,
		(vx_reference)stitch->seamfind_valid_array,
		(vx_reference)stitch->seamfind_weight_array,
		(vx_reference)stitch->seamfind_accum_array,
		(vx_reference)stitch->seamfind_pref_array,
		(vx_reference)stitch->seamfind_info_array,
		(vx_reference)stitch->seamfind_path_array,
		(vx_reference)stitch->seamfind_scene_array,
		(vx_reference)stitch->seamfind_weight_image,
		(vx_reference)stitch->blend_mask_image,
		(vx_reference)stitch->blend_offsets,
		(vx_reference)stitch->camera_remap,
		(vx_reference)stitch->overlay_remap,
		// intermediate temporary data
		(vx_reference)stitch->Img_input,
		(vx_reference)stitch->Img_overlay,
		(vx_reference)stitch->Img_overlay_rgba,
		(vx_reference)stitch->Img_output,
	};
	for (vx_size i = 0; i < dimof(refList); i++) {
		if (refList[i]) {
			bool isIntermediateTmpData = false, isForCpuUseOnly = false;
			const char * fileNameSuffix = GetFileNameSuffix(stitch, refList[i], isIntermediateTmpData, isForCpuUseOnly);
			if (fileNameSuffix && (!isIntermediateTmpData || dumpIntermediateTmpDataToo)) {
				char fileName[1024]; snprintf(fileName, sizeof(fileName), "%s-%s", fileNamePrefix, fileNameSuffix);
				vx_status status = DumpReference(refList[i], fileName);
				if (status != VX_SUCCESS)
					return status;
			}
		}
	}
	if (dumpIntermediateTmpDataToo && stitch->MULTIBAND_BLEND) {
		for (vx_int32 level = 0; level < stitch->num_bands; level++) {
			vx_status status;
			char fileName[1024];
			snprintf(fileName, sizeof(fileName), "%s-blend-pyr-mask-%d.raw", fileNamePrefix, level);
			status = DumpImage(stitch->pStitchMultiband[level].WeightPyrImgGaussian, fileName);
			if (status != VX_SUCCESS)
				return status;
			snprintf(fileName, sizeof(fileName), "%s-blend-pyr-gauss-%d.raw", fileNamePrefix, level);
			status = DumpImage(stitch->pStitchMultiband[level].DstPyrImgGaussian, fileName);
			if (status != VX_SUCCESS)
				return status;
			snprintf(fileName, sizeof(fileName), "%s-blend-pyr-lap-%d.raw", fileNamePrefix, level);
			status = DumpImage(stitch->pStitchMultiband[level].DstPyrImgLaplacian, fileName);
			if (status != VX_SUCCESS)
				return status;
			snprintf(fileName, sizeof(fileName), "%s-blend-pyr-lap-rec-%d.raw", fileNamePrefix, level);
			status = DumpImage(stitch->pStitchMultiband[level].DstPyrImgLaplacianRec, fileName);
			if (status != VX_SUCCESS)
				return status;
		}
	}
	return VX_SUCCESS;
}
static vx_status SyncInternalTables(ls_context stitch)
{
	vx_reference refList[] = {
		(vx_reference)stitch->ValidPixelEntry,
		(vx_reference)stitch->WarpRemapEntry,
		(vx_reference)stitch->cam_id_image,
		(vx_reference)stitch->group1_image,
		(vx_reference)stitch->group2_image,
		(vx_reference)stitch->weight_image,
		(vx_reference)stitch->valid_mask_image,
		(vx_reference)stitch->valid_array,
		(vx_reference)stitch->OverlapPixelEntry,
		(vx_reference)stitch->seamfind_valid_array,
		(vx_reference)stitch->seamfind_weight_array,
		(vx_reference)stitch->seamfind_accum_array,
		(vx_reference)stitch->seamfind_pref_array,
		(vx_reference)stitch->seamfind_info_array,
		(vx_reference)stitch->seamfind_path_array,
		(vx_reference)stitch->seamfind_weight_image,
		(vx_reference)stitch->seamfind_scene_array,
		(vx_reference)stitch->blend_mask_image,
		(vx_reference)stitch->RGBY1,
		(vx_reference)stitch->RGBY2,
		(vx_reference)stitch->overlay_remap,
		(vx_reference)stitch->camera_remap,
	};
	for (vx_size i = 0; i < dimof(refList); i++) {
		if (refList[i]) {
			vx_status status = vxDirective(refList[i], VX_DIRECTIVE_AMD_COPY_TO_OPENCL);
			if (status != VX_SUCCESS) {
				ls_printf("ERROR: SyncInternalTables: vxDirective([%d], VX_DIRECTIVE_AMD_COPY_TO_OPENCL) failed (%d)\n", (int)i, status);
				return status;
			}
		}
	}
	return VX_SUCCESS;
}
static vx_status quickSetupFilesLookup(ls_context stitch)
{
	FILE * fp = fopen("StitchTableSizes.txt", "r");	
	if (!fp) { stitch->SETUP_LOAD_FILES_FOUND = vx_false_e; }
	else{ stitch->SETUP_LOAD_FILES_FOUND = vx_true_e; }
	if (fp != NULL) fclose(fp);
	return VX_SUCCESS;
}
static vx_status quickSetupDumpTableSizes(ls_context stitch)
{
	FILE * fp = fopen("StitchTableSizes.txt", "w+"); 
	if (!fp) { 
		ls_printf("ERROR: quickSetupDumpTableSize: unable to create: StitchTableSizes.txt\n"); 
		if (fp != NULL)	fclose(fp);
		return VX_FAILURE;
	}
	fprintf(fp, VX_FMT_SIZE, stitch->table_sizes.blendOffsetTableSize); fprintf(fp, "\n");
	fprintf(fp, VX_FMT_SIZE, stitch->table_sizes.expCompOverlapTableSize); fprintf(fp, "\n");
	fprintf(fp, VX_FMT_SIZE, stitch->table_sizes.expCompValidTableSize); fprintf(fp, "\n");
	fprintf(fp, VX_FMT_SIZE, stitch->table_sizes.seamFindAccumTableSize); fprintf(fp, "\n");
	fprintf(fp, VX_FMT_SIZE, stitch->table_sizes.seamFindPathTableSize); fprintf(fp, "\n");
	fprintf(fp, VX_FMT_SIZE, stitch->table_sizes.seamFindPrefInfoTableSize); fprintf(fp, "\n");
	fprintf(fp, VX_FMT_SIZE, stitch->table_sizes.seamFindValidTableSize); fprintf(fp, "\n");
	fprintf(fp, VX_FMT_SIZE, stitch->table_sizes.seamFindWeightTableSize); fprintf(fp, "\n");
	fprintf(fp, VX_FMT_SIZE, stitch->table_sizes.warpTableSize); fprintf(fp, "\n");
	fclose(fp);

	return VX_SUCCESS;
}
static vx_status quickSetupDumpTables(ls_context stitch)
{
	vx_reference refList[] = {
		(vx_reference)stitch->ValidPixelEntry,
		(vx_reference)stitch->WarpRemapEntry,
		(vx_reference)stitch->RGBY1,
		(vx_reference)stitch->cam_id_image,
		(vx_reference)stitch->group1_image,
		(vx_reference)stitch->group2_image,
		(vx_reference)stitch->weight_image,
		(vx_reference)stitch->valid_array,
		(vx_reference)stitch->OverlapPixelEntry,
		(vx_reference)stitch->overlap_matrix,
		(vx_reference)stitch->RGBY2,
		(vx_reference)stitch->valid_mask_image,
		(vx_reference)stitch->seamfind_valid_array,
		(vx_reference)stitch->seamfind_weight_array,
		(vx_reference)stitch->seamfind_accum_array,
		(vx_reference)stitch->seamfind_pref_array,
		(vx_reference)stitch->seamfind_info_array,
		(vx_reference)stitch->seamfind_path_array,
		(vx_reference)stitch->seamfind_scene_array,
		(vx_reference)stitch->seamfind_weight_image,
		(vx_reference)stitch->blend_mask_image,
		(vx_reference)stitch->blend_offsets,
		(vx_reference)stitch->camera_remap,
		(vx_reference)stitch->overlay_remap,
	};
	for (vx_size i = 0; i < dimof(refList); i++) {
		if (refList[i]) {
			bool isIntermediateTmpData = false, isForCpuUseOnly = false;
			const char * fileNameSuffix = GetFileNameSuffix(stitch, refList[i], isIntermediateTmpData, isForCpuUseOnly);
			if (fileNameSuffix && (!isIntermediateTmpData)) {
				char fileName[1024]; snprintf(fileName, sizeof(fileName), "%s",fileNameSuffix);
				vx_status status = DumpReference(refList[i], fileName);
				if (status != VX_SUCCESS)
					return status;
			}
		}
	}
	return VX_SUCCESS;
}
vx_status loadImage(vx_image img, const char * fileName)
{
	FILE * fp = fopen(fileName, "r"); 
	if (!fp) {
		ls_printf("ERROR: loadImage: unable to open: %s\n", fileName); 
		if (fp != NULL)	fclose(fp); 
		return VX_FAILURE;
	}
	vx_df_image format = VX_DF_IMAGE_VIRT;
	vx_size num_planes = 0;
	vx_rectangle_t rectFull = { 0, 0, 0, 0 };
	ERROR_CHECK_STATUS(vxQueryImage(img, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));
	ERROR_CHECK_STATUS(vxQueryImage(img, VX_IMAGE_ATTRIBUTE_PLANES, &num_planes, sizeof(num_planes)));
	ERROR_CHECK_STATUS(vxQueryImage(img, VX_IMAGE_ATTRIBUTE_WIDTH, &rectFull.end_x, sizeof(rectFull.end_x)));
	ERROR_CHECK_STATUS(vxQueryImage(img, VX_IMAGE_ATTRIBUTE_HEIGHT, &rectFull.end_y, sizeof(rectFull.end_y)));
	// write all image planes from vx_image
	for (vx_uint32 plane = 0; plane < (vx_uint32)num_planes; plane++){
		vx_imagepatch_addressing_t addr = { 0 };
		vx_uint8 * src = NULL;
		ERROR_CHECK_STATUS(vxAccessImagePatch(img, &rectFull, plane, &addr, (void **)&src, VX_WRITE_ONLY));
		vx_size width = (addr.dim_x * addr.scale_x) / VX_SCALE_UNITY;
		vx_size width_in_bytes = (format == VX_DF_IMAGE_U1_AMD) ? ((width + 7) >> 3) : (width * addr.stride_x);
		for (vx_uint32 y = 0; y < addr.dim_y; y += addr.step_y){
			vx_uint8 *srcp = (vx_uint8 *)vxFormatImagePatchAddress2d(src, 0, y, &addr);
			ERROR_CHECK_FREAD_(fread(srcp, 1, width_in_bytes, fp), width_in_bytes);
		}
		ERROR_CHECK_STATUS(vxCommitImagePatch(img, &rectFull, plane, &addr, src));
	}
	fclose(fp);
	return VX_SUCCESS;
}
vx_status loadArray(vx_array arr, const char * fileName)
{
	FILE * fp = fopen(fileName, "r"); 
	if (!fp) {
		ls_printf("ERROR: loadArray: unable to open: %s\n", fileName);
		if (fp != NULL)	fclose(fp);
		return VX_FAILURE;
	}
	vx_size numItems, itemSize;
	ERROR_CHECK_STATUS_(vxQueryArray(arr, VX_ARRAY_ITEMSIZE, &itemSize, sizeof(itemSize)));
	ERROR_CHECK_STATUS_(vxQueryArray(arr, VX_ARRAY_CAPACITY, &numItems, sizeof(numItems)));
	StitchWarpRemapEntry validPixelEntry = { 0 };
	ERROR_CHECK_STATUS_(vxTruncateArray(arr, 0));
	ERROR_CHECK_STATUS_(vxAddArrayItems(arr, numItems, &validPixelEntry, 0));
	vx_map_id map_id;
	vx_uint8 * ptr;
	vx_size stride;
	ERROR_CHECK_STATUS_(vxMapArrayRange(arr, 0, numItems, &map_id, &stride, (void **)&ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0));
	ERROR_CHECK_FREAD_(fread(ptr, itemSize, numItems, fp),numItems);
	ERROR_CHECK_STATUS_(vxUnmapArrayRange(arr, map_id));
	fclose(fp);
	return VX_SUCCESS;
}
static vx_status loadMatrix(vx_matrix mat, const char * fileName)
{
	FILE * fp = fopen(fileName, "r"); 
	if (!fp) { 
		ls_printf("ERROR: loadMatrix: unable to read: %s\n", fileName);
		if (fp != NULL)	fclose(fp);
		return VX_FAILURE;
	}
	vx_size size;
	ERROR_CHECK_STATUS_(vxQueryMatrix(mat, VX_MATRIX_SIZE, &size, sizeof(size)));
	vx_uint8 * buf = new vx_uint8[size];
	ERROR_CHECK_STATUS_(vxCopyMatrix(mat, buf, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
	ERROR_CHECK_FREAD_(fread(buf, 4, size, fp),size);
	delete[] buf;
	fclose(fp);
	vx_size rows, columns;
	ERROR_CHECK_STATUS_(vxQueryMatrix(mat, VX_MATRIX_ROWS, &rows, sizeof(rows)));
	ERROR_CHECK_STATUS_(vxQueryMatrix(mat, VX_MATRIX_COLUMNS, &columns, sizeof(columns)));
	return VX_SUCCESS;
}
static vx_status loadRemap(vx_remap remap, const char * fileName)
{
	FILE * fp = fopen(fileName, "r"); 
	if (!fp) { 
		ls_printf("ERROR: loadRemap: unable to read: %s\n", fileName); 
		if (fp != NULL)	fclose(fp);
		return VX_FAILURE;
	}
	vx_uint32 dstWidth, dstHeight;
	ERROR_CHECK_STATUS_(vxQueryRemap(remap, VX_REMAP_DESTINATION_WIDTH, &dstWidth, sizeof(dstWidth)));
	ERROR_CHECK_STATUS_(vxQueryRemap(remap, VX_REMAP_DESTINATION_HEIGHT, &dstHeight, sizeof(dstHeight)));
	for (vx_uint32 y = 0; y < dstHeight; y++){
		for (vx_uint32 x = 0; x < dstWidth; x++){
			vx_float32 src_xy[2];
			ERROR_CHECK_STATUS_(vxGetRemapPoint(remap, x, y, &src_xy[0], &src_xy[1]));
			ERROR_CHECK_FREAD_(fread(src_xy, sizeof(src_xy), 1, fp),1);
		}
	}
	fclose(fp);
	return VX_SUCCESS;
}
static vx_status loadReference(vx_reference ref, const char * fileName)
{
	vx_enum type;
	ERROR_CHECK_STATUS_(vxQueryReference(ref, VX_REFERENCE_TYPE, &type, sizeof(type)));
	if (type == VX_TYPE_IMAGE) return loadImage((vx_image)ref, fileName);
	else if (type == VX_TYPE_ARRAY) return loadArray((vx_array)ref, fileName);
	else if (type == VX_TYPE_MATRIX) return loadMatrix((vx_matrix)ref, fileName);
	else if (type == VX_TYPE_REMAP) return loadRemap((vx_remap)ref, fileName);
	else return VX_ERROR_NOT_SUPPORTED;
}
static vx_status quickSetupLoadTableSizes(ls_context stitch)
{
	FILE * fp = fopen("StitchTableSizes.txt", "r"); 
	if (!fp) { 
		ls_printf("ERROR: quickSetupLoadTableSizes: unable to open: StitchTableSizes.txt\n"); 
		if (fp != NULL)	fclose(fp);
		return VX_FAILURE;
	}
	int readValue = 0;
	readValue = fscanf(fp, VX_FMT_SIZE, &stitch->table_sizes.blendOffsetTableSize); 
	if (!readValue) { ls_printf("ERROR: quickSetupLoadTableSizes: unable to read file\n"); return VX_FAILURE; }
	readValue = fscanf(fp, VX_FMT_SIZE, &stitch->table_sizes.expCompOverlapTableSize); 
	if (!readValue) { ls_printf("ERROR: quickSetupLoadTableSizes: unable to read file\n"); return VX_FAILURE; }
	readValue = fscanf(fp, VX_FMT_SIZE, &stitch->table_sizes.expCompValidTableSize); 
	if (!readValue) { ls_printf("ERROR: quickSetupLoadTableSizes: unable to read file\n"); return VX_FAILURE; }
	readValue = fscanf(fp, VX_FMT_SIZE, &stitch->table_sizes.seamFindAccumTableSize); 
	if (!readValue) { ls_printf("ERROR: quickSetupLoadTableSizes: unable to read file\n"); return VX_FAILURE; }
	readValue = fscanf(fp, VX_FMT_SIZE, &stitch->table_sizes.seamFindPathTableSize); 
	if (!readValue) { ls_printf("ERROR: quickSetupLoadTableSizes: unable to read file\n"); return VX_FAILURE; }
	readValue = fscanf(fp, VX_FMT_SIZE, &stitch->table_sizes.seamFindPrefInfoTableSize); 
	if (!readValue) { ls_printf("ERROR: quickSetupLoadTableSizes: unable to read file\n"); return VX_FAILURE; }
	readValue = fscanf(fp, VX_FMT_SIZE, &stitch->table_sizes.seamFindValidTableSize); 
	if (!readValue) { ls_printf("ERROR: quickSetupLoadTableSizes: unable to read file\n"); return VX_FAILURE; }
	readValue = fscanf(fp, VX_FMT_SIZE, &stitch->table_sizes.seamFindWeightTableSize); 
	if (!readValue) { ls_printf("ERROR: quickSetupLoadTableSizes: unable to read file\n"); return VX_FAILURE; }
	readValue = fscanf(fp, VX_FMT_SIZE, &stitch->table_sizes.warpTableSize); 
	if (!readValue) { ls_printf("ERROR: quickSetupLoadTableSizes: unable to read file\n"); return VX_FAILURE; }

	return VX_SUCCESS;
}
static vx_status quickSetupLoadTables(ls_context stitch)
{
	vx_reference refList[] = {
		// intermediate tables and data that needs initialization
		(vx_reference)stitch->ValidPixelEntry,
		(vx_reference)stitch->WarpRemapEntry,
		(vx_reference)stitch->RGBY1,
		(vx_reference)stitch->cam_id_image,
		(vx_reference)stitch->group1_image,
		(vx_reference)stitch->group2_image,
		(vx_reference)stitch->weight_image,
		(vx_reference)stitch->valid_array,
		(vx_reference)stitch->OverlapPixelEntry,
		(vx_reference)stitch->overlap_matrix,
		(vx_reference)stitch->RGBY2,
		(vx_reference)stitch->valid_mask_image,
		(vx_reference)stitch->seamfind_valid_array,
		(vx_reference)stitch->seamfind_weight_array,
		(vx_reference)stitch->seamfind_accum_array,
		(vx_reference)stitch->seamfind_pref_array,
		(vx_reference)stitch->seamfind_info_array,
		(vx_reference)stitch->seamfind_path_array,
		(vx_reference)stitch->seamfind_scene_array,
		(vx_reference)stitch->seamfind_weight_image,
		(vx_reference)stitch->blend_mask_image,
		(vx_reference)stitch->blend_offsets,
		(vx_reference)stitch->camera_remap,
		(vx_reference)stitch->overlay_remap,
	};
	for (vx_size i = 0; i < dimof(refList); i++) {
		if (refList[i]) {
			bool isIntermediateTmpData = false, isForCpuUseOnly = false;
			const char * fileNameSuffix = GetFileNameSuffix(stitch, refList[i], isIntermediateTmpData, isForCpuUseOnly);
			if (fileNameSuffix && (!isIntermediateTmpData)) {
				char fileName[1024]; snprintf(fileName, sizeof(fileName), "%s", fileNameSuffix);
				vx_status status = loadReference(refList[i], fileName);
				if (status != VX_SUCCESS)
					return status;
			}
		}
	}
	return VX_SUCCESS;
}
static vx_status setupQuickInitializeParams(ls_context stitch)
{
	vx_uint32 camWidth = stitch->camera_rgb_buffer_width / stitch->num_camera_columns;
	vx_uint32 camHeight = stitch->camera_rgb_buffer_height / stitch->num_camera_rows;
	vx_uint32 numCamera = stitch->num_camera_rows * stitch->num_camera_columns;
	vx_uint32 eqrWidth = stitch->output_rgb_buffer_width;
	vx_uint32 eqrHeight = stitch->output_rgb_buffer_height;
	// compute camera warp parameters and check for supported lens types
	float Mcam[32 * 9], Tcam[32 * 3], fcam[32 * 2], Mr[3 * 3];
	vx_status status = CalculateCameraWarpParameters(numCamera, camWidth, camHeight, &stitch->rig_par, stitch->camera_par, Mcam, Tcam, fcam, Mr);
	if (status != VX_SUCCESS) return status;

	vx_uint32 paddingPixelCount = stitch->stitchInitData->paddingPixelCount;
	const camera_lens_params * lens = &stitch->camera_par[0].lens;

	// add nodes to the graph and verify.
	stitch->stitchInitData->params = { 0 };
	stitch->stitchInitData->params.camWidth = camWidth;
	stitch->stitchInitData->params.camHeight = camHeight;
	stitch->stitchInitData->params.paddingPixelCount = paddingPixelCount;
	float cam_params_val = 0.0f;
	bool lens_fish_eye = 0;
	vx_map_id mapIdCamPar;
	vx_size stride = sizeof(vx_size);
	float *cam_params;
	ERROR_CHECK_STATUS(vxTruncateArray(stitch->stitchInitData->CameraParamsArr, 0));
	ERROR_CHECK_STATUS(vxAddArrayItems(stitch->stitchInitData->CameraParamsArr, 32 * numCamera, &cam_params_val, 0));
	ERROR_CHECK_STATUS_(vxMapArrayRange(stitch->stitchInitData->CameraParamsArr, 0, 32 * numCamera, &mapIdCamPar, &stride, (void **)&cam_params, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
	const float * T = Tcam, *M = Mcam, *f = fcam;
	for (vx_uint32 cam = 0; cam < numCamera; cam++, T += 3, M += 9, f += 2) {
		// perform lens distortion and warp for each pixel in the equirectangular destination image
		const camera_lens_params * lens = &stitch->camera_par[cam].lens;
		float k0 = 1.0f - (lens->k1 + lens->k2 + lens->k3);
		float left = 0, top = 0, right = (float)camWidth, bottom = (float)camHeight;
		if (lens->lens_type <= ptgui_lens_fisheye_circ && (lens->reserved[3] != 0 || lens->reserved[4] != 0 || lens->reserved[5] != 0 || lens->reserved[6] != 0)) {
			left = std::max(left, lens->reserved[3]);
			top = std::max(top, lens->reserved[4]);
			right = std::min(right, lens->reserved[5]);
			bottom = std::min(bottom, lens->reserved[6]);
		}
		stitch->stitchInitData->params.camId = cam;
		stitch->stitchInitData->params.lens_type = lens->lens_type;
		stitch->stitchInitData->params.F0 = f[0]; stitch->stitchInitData->params.F1 = f[1];
		stitch->stitchInitData->params.TCam0 = T[0];
		stitch->stitchInitData->params.TCam1 = T[1];
		stitch->stitchInitData->params.TCam2 = T[2];
		cam_params[0] = left; cam_params[1] = top;
		cam_params[2] = right; cam_params[3] = bottom;
		cam_params[4] = lens->k1; cam_params[5] = lens->k2;
		cam_params[6] = lens->k3; cam_params[7] = k0;
		cam_params[8] = lens->du0; cam_params[9] = lens->dv0;
		cam_params[10] = lens->r_crop;
		cam_params[11] = f[0]; cam_params[12] = f[1];
		cam_params[13] = T[0];
		cam_params[14] = T[1];
		cam_params[15] = T[2];
		memcpy(&cam_params[16], (void*)M, sizeof(float) * 9);
		cam_params[25] = (float)lens->lens_type;
		lens_fish_eye = (lens->lens_type == ptgui_lens_fisheye_circ);
		cam_params += 32;
	}
	ERROR_CHECK_STATUS_(vxUnmapArrayRange(stitch->stitchInitData->CameraParamsArr, mapIdCamPar));
	stitch->stitchInitData->params.camId = numCamera;
	stitch->stitchInitData->lens_fish_eye = lens_fish_eye;

	return VX_SUCCESS;
}
static vx_status setupQuickInitializeGraph(ls_context stitch)
{
	vx_uint32 numCamera = stitch->num_camera_rows * stitch->num_camera_columns;
	vx_uint32 eqrWidth = stitch->output_rgb_buffer_width;
	vx_uint32 eqrHeight = stitch->output_rgb_buffer_height;
	vx_uint32 paddingPixelCount = stitch->stitchInitData->paddingPixelCount;
	bool lens_fish_eye =  stitch->stitchInitData->lens_fish_eye;

	if (lens_fish_eye)
	{
		ERROR_CHECK_OBJECT_(stitch->stitchInitData->calc_warp_maps_node = stitchInitCalcCamWarpMaps(stitch->stitchInitData->graphInitialize, &stitch->stitchInitData->params,
			stitch->stitchInitData->CameraParamsArr, stitch->stitchInitData->ValidPixelMap,
			NULL, stitch->stitchInitData->SrcCoordMap, stitch->stitchInitData->CameraZBuffArr));
	}
	else
	{
		ERROR_CHECK_OBJECT_(stitch->stitchInitData->calc_warp_maps_node = stitchInitCalcCamWarpMaps(stitch->stitchInitData->graphInitialize, &stitch->stitchInitData->params,
			stitch->stitchInitData->CameraParamsArr, stitch->stitchInitData->ValidPixelMap,
			stitch->stitchInitData->PaddedPixMap, stitch->stitchInitData->SrcCoordMap, stitch->stitchInitData->CameraZBuffArr));
	}
	ERROR_CHECK_OBJECT_(stitch->stitchInitData->calc_default_idx_node = stitchInitCalcDefCamIdxNode(stitch->stitchInitData->graphInitialize, numCamera,
		eqrWidth, eqrHeight, stitch->stitchInitData->CameraZBuffArr, stitch->stitchInitData->DefaultCamMap));

	if (paddingPixelCount && stitch->stitchInitData->PaddedPixMap && lens_fish_eye)
	{
		ERROR_CHECK_OBJECT_(stitch->stitchInitData->pad_dilate_node = stitchInitExtendPadDilateNode(stitch->stitchInitData->graphInitialize, paddingPixelCount, stitch->stitchInitData->ValidPixelMap, stitch->stitchInitData->PaddedPixMap));
	}
	ERROR_CHECK_STATUS_(vxVerifyGraph(stitch->stitchInitData->graphInitialize));

	return VX_SUCCESS;
}
static vx_status AllocateLensModelBuffersForCamera(ls_context stitch)
{
	stitch->camSrcMap = new StitchCoord2dFloat[stitch->output_rgb_buffer_width * stitch->output_rgb_buffer_height * stitch->num_cameras];
	stitch->validPixelCamMap = new vx_uint32[stitch->output_rgb_buffer_width * stitch->output_rgb_buffer_height];
	stitch->camIndexTmpBuf = new vx_float32[stitch->output_rgb_buffer_width * stitch->output_rgb_buffer_height];
	stitch->camIndexBuf = new vx_uint8[stitch->output_rgb_buffer_width * stitch->output_rgb_buffer_height];
	if (stitch->stitching_mode == stitching_mode_normal) {
		if (stitch->EXPO_COMP) {
			stitch->overlapMatrixBuf = new vx_int32[stitch->num_cameras * stitch->num_cameras];
		}
		if (stitch->MULTIBAND_BLEND) {
			if (stitch->live_stitch_attr[LIVE_STITCH_ATTR_MULTIBAND_PAD_PIXELS] == 0) {
				stitch->paddingPixelCount = (stitch->num_bands <= 4) ? 64 : 128;
			}
			else {
				stitch->paddingPixelCount = (vx_uint32)stitch->live_stitch_attr[LIVE_STITCH_ATTR_MULTIBAND_PAD_PIXELS];
			}
			stitch->paddedPixelCamMap = new vx_uint32[stitch->output_rgb_buffer_width * stitch->output_rgb_buffer_height];
		}
		stitch->overlapRectBuf = new vx_rectangle_t[2 * stitch->num_cameras * stitch->num_cameras];
		for (vx_uint32 cam = 0; cam < stitch->num_cameras; cam++) {
			stitch->overlapValid[cam] = stitch->overlapRectBuf + cam * stitch->num_cameras;
			stitch->overlapPadded[cam] = stitch->overlapValid[cam] + stitch->num_cameras*stitch->num_cameras;
		}
	}
	if (!stitch->USE_CPU_INIT && !stitch->stitchInitData){
		vx_enum StitchCoord2dFloatType;
		stitch->stitchInitData = new StitchInitializeData;
		memset(stitch->stitchInitData, 0, sizeof(StitchInitializeData));
		vx_size arr_size = (stitch->output_rgb_buffer_width * stitch->output_rgb_buffer_height);
		ERROR_CHECK_TYPE_(StitchCoord2dFloatType = vxRegisterUserStruct(stitch->context, sizeof(StitchCoord2dFloat)));
		ERROR_CHECK_OBJECT_(stitch->stitchInitData->graphInitialize = vxCreateGraph(stitch->context));
		ERROR_CHECK_OBJECT_(stitch->stitchInitData->ValidPixelMap = CreateAlignedImage(stitch, stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height, 16, VX_DF_IMAGE_U32, VX_MEMORY_TYPE_OPENCL));
		if (stitch->MULTIBAND_BLEND)
		{
			ERROR_CHECK_OBJECT_(stitch->stitchInitData->PaddedPixMap = CreateAlignedImage(stitch, stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height, 16, VX_DF_IMAGE_U32, VX_MEMORY_TYPE_OPENCL));
		}
		ERROR_CHECK_OBJECT_(stitch->stitchInitData->DefaultCamMap = CreateAlignedImage(stitch, stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height, 16, VX_DF_IMAGE_U8, VX_MEMORY_TYPE_OPENCL));
		ERROR_CHECK_OBJECT_(stitch->stitchInitData->CameraParamsArr = vxCreateArray(stitch->context, VX_TYPE_FLOAT32, 32 * stitch->num_cameras));
		ERROR_CHECK_OBJECT_(stitch->stitchInitData->SrcCoordMap = CreateAlignedImage(stitch, stitch->output_rgb_buffer_width * 2, stitch->output_rgb_buffer_height*stitch->num_cameras, 16, VX_DF_IMAGE_U32, VX_MEMORY_TYPE_OPENCL));
		ERROR_CHECK_OBJECT_(stitch->stitchInitData->CameraZBuffArr = vxCreateVirtualArray(stitch->stitchInitData->graphInitialize, VX_TYPE_FLOAT32, arr_size*stitch->num_cameras));

		// Quick Initailize enabled
		if (stitch->stitchInitData && stitch->stitchInitData->graphInitialize && !stitch->SETUP_LOAD_FILES_FOUND)
		{
			if (stitch->stitching_mode == stitching_mode_quick_and_dirty) stitch->stitchInitData->paddingPixelCount = 0;
			else stitch->stitchInitData->paddingPixelCount = stitch->paddingPixelCount;
			ERROR_CHECK_STATUS_(setupQuickInitializeParams(stitch));
			ERROR_CHECK_STATUS_(setupQuickInitializeGraph(stitch));
		}
	}
	return VX_SUCCESS;
}
static vx_status AllocateLensModelBuffersForOverlay(ls_context stitch)
{
	stitch->overlaySrcMap = new StitchCoord2dFloat[stitch->output_rgb_buffer_width * stitch->output_rgb_buffer_height * stitch->num_overlays];
	stitch->validPixelOverlayMap = new vx_uint32[stitch->output_rgb_buffer_width * stitch->output_rgb_buffer_height];
	stitch->overlayIndexTmpBuf = new vx_float32[stitch->output_rgb_buffer_width * stitch->output_rgb_buffer_height];
	stitch->overlayIndexBuf = new vx_uint8[stitch->output_rgb_buffer_width * stitch->output_rgb_buffer_height];
	return VX_SUCCESS;
}
static vx_status InitializeInternalTablesForRemap(ls_context stitch, vx_remap remap,
	vx_uint32 numCamera, vx_uint32 numCameraColumns, vx_uint32 camWidth, vx_uint32 camHeight, vx_uint32 eqrWidth, vx_uint32 eqrHeight,
	const rig_params * rig_par, const camera_params * cam_par,
	StitchCoord2dFloat * srcMap, vx_uint32 * validPixelMap, vx_float32 * camIndexTmpBuf, vx_uint8 * camIndexBuf)
{
	// compute lens distortion and warp models
	vx_status status = CalculateLensDistortionAndWarpMaps(stitch->stitchInitData, numCamera, camWidth, camHeight, eqrWidth, eqrHeight,
		rig_par, cam_par, validPixelMap, 0, nullptr, srcMap, camIndexTmpBuf, camIndexBuf);

	if (status != VX_SUCCESS) {
		vxAddLogEntry((vx_reference)remap, status, "ERROR: InitializeInternalTablesForRemap: CalculateLensDistortionAndWarpMaps() failed (%d)\n", status);
		return status;
	}

	{ // initialize remap table
		vx_uint32 pixelsPerEqrImage = eqrWidth * eqrHeight;
		vx_uint32 x_offset[256], y_offset[256];
		for (vx_uint32 camId = 0; camId < numCamera; camId++) {
			x_offset[camId] = (camId % numCameraColumns) * camWidth;
			y_offset[camId] = (camId / numCameraColumns) * camHeight;
		}
		for (vx_uint32 y = 0, pos = 0; y < eqrHeight; y++) {
			for (vx_uint32 x = 0; x < eqrWidth; x++, pos++) {
				vx_float32 x_src = -1, y_src = -1;
				vx_uint32 camId = camIndexBuf[pos];
				if (camId < numCamera) {
					const StitchCoord2dFloat * mapEntry = &srcMap[pos + camId * pixelsPerEqrImage];
					x_src = mapEntry->x + x_offset[camId];
					y_src = mapEntry->y + y_offset[camId];
				}
				vxSetRemapPoint(remap, x, y, x_src, y_src);
			}
		}
	}

	return VX_SUCCESS;
}
static vx_status InitializeInternalTablesForCamera(ls_context stitch)
{
	vx_uint32 numCamera = stitch->num_cameras;
	vx_uint32 eqrWidth = stitch->output_rgb_buffer_width;
	vx_uint32 eqrHeight = stitch->output_rgb_buffer_height;
	const vx_uint32 * validPixelCamMap = stitch->validPixelCamMap;
	const vx_uint32 * paddedPixelCamMap = stitch->paddedPixelCamMap;
	const StitchCoord2dFloat * camSrcMap = stitch->camSrcMap;
	const vx_rectangle_t * const * overlapValid = stitch->overlapValid;
	const vx_rectangle_t * const * overlapPadded = stitch->overlapPadded;
	const vx_uint32 * validCamOverlapInfo = stitch->validCamOverlapInfo;
	const vx_uint32 * paddedCamOverlapInfo = stitch->paddedCamOverlapInfo;
	const vx_uint8 * camIndexBuf = stitch->camIndexBuf;

	if (stitch->feature_enable_reinitialize)
	{
		// compute lens distortion and warp models
		vx_status status = CalculateLensDistortionAndWarpMaps(!stitch->USE_CPU_INIT ? stitch->stitchInitData : nullptr, stitch->num_cameras,
			stitch->camera_rgb_buffer_width / stitch->num_camera_columns,
			stitch->camera_rgb_buffer_height / stitch->num_camera_rows,
			stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height,
			&stitch->rig_par, stitch->camera_par,
			stitch->validPixelCamMap, stitch->paddingPixelCount, stitch->paddedPixelCamMap,
			stitch->camSrcMap, stitch->camIndexTmpBuf, stitch->camIndexBuf);
		if (status != VX_SUCCESS) {
			vxAddLogEntry((vx_reference)stitch->context, status, "ERROR: AllocateInternalTablesForCamera: CalculateLensDistortionAndWarpMaps() failed (%d)\n", status);
			return status;
		}
		stitch->overlapCount = CalculateValidOverlapRegions(stitch->num_cameras,
			stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height,
			stitch->validPixelCamMap, stitch->overlapValid, stitch->validCamOverlapInfo,
			stitch->paddedPixelCamMap, stitch->overlapPadded, stitch->paddedCamOverlapInfo);
		if (stitch->overlapCount > 6) {
			vxAddLogEntry((vx_reference)stitch->context, status, "ERROR: AllocateInternalTablesForCamera: number of overlaps (%d) greater than 6 not supported\n", stitch->overlapCount);
			return VX_ERROR_NOT_SUPPORTED;
		}
	}

	{ // initialize warp tables
		StitchValidPixelEntry validPixelEntry = { 0 }, *validPixelBuf = nullptr;
		StitchWarpRemapEntry warpRemapEntry = { 0 }, *warpRemapBuf = nullptr;
		ERROR_CHECK_STATUS_(vxTruncateArray(stitch->ValidPixelEntry, 0));
		ERROR_CHECK_STATUS_(vxTruncateArray(stitch->WarpRemapEntry, 0));
		ERROR_CHECK_STATUS_(vxAddArrayItems(stitch->ValidPixelEntry, stitch->table_sizes.warpTableSize, &validPixelEntry, 0));
		ERROR_CHECK_STATUS_(vxAddArrayItems(stitch->WarpRemapEntry, stitch->table_sizes.warpTableSize, &warpRemapEntry, 0));
		vx_size stride = 0, warpEntryCount = 0; vx_map_id map_id_valid = 0, map_id_warp = 0;
		ERROR_CHECK_STATUS_(vxMapArrayRange(stitch->ValidPixelEntry, 0, stitch->table_sizes.warpTableSize, &map_id_valid, &stride, (void **)&validPixelBuf, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0));
		ERROR_CHECK_STATUS_(vxMapArrayRange(stitch->WarpRemapEntry, 0, stitch->table_sizes.warpTableSize, &map_id_warp, &stride, (void **)&warpRemapBuf, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0));
		vx_status status = GenerateWarpBuffers(numCamera, eqrWidth, eqrHeight,
			validPixelCamMap, paddedPixelCamMap, camSrcMap,
			stitch->num_camera_columns, stitch->camera_rgb_buffer_width / stitch->num_camera_columns,
			stitch->table_sizes.warpTableSize, validPixelBuf, warpRemapBuf, &warpEntryCount);
		ERROR_CHECK_STATUS_(vxUnmapArrayRange(stitch->ValidPixelEntry, map_id_valid));
		ERROR_CHECK_STATUS_(vxUnmapArrayRange(stitch->WarpRemapEntry, map_id_warp));
		if (status != VX_SUCCESS) {
			ls_printf("ERROR: InitializeInternalTablesForCamera: GenerateWarpBuffers() failed (%d)\n", status);
			return status;
		}
		if (!stitch->feature_enable_reinitialize && (warpEntryCount != stitch->table_sizes.warpTableSize)) {
			ls_printf("ERROR: InitializeInternalTablesForCamera: GenerateWarpBuffers output doesn't have enough entries (%d) expected (%d)\n", (vx_uint32)warpEntryCount, (vx_uint32)stitch->table_sizes.warpTableSize);
			return VX_FAILURE;
		}
		if (stitch->feature_enable_reinitialize && (warpEntryCount > stitch->table_sizes.warpTableSize)) {
			ls_printf("ERROR: InitializeInternalTablesForCamera: GenerateWarpBuffers output has more entries (%d) than (%d)\n", (vx_uint32)warpEntryCount, (vx_uint32)stitch->table_sizes.warpTableSize);
			return VX_FAILURE;
		}
		ERROR_CHECK_STATUS_(vxTruncateArray(stitch->ValidPixelEntry, warpEntryCount));
		ERROR_CHECK_STATUS_(vxTruncateArray(stitch->WarpRemapEntry, warpEntryCount));
	}

	{ // initialize merge tables
		vx_rectangle_t rectId = { 0, 0, eqrWidth >> 3, eqrHeight };
		vx_imagepatch_addressing_t addrId, addrG1, addrG2;
		vx_map_id map_id_camId, map_id_camG1, map_id_camG2;
		vx_uint8 * ptr_camId; StitchMergeCamIdEntry * ptr_camG1, *ptr_camG2;
		ERROR_CHECK_STATUS_(vxMapImagePatch(stitch->cam_id_image, &rectId, 0, &map_id_camId, &addrId, (void **)&ptr_camId, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
		ERROR_CHECK_STATUS_(vxMapImagePatch(stitch->group1_image, &rectId, 0, &map_id_camG1, &addrG1, (void **)&ptr_camG1, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
		ERROR_CHECK_STATUS_(vxMapImagePatch(stitch->group2_image, &rectId, 0, &map_id_camG2, &addrG2, (void **)&ptr_camG2, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
		vx_status status = GenerateMergeBuffers(numCamera, eqrWidth, eqrHeight,
			validPixelCamMap, paddedPixelCamMap,
			addrId.stride_y, addrG1.stride_y, addrG2.stride_y, ptr_camId, ptr_camG1, ptr_camG2);
		ERROR_CHECK_STATUS_(vxUnmapImagePatch(stitch->cam_id_image, map_id_camId));
		ERROR_CHECK_STATUS_(vxUnmapImagePatch(stitch->group1_image, map_id_camG1));
		ERROR_CHECK_STATUS_(vxUnmapImagePatch(stitch->group2_image, map_id_camG2));
		if (status != VX_SUCCESS) {
			ls_printf("ERROR: InitializeInternalTablesForCamera: GenerateMergeBuffers() failed (%d)\n", status);
			return status;
		}
	}
	{ // initialize weight and valid mask images
		vx_rectangle_t rectMask = { 0, 0, eqrWidth, eqrHeight * numCamera };
		vx_imagepatch_addressing_t addrMask;
		vx_map_id map_id_mask;
		vx_uint8 * ptr_mask;
		ERROR_CHECK_STATUS_(vxMapImagePatch(stitch->weight_image, &rectMask, 0, &map_id_mask, &addrMask, (void **)&ptr_mask, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
		GenerateDefaultMergeMaskImage(numCamera, eqrWidth, eqrHeight, camIndexBuf, addrMask.stride_y, ptr_mask);
		ERROR_CHECK_STATUS_(vxUnmapImagePatch(stitch->weight_image, map_id_mask));
	}
	if (stitch->EXPO_COMP)
	{ // exposure comp tables
		StitchExpCompCalcEntry validEntry = { 0 }, *validBuf = nullptr;
		StitchOverlapPixelEntry overlapEntry = { 0 }, *overlapBuf = nullptr;
		vx_size stride = 0, validEntryCount = 0, overlapEntryCount = 0; vx_map_id map_id_valid = 0, map_id_overlap = 0;
		ERROR_CHECK_STATUS_(vxTruncateArray(stitch->valid_array, 0));
		ERROR_CHECK_STATUS_(vxAddArrayItems(stitch->valid_array, stitch->table_sizes.expCompValidTableSize, &validEntry, 0));
		ERROR_CHECK_STATUS_(vxMapArrayRange(stitch->valid_array, 0, stitch->table_sizes.expCompValidTableSize, &map_id_valid, &stride, (void **)&validBuf, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0)); // TBD: use no gap flag
		if (stitch->EXPO_COMP < 3) {
			ERROR_CHECK_STATUS_(vxTruncateArray(stitch->OverlapPixelEntry, 0));
			ERROR_CHECK_STATUS_(vxAddArrayItems(stitch->OverlapPixelEntry, stitch->table_sizes.expCompOverlapTableSize, &overlapEntry, 0));
			ERROR_CHECK_STATUS_(vxMapArrayRange(stitch->OverlapPixelEntry, 0, stitch->table_sizes.expCompOverlapTableSize, &map_id_overlap, &stride, (void **)&overlapBuf, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0));
		}
		vx_status status = GenerateExpCompBuffers(numCamera, eqrWidth, eqrHeight,
			validPixelCamMap, overlapValid, validCamOverlapInfo,
			paddedPixelCamMap, overlapPadded, paddedCamOverlapInfo,
			stitch->table_sizes.expCompValidTableSize,
			stitch->table_sizes.expCompOverlapTableSize,
			validBuf, overlapBuf, &validEntryCount, &overlapEntryCount, stitch->overlapMatrixBuf);
		ERROR_CHECK_STATUS_(vxUnmapArrayRange(stitch->valid_array, map_id_valid));
		if (stitch->EXPO_COMP < 3) {
			ERROR_CHECK_STATUS_(vxUnmapArrayRange(stitch->OverlapPixelEntry, map_id_overlap));
		}
		if (status != VX_SUCCESS) {
			ls_printf("ERROR: InitializeInternalTablesForCamera: GenerateExpCompBuffers() failed (%d)\n", status);
			return status;
		}
		if (!stitch->feature_enable_reinitialize && (validEntryCount != stitch->table_sizes.expCompValidTableSize || overlapEntryCount != stitch->table_sizes.expCompOverlapTableSize)) {
			ls_printf("ERROR: InitializeInternalTablesForCamera: GenerateExpCompBuffers output doesn't have enough entries (%d,%d) expected (%d,%d)\n",
				(vx_uint32)validEntryCount, (vx_uint32)overlapEntryCount,
				(vx_uint32)stitch->table_sizes.expCompValidTableSize, (vx_uint32)stitch->table_sizes.expCompOverlapTableSize);
			return VX_FAILURE;
		}
		if (stitch->feature_enable_reinitialize && (validEntryCount > stitch->table_sizes.expCompValidTableSize || overlapEntryCount > stitch->table_sizes.expCompOverlapTableSize)) {
			ls_printf("ERROR: InitializeInternalTablesForCamera: GenerateExpCompBuffers output has more entries (%d,%d) than (%d,%d)\n",
				(vx_uint32)validEntryCount, (vx_uint32)overlapEntryCount,
				(vx_uint32)stitch->table_sizes.expCompValidTableSize, (vx_uint32)stitch->table_sizes.expCompOverlapTableSize);
			return VX_FAILURE;
		}
		ERROR_CHECK_STATUS_(vxTruncateArray(stitch->valid_array, validEntryCount));
		if (stitch->EXPO_COMP < 3) {
			ERROR_CHECK_STATUS_(vxTruncateArray(stitch->OverlapPixelEntry, overlapEntryCount));
		}
		ERROR_CHECK_STATUS_(vxWriteMatrix(stitch->overlap_matrix, stitch->overlapMatrixBuf));
	}

	if (stitch->SEAM_FIND)
	{ // seam find tables
		vx_size seamFindValidEntryCount, seamFindWeightEntryCount, seamFindAccumEntryCount;
		vx_size seamFindPrefInfoEntryCount, seamFindPathEntryCount, stride;
		vx_map_id mapIdValid, mapIdWeight, mapIdAccum, mapIdPref, mapIdInfo;
		StitchSeamFindValidEntry validEntry = { 0 }, *validTable = nullptr;
		StitchSeamFindWeightEntry weightEntry = { 0 }, *weightTable = nullptr;
		StitchSeamFindAccumEntry accumEntry = { 0 }, *accumTable = nullptr;
		StitchSeamFindPreference prefEntry = { 0 }, *prefTable = nullptr;
		StitchSeamFindInformation infoEntry = { 0 }, *infoTable = nullptr;
		ERROR_CHECK_STATUS_(vxTruncateArray(stitch->seamfind_valid_array, 0));
		ERROR_CHECK_STATUS_(vxTruncateArray(stitch->seamfind_weight_array, 0));
		ERROR_CHECK_STATUS_(vxTruncateArray(stitch->seamfind_accum_array, 0));
		ERROR_CHECK_STATUS_(vxTruncateArray(stitch->seamfind_pref_array, 0));
		ERROR_CHECK_STATUS_(vxTruncateArray(stitch->seamfind_info_array, 0));
		ERROR_CHECK_STATUS_(vxAddArrayItems(stitch->seamfind_valid_array, stitch->table_sizes.seamFindValidTableSize, &validEntry, 0));
		ERROR_CHECK_STATUS_(vxAddArrayItems(stitch->seamfind_weight_array, stitch->table_sizes.seamFindWeightTableSize, &weightEntry, 0));
		ERROR_CHECK_STATUS_(vxAddArrayItems(stitch->seamfind_accum_array, stitch->table_sizes.seamFindAccumTableSize, &accumEntry, 0));
		ERROR_CHECK_STATUS_(vxAddArrayItems(stitch->seamfind_pref_array, stitch->table_sizes.seamFindPrefInfoTableSize, &prefEntry, 0));
		ERROR_CHECK_STATUS_(vxAddArrayItems(stitch->seamfind_info_array, stitch->table_sizes.seamFindPrefInfoTableSize, &infoEntry, 0));
		ERROR_CHECK_STATUS_(vxMapArrayRange(stitch->seamfind_valid_array, 0, stitch->table_sizes.seamFindValidTableSize, &mapIdValid, &stride, (void **)&validTable, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
		ERROR_CHECK_STATUS_(vxMapArrayRange(stitch->seamfind_weight_array, 0, stitch->table_sizes.seamFindWeightTableSize, &mapIdWeight, &stride, (void **)&weightTable, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
		ERROR_CHECK_STATUS_(vxMapArrayRange(stitch->seamfind_accum_array, 0, stitch->table_sizes.seamFindAccumTableSize, &mapIdAccum, &stride, (void **)&accumTable, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
		ERROR_CHECK_STATUS_(vxMapArrayRange(stitch->seamfind_pref_array, 0, stitch->table_sizes.seamFindPrefInfoTableSize, &mapIdPref, &stride, (void **)&prefTable, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
		ERROR_CHECK_STATUS_(vxMapArrayRange(stitch->seamfind_info_array, 0, stitch->table_sizes.seamFindPrefInfoTableSize, &mapIdInfo, &stride, (void **)&infoTable, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
		vx_status status = GenerateSeamFindBuffers(numCamera, eqrWidth, eqrHeight, stitch->camera_par,
			validPixelCamMap, overlapValid, validCamOverlapInfo,
			paddedPixelCamMap, overlapPadded, paddedCamOverlapInfo,
			stitch->live_stitch_attr,
			stitch->table_sizes.seamFindValidTableSize,
			stitch->table_sizes.seamFindWeightTableSize,
			stitch->table_sizes.seamFindAccumTableSize,
			stitch->table_sizes.seamFindPrefInfoTableSize,
			validTable, weightTable, accumTable, prefTable, infoTable,
			&seamFindValidEntryCount, &seamFindWeightEntryCount,
			&seamFindAccumEntryCount, &seamFindPrefInfoEntryCount, &seamFindPathEntryCount);
		ERROR_CHECK_STATUS_(vxUnmapArrayRange(stitch->seamfind_valid_array, mapIdValid));
		ERROR_CHECK_STATUS_(vxUnmapArrayRange(stitch->seamfind_weight_array, mapIdWeight));
		ERROR_CHECK_STATUS_(vxUnmapArrayRange(stitch->seamfind_accum_array, mapIdAccum));
		ERROR_CHECK_STATUS_(vxUnmapArrayRange(stitch->seamfind_pref_array, mapIdPref));
		ERROR_CHECK_STATUS_(vxUnmapArrayRange(stitch->seamfind_info_array, mapIdInfo));
		ERROR_CHECK_STATUS_(vxTruncateArray(stitch->seamfind_valid_array, seamFindValidEntryCount));
		ERROR_CHECK_STATUS_(vxTruncateArray(stitch->seamfind_weight_array, seamFindWeightEntryCount));
		ERROR_CHECK_STATUS_(vxTruncateArray(stitch->seamfind_accum_array, seamFindAccumEntryCount));
		ERROR_CHECK_STATUS_(vxTruncateArray(stitch->seamfind_pref_array, seamFindPrefInfoEntryCount));
		ERROR_CHECK_STATUS_(vxTruncateArray(stitch->seamfind_info_array, seamFindPrefInfoEntryCount));
		if (status != VX_SUCCESS) {
			ls_printf("ERROR: InitializeInternalTablesForCamera: GenerateSeamFindBuffers() failed (%d)\n", status);
			return status;
		}
		if (!stitch->feature_enable_reinitialize && (seamFindValidEntryCount != stitch->table_sizes.seamFindValidTableSize ||
			seamFindWeightEntryCount != stitch->table_sizes.seamFindWeightTableSize || seamFindAccumEntryCount != stitch->table_sizes.seamFindAccumTableSize ||
			seamFindPrefInfoEntryCount != stitch->table_sizes.seamFindPrefInfoTableSize || seamFindPathEntryCount != stitch->table_sizes.seamFindPathTableSize))
		{
			ls_printf("ERROR: InitializeInternalTablesForCamera: GenerateSeamFindBuffers output doesn't have enough entries (%d,%d,%d,%d,%d) expected (%d,%d,%d,%d,%d)\n",
				(vx_uint32)seamFindValidEntryCount, (vx_uint32)seamFindWeightEntryCount,
				(vx_uint32)seamFindAccumEntryCount, (vx_uint32)seamFindPrefInfoEntryCount,
				(vx_uint32)seamFindPathEntryCount,
				(vx_uint32)stitch->table_sizes.seamFindValidTableSize, (vx_uint32)stitch->table_sizes.seamFindWeightTableSize,
				(vx_uint32)stitch->table_sizes.seamFindAccumTableSize, (vx_uint32)stitch->table_sizes.seamFindPrefInfoTableSize,
				(vx_uint32)stitch->table_sizes.seamFindPathTableSize);
			return VX_FAILURE;
		}
		if (stitch->feature_enable_reinitialize && (seamFindValidEntryCount > stitch->table_sizes.seamFindValidTableSize ||
			seamFindWeightEntryCount > stitch->table_sizes.seamFindWeightTableSize || seamFindAccumEntryCount > stitch->table_sizes.seamFindAccumTableSize ||
			seamFindPrefInfoEntryCount > stitch->table_sizes.seamFindPrefInfoTableSize || seamFindPathEntryCount > stitch->table_sizes.seamFindPathTableSize))
		{
			ls_printf("ERROR: InitializeInternalTablesForCamera: GenerateSeamFindBuffers output has more entries (%d,%d,%d,%d,%d) than (%d,%d,%d,%d,%d)\n",
				(vx_uint32)seamFindValidEntryCount, (vx_uint32)seamFindWeightEntryCount,
				(vx_uint32)seamFindAccumEntryCount, (vx_uint32)seamFindPrefInfoEntryCount,
				(vx_uint32)seamFindPathEntryCount,
				(vx_uint32)stitch->table_sizes.seamFindValidTableSize, (vx_uint32)stitch->table_sizes.seamFindWeightTableSize,
				(vx_uint32)stitch->table_sizes.seamFindAccumTableSize, (vx_uint32)stitch->table_sizes.seamFindPrefInfoTableSize,
				(vx_uint32)stitch->table_sizes.seamFindPathTableSize);
			return VX_FAILURE;
		}
		// reset current frame value and path & scene arrays (if used)
		stitch->current_frame_value = 0;
		ERROR_CHECK_STATUS_(vxWriteScalarValue(stitch->current_frame, &stitch->current_frame_value));
		StitchSeamFindPathEntry pathEntry = { 0 };
		ERROR_CHECK_STATUS_(vxTruncateArray(stitch->seamfind_path_array, 0));
		ERROR_CHECK_STATUS_(vxAddArrayItems(stitch->seamfind_path_array, seamFindPathEntryCount, &pathEntry, 0));
		if (stitch->seamfind_scene_array) {
			StitchSeamFindSceneEntry sceneEntry = { 0 };
			ERROR_CHECK_STATUS_(vxTruncateArray(stitch->seamfind_scene_array, 0));
			ERROR_CHECK_STATUS_(vxAddArrayItems(stitch->seamfind_scene_array, seamFindPrefInfoEntryCount, &sceneEntry, 0));
		}
		
		// initialize seamfind mask image
		vx_rectangle_t rectMask = { 0, 0, eqrWidth, eqrHeight * numCamera };
		vx_imagepatch_addressing_t addrMask;
		vx_map_id map_id_mask;
		vx_uint8 * ptr_mask;
		ERROR_CHECK_STATUS_(vxMapImagePatch(stitch->seamfind_weight_image, &rectMask, 0, &map_id_mask, &addrMask, (void **)&ptr_mask, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
		GenerateDefaultMergeMaskImage(numCamera, eqrWidth, eqrHeight, camIndexBuf, addrMask.stride_y, ptr_mask);
		ERROR_CHECK_STATUS_(vxUnmapImagePatch(stitch->seamfind_weight_image, map_id_mask));
	}

	if (stitch->MULTIBAND_BLEND)
	{ // multiband blend tables
		vx_size stride;
		StitchBlendValidEntry blendValidEntry = { 0 }, *blendOffsetTable = nullptr;
		vx_map_id mapIdValid;
		ERROR_CHECK_STATUS_(vxTruncateArray(stitch->blend_offsets, 0));
		ERROR_CHECK_STATUS_(vxAddArrayItems(stitch->blend_offsets, stitch->table_sizes.blendOffsetTableSize, &blendValidEntry, 0));
		ERROR_CHECK_STATUS_(vxMapArrayRange(stitch->blend_offsets, 0, stitch->table_sizes.blendOffsetTableSize, &mapIdValid, &stride, (void **)&blendOffsetTable, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
		vx_status status = GenerateBlendBuffers(numCamera, eqrWidth, eqrHeight, stitch->num_bands,
			validPixelCamMap, paddedPixelCamMap, overlapPadded, paddedCamOverlapInfo,
			stitch->multibandBlendOffsetIntoBuffer, stitch->table_sizes.blendOffsetTableSize, blendOffsetTable);
		ERROR_CHECK_STATUS_(vxUnmapArrayRange(stitch->blend_offsets, mapIdValid));
		if (status != VX_SUCCESS) {
			ls_printf("ERROR: InitializeInternalTablesForCamera: GenerateBlendBuffers() failed (%d)\n", status);
			return status;
		}
	}

	if (stitch->valid_mask_image)
	{ // initialize valid pixel mask
		vx_rectangle_t rectMask = { 0, 0, eqrWidth, eqrHeight * numCamera };
		vx_imagepatch_addressing_t addrMask;
		vx_map_id map_id_mask;
		vx_uint8 * ptr_mask;
		ERROR_CHECK_STATUS_(vxMapImagePatch(stitch->valid_mask_image, &rectMask, 0, &map_id_mask, &addrMask, (void **)&ptr_mask, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
		GenerateValidMaskImage(numCamera, eqrWidth, eqrHeight, validPixelCamMap, addrMask.stride_y, ptr_mask);
		ERROR_CHECK_STATUS_(vxUnmapImagePatch(stitch->valid_mask_image, map_id_mask));
	}

	// initialize blend mask image
	if (stitch->blend_mask_image) {
		vx_rectangle_t rectMask = { 0, 0, eqrWidth, eqrHeight * numCamera };
		vx_imagepatch_addressing_t addrMask;
		vx_map_id map_id_mask;
		vx_uint8 * ptr_mask;
		ERROR_CHECK_STATUS_(vxMapImagePatch(stitch->blend_mask_image, &rectMask, 0, &map_id_mask, &addrMask, (void **)&ptr_mask, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
		memset(ptr_mask, 255, addrMask.stride_y * addrMask.dim_y);
		ERROR_CHECK_STATUS_(vxUnmapImagePatch(stitch->blend_mask_image, map_id_mask));
	}
	{ // initialize RGBY1 & RGBY2 to invalid pixels and sync to GPU
		vx_rectangle_t rect = { 0, 0, eqrWidth, eqrHeight * numCamera };
		vx_imagepatch_addressing_t addr;
		vx_map_id map_id;
		vx_uint32 * ptr;
		const __m128i r0 = _mm_set1_epi32(0x80000000);
		ERROR_CHECK_STATUS_(vxMapImagePatch(stitch->RGBY1, &rect, 0, &map_id, &addr, (void **)&ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
		__m128i *dst = (__m128i*) ptr;
		vx_size size_in_bytes = (addr.stride_y * addr.dim_y)&~127;
		for (vx_uint32 i = 0; i < size_in_bytes; i += 128){
			_mm_store_si128(dst++, r0);
			_mm_store_si128(dst++, r0);
			_mm_store_si128(dst++, r0);
			_mm_store_si128(dst++, r0);
			_mm_store_si128(dst++, r0);
			_mm_store_si128(dst++, r0);
			_mm_store_si128(dst++, r0);
			_mm_store_si128(dst++, r0);
		}
		ERROR_CHECK_STATUS_(vxUnmapImagePatch(stitch->RGBY1, map_id));
		if (stitch->RGBY2) {
			ERROR_CHECK_STATUS_(vxMapImagePatch(stitch->RGBY2, &rect, 0, &map_id, &addr, (void **)&ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
			__m128i *dst = (__m128i*) ptr;
			vx_size size_in_bytes = (addr.stride_y * addr.dim_y)&~127;
			for (vx_uint32 i = 0; i < size_in_bytes; i += 128){
				_mm_store_si128(dst++, r0);
				_mm_store_si128(dst++, r0);
				_mm_store_si128(dst++, r0);
				_mm_store_si128(dst++, r0);
				_mm_store_si128(dst++, r0);
				_mm_store_si128(dst++, r0);
				_mm_store_si128(dst++, r0);
				_mm_store_si128(dst++, r0);
			}
			ERROR_CHECK_STATUS_(vxUnmapImagePatch(stitch->RGBY2, map_id));
		}
	}
	return VX_SUCCESS;
}
static vx_status AllocateInternalTablesForCamera(ls_context stitch)
{
	// make sure to allocate internal buffers for initialize atleast once
	if (!stitch->camSrcMap) {
		vx_status status = AllocateLensModelBuffersForCamera(stitch);
		if (status)
			return status;
	}

	if (!stitch->feature_enable_reinitialize)
	{
		if (!stitch->SETUP_LOAD_FILES_FOUND)
		{
			// when re-initialize support is not required, only allocate smallest buffers needed
			// ------
			// compute lens distortion and warp models
			vx_status status = CalculateLensDistortionAndWarpMaps(!stitch->USE_CPU_INIT ? stitch->stitchInitData : nullptr, stitch->num_cameras,
				stitch->camera_rgb_buffer_width / stitch->num_camera_columns,
				stitch->camera_rgb_buffer_height / stitch->num_camera_rows,
				stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height,
				&stitch->rig_par, stitch->camera_par,
				stitch->validPixelCamMap, stitch->paddingPixelCount, stitch->paddedPixelCamMap,
				stitch->camSrcMap, stitch->camIndexTmpBuf, stitch->camIndexBuf);
			if (status != VX_SUCCESS) {
				vxAddLogEntry((vx_reference)stitch->context, status, "ERROR: AllocateInternalTablesForCamera: CalculateLensDistortionAndWarpMaps() failed (%d)\n", status);
				return status;
			}
			stitch->overlapCount = CalculateValidOverlapRegions(stitch->num_cameras,
				stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height,
				stitch->validPixelCamMap, stitch->overlapValid, stitch->validCamOverlapInfo,
				stitch->paddedPixelCamMap, stitch->overlapPadded, stitch->paddedCamOverlapInfo);
			if (stitch->overlapCount > 6) {
				vxAddLogEntry((vx_reference)stitch->context, status, "ERROR: AllocateInternalTablesForCamera: number of overlaps (%d) greater than 6 not supported\n", stitch->overlapCount);
				return VX_ERROR_NOT_SUPPORTED;
			}
			// calculate minimum buffer sizes needed
			CalculateSmallestWarpBufferSizes(stitch->num_cameras, stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height,
				stitch->validPixelCamMap, stitch->paddedPixelCamMap, &stitch->table_sizes.warpTableSize);
			if (stitch->EXPO_COMP) {
				CalculateSmallestExpCompBufferSizes(stitch->num_cameras, stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height,
					stitch->validPixelCamMap, stitch->overlapValid, stitch->validCamOverlapInfo,
					stitch->paddedPixelCamMap, stitch->overlapPadded, stitch->paddedCamOverlapInfo,
					&stitch->table_sizes.expCompValidTableSize, &stitch->table_sizes.expCompOverlapTableSize);
				if (stitch->EXPO_COMP >= 3) {
					// no overlap table needed
					stitch->table_sizes.expCompOverlapTableSize = 0;
				}
			}
			if (stitch->SEAM_FIND) {
				CalculateSmallestSeamFindBufferSizes(stitch->num_cameras, stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height, stitch->camera_par,
					stitch->validPixelCamMap, stitch->overlapValid, stitch->validCamOverlapInfo,
					stitch->paddedPixelCamMap, stitch->overlapPadded, stitch->paddedCamOverlapInfo, stitch->live_stitch_attr,
					&stitch->table_sizes.seamFindValidTableSize, &stitch->table_sizes.seamFindWeightTableSize, &stitch->table_sizes.seamFindAccumTableSize,
					&stitch->table_sizes.seamFindPrefInfoTableSize, &stitch->table_sizes.seamFindPathTableSize);
			}
			if (stitch->MULTIBAND_BLEND) {
				ERROR_CHECK_ALLOC_(stitch->multibandBlendOffsetIntoBuffer = new vx_size[stitch->num_bands]());
				CalculateSmallestBlendBufferSizes(stitch->num_cameras, stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height, stitch->num_bands,
					stitch->validPixelCamMap, stitch->paddedPixelCamMap, stitch->overlapPadded, stitch->paddedCamOverlapInfo,
					stitch->multibandBlendOffsetIntoBuffer, &stitch->table_sizes.blendOffsetTableSize);
			}
			//If load Buffer - dump table sizes
			if (stitch->SETUP_LOAD){
				status = quickSetupDumpTableSizes(stitch);
				if (status != VX_SUCCESS) {
					vxAddLogEntry((vx_reference)stitch->context, status, "ERROR: AllocateInternalTablesForCamera: quickSetupDumpTableSizes() failed (%d)\n", status);
					return status;
				}
			}
		}
		else{
			//If load Buffer - load table sizes
			vx_status status = quickSetupLoadTableSizes(stitch);
			if (status != VX_SUCCESS) {
				vxAddLogEntry((vx_reference)stitch->context, status, "ERROR: AllocateInternalTablesForCamera: quickSetupLoadTableSizes() failed (%d)\n", status);
				return status;
			}
		}
	}
	else
	{
		// when re-initialize support is required, allocate largest buffers to accomodate changes during reinitialize
		CalculateLargestWarpBufferSizes(stitch->num_cameras, stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height,
			&stitch->table_sizes.warpTableSize);
		if (stitch->EXPO_COMP) {
			CalculateLargestExpCompBufferSizes(stitch->num_cameras, stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height,
				&stitch->table_sizes.expCompValidTableSize, &stitch->table_sizes.expCompOverlapTableSize);
		}
		if (stitch->SEAM_FIND) {
			CalculateLargestSeamFindBufferSizes(stitch->num_cameras, stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height,
				&stitch->table_sizes.seamFindValidTableSize, &stitch->table_sizes.seamFindWeightTableSize, &stitch->table_sizes.seamFindAccumTableSize,
				&stitch->table_sizes.seamFindPrefInfoTableSize, &stitch->table_sizes.seamFindPathTableSize);
		}
		if (stitch->MULTIBAND_BLEND) {
			ERROR_CHECK_ALLOC_(stitch->multibandBlendOffsetIntoBuffer = new vx_size[stitch->num_bands]());
			CalculateLargestBlendBufferSizes(stitch->num_cameras, stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height, stitch->num_bands,
				stitch->multibandBlendOffsetIntoBuffer, &stitch->table_sizes.blendOffsetTableSize);
		}
	}
	// disable features if can't be supported with buffer sizes
	if (stitch->EXPO_COMP) {
		if (stitch->table_sizes.expCompValidTableSize == 0 || ((stitch->EXPO_COMP < 3) && stitch->table_sizes.expCompOverlapTableSize == 0)) {
			ls_printf("WARNING: AllocateInternalTablesForCamera: ExpComp has been disabled because of not enough overlap\n");
			stitch->EXPO_COMP = 0;
		}
	}
	if (stitch->SEAM_FIND) {
		if (stitch->table_sizes.seamFindValidTableSize == 0 || stitch->table_sizes.seamFindWeightTableSize == 0 ||
			stitch->table_sizes.seamFindAccumTableSize == 0 || stitch->table_sizes.seamFindPrefInfoTableSize == 0 ||
			stitch->table_sizes.seamFindPathTableSize == 0)
		{
			ls_printf("WARNING: AllocateInternalTablesForCamera: SeamFind has been disabled because of not enough overlap\n");
			stitch->SEAM_FIND = 0;
		}
	}
	if (stitch->MULTIBAND_BLEND) {
		if (stitch->table_sizes.blendOffsetTableSize == 0)
		{
			ls_printf("WARNING: AllocateInternalTablesForCamera: Blend has been disabled because of not enough overlap\n");
			stitch->MULTIBAND_BLEND = 0;
		}
	}
	if (stitch->table_sizes.warpTableSize == 0) {
		ls_printf("ERROR: AllocateInternalTablesForCamera: there are no pixels to warp: check parameters\n");
		return VX_ERROR_INVALID_PARAMETERS;
	}

	// create data objects needed by warp kernel
	vx_enum StitchValidPixelEntryType, StitchWarpRemapEntryType;
	ERROR_CHECK_TYPE_(StitchValidPixelEntryType = vxRegisterUserStruct(stitch->context, sizeof(StitchValidPixelEntry)));
	ERROR_CHECK_TYPE_(StitchWarpRemapEntryType = vxRegisterUserStruct(stitch->context, sizeof(StitchWarpRemapEntry)));
	ERROR_CHECK_OBJECT_(stitch->ValidPixelEntry = vxCreateArray(stitch->context, StitchValidPixelEntryType, stitch->table_sizes.warpTableSize));
	ERROR_CHECK_OBJECT_(stitch->WarpRemapEntry = vxCreateArray(stitch->context, StitchWarpRemapEntryType, stitch->table_sizes.warpTableSize));
	ERROR_CHECK_OBJECT_(stitch->RGBY1 = vxCreateImage(stitch->context, stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height * stitch->num_cameras, VX_DF_IMAGE_RGBX));
	// create data objects needed by merge kernel
	ERROR_CHECK_OBJECT_(stitch->weight_image = vxCreateImage(stitch->context, stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height * stitch->num_cameras, VX_DF_IMAGE_U8));
	ERROR_CHECK_OBJECT_(stitch->cam_id_image = vxCreateImage(stitch->context, (stitch->output_rgb_buffer_width / 8), stitch->output_rgb_buffer_height, VX_DF_IMAGE_U8));
	ERROR_CHECK_OBJECT_(stitch->group1_image = vxCreateImage(stitch->context, (stitch->output_rgb_buffer_width / 8), stitch->output_rgb_buffer_height, VX_DF_IMAGE_U16));
	ERROR_CHECK_OBJECT_(stitch->group2_image = vxCreateImage(stitch->context, (stitch->output_rgb_buffer_width / 8), stitch->output_rgb_buffer_height, VX_DF_IMAGE_U16));
	// create data objects needed by exposure comp kernel
	if (stitch->EXPO_COMP) {
		vx_enum StitchOverlapPixelEntryType, StitchExpCompCalcEntryType;
		vx_float32 one = 1.0f; // initialize gain_array with default gains as one
		ERROR_CHECK_TYPE_(StitchOverlapPixelEntryType = vxRegisterUserStruct(stitch->context, sizeof(StitchOverlapPixelEntry)));
		ERROR_CHECK_TYPE_(StitchExpCompCalcEntryType = vxRegisterUserStruct(stitch->context, sizeof(StitchExpCompCalcEntry)));
		ERROR_CHECK_OBJECT_(stitch->valid_array = vxCreateArray(stitch->context, StitchExpCompCalcEntryType, stitch->table_sizes.expCompValidTableSize));
		if (stitch->EXPO_COMP < 3) {
			ERROR_CHECK_OBJECT_(stitch->OverlapPixelEntry = vxCreateArray(stitch->context, StitchOverlapPixelEntryType, stitch->table_sizes.expCompOverlapTableSize));
		}
		ERROR_CHECK_OBJECT_(stitch->overlap_matrix = vxCreateMatrix(stitch->context, VX_TYPE_INT32, stitch->num_cameras, stitch->num_cameras));
		ERROR_CHECK_OBJECT_(stitch->RGBY2 = vxCreateImage(stitch->context, stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height * stitch->num_cameras, VX_DF_IMAGE_RGBX));
		if (stitch->EXPO_COMP == 1) {
			ERROR_CHECK_OBJECT_(stitch->A_matrix = vxCreateMatrix(stitch->context, VX_TYPE_INT32, stitch->num_cameras, stitch->num_cameras));
			ERROR_CHECK_ALLOC_(stitch->A_matrix_initial_value = new vx_int32[stitch->num_cameras * stitch->num_cameras]());
			ERROR_CHECK_STATUS_(vxWriteMatrix(stitch->A_matrix, stitch->A_matrix_initial_value));
		}
		else if (stitch->EXPO_COMP == 2) {
			ERROR_CHECK_OBJECT_(stitch->A_matrix = vxCreateMatrix(stitch->context, VX_TYPE_INT32, stitch->num_cameras, stitch->num_cameras*3));
			ERROR_CHECK_ALLOC_(stitch->A_matrix_initial_value = new vx_int32[stitch->num_cameras * stitch->num_cameras*3]());
			ERROR_CHECK_STATUS_(vxWriteMatrix(stitch->A_matrix, stitch->A_matrix_initial_value));
		}
		// create gain array
		ERROR_CHECK_OBJECT_(stitch->gain_array = vxCreateArray(stitch->context, VX_TYPE_FLOAT32, stitch->num_cameras * stitch->EXPO_COMP_GAINW * stitch->EXPO_COMP_GAINH*stitch->EXPO_COMP_GAINC));
		ERROR_CHECK_STATUS_(vxAddArrayItems(stitch->gain_array, stitch->num_cameras*stitch->EXPO_COMP_GAINW * stitch->EXPO_COMP_GAINC *stitch->EXPO_COMP_GAINH, &one, 0));
	}
	// create data objects needed by seamfind kernel
	if (stitch->SEAM_FIND) {
		vx_enum StitchSeamFindValidEntryType, StitchSeamFindWeightEntryType;
		vx_enum StitchSeamFindAccumEntryType, StitchSeamFindPreferenceType;
		vx_enum StitchSeamFindInformationType, StitchSeamFindPathEntryType;
		ERROR_CHECK_TYPE_(StitchSeamFindValidEntryType = vxRegisterUserStruct(stitch->context, sizeof(StitchSeamFindValidEntry)));
		ERROR_CHECK_TYPE_(StitchSeamFindWeightEntryType = vxRegisterUserStruct(stitch->context, sizeof(StitchSeamFindWeightEntry)));
		ERROR_CHECK_TYPE_(StitchSeamFindAccumEntryType = vxRegisterUserStruct(stitch->context, sizeof(StitchSeamFindAccumEntry)));
		ERROR_CHECK_TYPE_(StitchSeamFindPreferenceType = vxRegisterUserStruct(stitch->context, sizeof(StitchSeamFindPreference)));
		ERROR_CHECK_TYPE_(StitchSeamFindInformationType = vxRegisterUserStruct(stitch->context, sizeof(StitchSeamFindInformation)));
		ERROR_CHECK_TYPE_(StitchSeamFindPathEntryType = vxRegisterUserStruct(stitch->context, sizeof(StitchSeamFindPathEntry)));
		ERROR_CHECK_OBJECT_(stitch->seamfind_valid_array = vxCreateArray(stitch->context, StitchSeamFindValidEntryType, stitch->table_sizes.seamFindValidTableSize));
		ERROR_CHECK_OBJECT_(stitch->seamfind_weight_array = vxCreateArray(stitch->context, StitchSeamFindWeightEntryType, stitch->table_sizes.seamFindWeightTableSize));
		ERROR_CHECK_OBJECT_(stitch->seamfind_accum_array = vxCreateArray(stitch->context, StitchSeamFindAccumEntryType, stitch->table_sizes.seamFindAccumTableSize));
		ERROR_CHECK_OBJECT_(stitch->seamfind_pref_array = vxCreateArray(stitch->context, StitchSeamFindPreferenceType, stitch->table_sizes.seamFindPrefInfoTableSize));
		ERROR_CHECK_OBJECT_(stitch->seamfind_info_array = vxCreateArray(stitch->context, StitchSeamFindInformationType, stitch->table_sizes.seamFindPrefInfoTableSize));
		ERROR_CHECK_OBJECT_(stitch->seamfind_path_array = vxCreateArray(stitch->context, StitchSeamFindPathEntryType, stitch->table_sizes.seamFindPathTableSize));
		ERROR_CHECK_OBJECT_(stitch->warp_luma_image = vxCreateVirtualImage(stitch->graphStitch, stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height * stitch->num_cameras, VX_DF_IMAGE_U8));
		if (!stitch->SEAM_COST_SELECT) {
			ERROR_CHECK_OBJECT_(stitch->sobelx_image = vxCreateVirtualImage(stitch->graphStitch, stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height * stitch->num_cameras, VX_DF_IMAGE_S16));
			ERROR_CHECK_OBJECT_(stitch->sobely_image = vxCreateVirtualImage(stitch->graphStitch, stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height * stitch->num_cameras, VX_DF_IMAGE_S16));
			ERROR_CHECK_OBJECT_(stitch->sobel_magnitude_s16_image = vxCreateVirtualImage(stitch->graphStitch, stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height * stitch->num_cameras, VX_DF_IMAGE_S16));
		}
		ERROR_CHECK_OBJECT_(stitch->sobel_magnitude_image = vxCreateVirtualImage(stitch->graphStitch, stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height * stitch->num_cameras, VX_DF_IMAGE_U8));
		ERROR_CHECK_OBJECT_(stitch->sobel_phase_image = vxCreateVirtualImage(stitch->graphStitch, stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height * stitch->num_cameras, VX_DF_IMAGE_U8));
		ERROR_CHECK_OBJECT_(stitch->seamfind_weight_image = vxCreateImage(stitch->context, stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height * stitch->num_cameras, VX_DF_IMAGE_U8));
		ERROR_CHECK_OBJECT_(stitch->current_frame = vxCreateScalar(stitch->context, VX_TYPE_UINT32, &stitch->current_frame_value));
		if (stitch->SEAM_REFRESH) {
			vx_enum StitchSeamSceneType;
			ERROR_CHECK_TYPE_(StitchSeamSceneType = vxRegisterUserStruct(stitch->context, sizeof(StitchSeamFindSceneEntry)));
			ERROR_CHECK_OBJECT_(stitch->seamfind_scene_array = vxCreateArray(stitch->context, StitchSeamSceneType, stitch->table_sizes.seamFindPrefInfoTableSize));
			ERROR_CHECK_OBJECT_(stitch->scene_threshold = vxCreateScalar(stitch->context, VX_TYPE_UINT32, &stitch->scene_threshold_value));
		}
		if (stitch->SEAM_COST_SELECT) {
			vx_uint32 cost_enable = 1;
			ERROR_CHECK_OBJECT_(stitch->seam_cost_enable = vxCreateScalar(stitch->context, VX_TYPE_UINT32, &cost_enable));
		}
	}
	// create data objects needed by multiband blend
	if (stitch->MULTIBAND_BLEND) {
		vx_enum StitchBlendValidType;
		ERROR_CHECK_TYPE_(StitchBlendValidType = vxRegisterUserStruct(stitch->context, sizeof(StitchBlendValidEntry)));
		ERROR_CHECK_OBJECT_(stitch->blend_offsets = vxCreateArray(stitch->context, StitchBlendValidType, stitch->table_sizes.blendOffsetTableSize));
		ERROR_CHECK_ALLOC_(stitch->pStitchMultiband = new StitchMultibandData[stitch->num_bands]());
		memset(stitch->pStitchMultiband, 0, sizeof(StitchMultibandData)*stitch->num_bands);
		stitch->pStitchMultiband[0].WeightPyrImgGaussian = stitch->SEAM_FIND ? stitch->seamfind_weight_image : stitch->weight_image;	// for level#0: weight image is mask image after seem find
		stitch->pStitchMultiband[0].DstPyrImgGaussian = stitch->EXPO_COMP ? stitch->RGBY2 : stitch->RGBY1;			// for level#0: dst image is image after exposure_comp
		ERROR_CHECK_OBJECT_(stitch->pStitchMultiband[0].DstPyrImgLaplacian = CreateAlignedImage(stitch, stitch->output_rgb_buffer_width, (stitch->output_rgb_buffer_height * stitch->num_cameras), 8, VX_DF_IMAGE_RGB4_AMD, VX_MEMORY_TYPE_OPENCL));
		ERROR_CHECK_OBJECT_(stitch->pStitchMultiband[0].DstPyrImgLaplacianRec = CreateAlignedImage(stitch, stitch->output_rgb_buffer_width, (stitch->output_rgb_buffer_height * stitch->num_cameras), 8, VX_DF_IMAGE_RGBX, VX_MEMORY_TYPE_OPENCL));
		for (vx_int32 level = 1, levelAlign = 1; level < stitch->num_bands; level++, levelAlign = ((levelAlign << 1) | 1)) {
			vx_uint32 width_l = (stitch->output_rgb_buffer_width + levelAlign) >> level;
			vx_uint32 height_l = ((stitch->output_rgb_buffer_height + levelAlign) >> level) * stitch->num_cameras;
			ERROR_CHECK_OBJECT_(stitch->pStitchMultiband[level].WeightPyrImgGaussian = CreateAlignedImage(stitch, width_l, height_l, 16, VX_DF_IMAGE_U8, VX_MEMORY_TYPE_OPENCL));
			ERROR_CHECK_OBJECT_(stitch->pStitchMultiband[level].DstPyrImgGaussian = CreateAlignedImage(stitch, width_l, height_l, 8, VX_DF_IMAGE_RGBX, VX_MEMORY_TYPE_OPENCL));
			ERROR_CHECK_OBJECT_(stitch->pStitchMultiband[level].DstPyrImgLaplacian = CreateAlignedImage(stitch, width_l, height_l, 8, VX_DF_IMAGE_RGB4_AMD, VX_MEMORY_TYPE_OPENCL));
			ERROR_CHECK_OBJECT_(stitch->pStitchMultiband[level].DstPyrImgLaplacianRec = CreateAlignedImage(stitch, width_l, height_l, 8, VX_DF_IMAGE_RGB4_AMD, VX_MEMORY_TYPE_OPENCL));
		}
		for (int level = 0; level < stitch->num_bands; level++) {
			stitch->pStitchMultiband[level].valid_array_offset = (vx_uint32)stitch->multibandBlendOffsetIntoBuffer[level];
		}
		ERROR_CHECK_OBJECT_(stitch->blend_mask_image = vxCreateImage(stitch->context, stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height * stitch->num_cameras, VX_DF_IMAGE_U8));
	}
    if (stitch->SEAM_FIND || stitch->EXPO_COMP == 1 || stitch->EXPO_COMP == 2) {
		ERROR_CHECK_OBJECT_(stitch->valid_mask_image = vxCreateImage(stitch->context, stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height * stitch->num_cameras, VX_DF_IMAGE_U8));
	}
	vx_status status = VX_FAILURE;

	if (!stitch->SETUP_LOAD_FILES_FOUND){
		// initialize internal tables
		status = InitializeInternalTablesForCamera(stitch);
		if (status != VX_SUCCESS) {
			vxAddLogEntry((vx_reference)stitch->context, status, "ERROR: AllocateInternalTablesForCamera: InitializeInternalTablesForCamera() failed (%d)\n", status);
			return status;
		}
		if (stitch->SETUP_LOAD){
			status = quickSetupDumpTables(stitch);
			if (status != VX_SUCCESS) {
				vxAddLogEntry((vx_reference)stitch->context, status, "ERROR: AllocateInternalTablesForCamera: quickSetupDumpTables() failed (%d)\n", status);
				return status;
			}
		}
	}
	else{
		status = quickSetupLoadTables(stitch);
		if (status != VX_SUCCESS) {
			vxAddLogEntry((vx_reference)stitch->context, status, "ERROR: AllocateInternalTablesForCamera: quickSetupLoadTables() failed (%d)\n", status);
			return status;
		}
	}

	if (!stitch->feature_enable_reinitialize) {
		// release internal buffers as they are not needed anymore
		if (stitch->validPixelCamMap) { delete[] stitch->validPixelCamMap; stitch->validPixelCamMap = nullptr; }
		if (stitch->paddedPixelCamMap) { delete[] stitch->paddedPixelCamMap; stitch->paddedPixelCamMap = nullptr; }
		if (stitch->camSrcMap) { delete[] stitch->camSrcMap; stitch->camSrcMap = nullptr; }
		if (stitch->overlapRectBuf) { delete[] stitch->overlapRectBuf; stitch->overlapRectBuf = nullptr; }
		if (stitch->camIndexTmpBuf) { delete[] stitch->camIndexTmpBuf; stitch->camIndexTmpBuf = nullptr; }
		if (stitch->camIndexBuf) { delete[] stitch->camIndexBuf; stitch->camIndexBuf = nullptr; }
		if (stitch->overlapMatrixBuf) { delete[] stitch->overlapMatrixBuf; stitch->overlapMatrixBuf = nullptr; }
	}

	return VX_SUCCESS;
}
/*****************************************************************************************************************************************
functions to encode stitched output
*****************************************************************************************************************************************/
static vx_status EncodeCreateImageFromROI(ls_context stitch)
{
	vx_uint32 dst_tile_height = (vx_uint32)(stitch->live_stitch_attr[LIVE_STITCH_ATTR_OUTPUT_ENCODER_HEIGHT] / stitch->live_stitch_attr[LIVE_STITCH_ATTR_OUTPUT_TILE_NUM_Y]);

	for (vx_uint32 i = 0; i < stitch->num_encode_sections; i++)
	{
		// create src & dst images from ROI's
		ERROR_CHECK_OBJECT_(stitch->encode_src_rgb_imgs[i] = vxCreateImageFromROI(stitch->Img_output_rgb, &stitch->src_encode_tile_rect[i]));

		// get the destination Image ID and create Img from ROI
		vx_uint32 outputTileID = stitch->dst_encode_tile_rect[i].start_y / dst_tile_height;
		stitch->dst_encode_tile_rect[i].start_y = stitch->dst_encode_tile_rect[i].start_y % dst_tile_height;
		stitch->dst_encode_tile_rect[i].end_y = stitch->dst_encode_tile_rect[i].end_y % dst_tile_height;
		ERROR_CHECK_OBJECT_(stitch->encode_dst_imgs[i] = vxCreateImageFromROI(stitch->encodetileOutput[outputTileID], &stitch->dst_encode_tile_rect[i]));

		// color covert the image ROI's
		ERROR_CHECK_OBJECT(stitch->encode_color_convert_nodes[i] = vxColorConvertNode(stitch->graphStitch, stitch->encode_src_rgb_imgs[i], stitch->encode_dst_imgs[i]));
	}
	return VX_SUCCESS;
}
static vx_status EncodeProcessImage(ls_context stitch)
{
	// src tile dimesions
	vx_uint32 src_tile_width = (vx_uint32)(stitch->output_rgb_buffer_width / stitch->live_stitch_attr[LIVE_STITCH_ATTR_OUTPUT_TILE_NUM_X]);
	vx_uint32 src_tile_height = (vx_uint32)(stitch->output_rgb_buffer_height / stitch->live_stitch_attr[LIVE_STITCH_ATTR_OUTPUT_TILE_NUM_Y]);
	vx_uint32 src_tile_overlap = (vx_uint32)stitch->live_stitch_attr[LIVE_STITCH_ATTR_OUTPUT_src_tile_overlap];
	// dst tile dimesions
	vx_uint32 dst_tile_width = (vx_uint32)(stitch->live_stitch_attr[LIVE_STITCH_ATTR_OUTPUT_ENCODER_WIDTH] / stitch->live_stitch_attr[LIVE_STITCH_ATTR_OUTPUT_TILE_NUM_X]);
	vx_uint32 dst_tile_height = (vx_uint32)(stitch->live_stitch_attr[LIVE_STITCH_ATTR_OUTPUT_ENCODER_HEIGHT] / stitch->live_stitch_attr[LIVE_STITCH_ATTR_OUTPUT_TILE_NUM_Y]);

	// calculate encode regions
	vx_uint32 roi_img = 0;
	vx_uint32 src_tile_start_y = 0, dst_tile_start_y = 0;
	for (vx_uint32 j = 0; j < (vx_uint32)stitch->live_stitch_attr[LIVE_STITCH_ATTR_OUTPUT_TILE_NUM_Y]; j++){
		vx_uint32  src_tile_start_x = 0, dst_tile_start_x = 0;
		for (vx_uint32 i = 0; i < (vx_uint32)stitch->live_stitch_attr[LIVE_STITCH_ATTR_OUTPUT_TILE_NUM_X]; i++){
			// source tile rectangle
			vx_int32 tstart_x = src_tile_start_x - src_tile_overlap, tend_x = src_tile_start_x + src_tile_width + src_tile_overlap;
			vx_int32 tstart_y = src_tile_start_y - src_tile_overlap; if (tstart_y < 0) { tstart_y = 0; }
			vx_int32 tend_y = src_tile_start_y + src_tile_height + src_tile_overlap; if (tend_y >(vx_int32)stitch->output_rgb_buffer_height) { tend_y = (vx_int32)stitch->output_rgb_buffer_height; }
			// destination tile rectangle
			vx_int32 output_tstart_x = dst_tile_start_x, output_tend_x = dst_tile_start_x + dst_tile_width;
			vx_int32 output_tstart_y = dst_tile_start_y, output_tend_y = dst_tile_start_y + dst_tile_height;
			// tile region warp around
			if (tstart_x < 0){
				// overlap rectangle
				stitch->src_encode_tile_rect[roi_img].start_x = stitch->output_rgb_buffer_width - src_tile_overlap;	stitch->src_encode_tile_rect[roi_img].end_x = stitch->output_rgb_buffer_width;
				stitch->src_encode_tile_rect[roi_img].start_y = tstart_y; stitch->src_encode_tile_rect[roi_img].end_y = tend_y;
				// output overlap rectangle
				stitch->dst_encode_tile_rect[roi_img].start_x = output_tstart_x;	stitch->dst_encode_tile_rect[roi_img].end_x = output_tstart_x + src_tile_overlap;
				stitch->dst_encode_tile_rect[roi_img].start_y = output_tstart_y; stitch->dst_encode_tile_rect[roi_img].end_y = output_tstart_y + (tend_y - tstart_y);
				roi_img++;
				// image rectangle
				stitch->src_encode_tile_rect[roi_img].start_x = 0; stitch->src_encode_tile_rect[roi_img].end_x = tend_x;
				stitch->src_encode_tile_rect[roi_img].start_y = tstart_y; stitch->src_encode_tile_rect[roi_img].end_y = tend_y;
				// output image rectangle
				stitch->dst_encode_tile_rect[roi_img].start_x = output_tstart_x + src_tile_overlap;	stitch->dst_encode_tile_rect[roi_img].end_x = stitch->dst_encode_tile_rect[roi_img].start_x + tend_x;
				stitch->dst_encode_tile_rect[roi_img].start_y = output_tstart_y; stitch->dst_encode_tile_rect[roi_img].end_y = output_tstart_y + (tend_y - tstart_y);
				roi_img++;
			}
			// tile region
			if (tstart_x >= 0 && tend_x <= (vx_int32)stitch->output_rgb_buffer_width){
				// image rectangle
				stitch->src_encode_tile_rect[roi_img].start_x = tstart_x; stitch->src_encode_tile_rect[roi_img].end_x = tend_x;
				stitch->src_encode_tile_rect[roi_img].start_y = tstart_y; stitch->src_encode_tile_rect[roi_img].end_y = tend_y;
				// output image rectangle
				stitch->dst_encode_tile_rect[roi_img].start_x = output_tstart_x;	stitch->dst_encode_tile_rect[roi_img].end_x = output_tend_x;
				stitch->dst_encode_tile_rect[roi_img].start_y = output_tstart_y; stitch->dst_encode_tile_rect[roi_img].end_y = output_tstart_y + (tend_y - tstart_y);
				roi_img++;
			}
			// tile region warp around
			if (tstart_x >= 0 && tend_x > (vx_int32)stitch->output_rgb_buffer_width){
				// image rectangle
				stitch->src_encode_tile_rect[roi_img].start_x = tstart_x; stitch->src_encode_tile_rect[roi_img].end_x = stitch->output_rgb_buffer_width;
				stitch->src_encode_tile_rect[roi_img].start_y = tstart_y; stitch->src_encode_tile_rect[roi_img].end_y = tend_y;
				// output image rectangle
				stitch->dst_encode_tile_rect[roi_img].start_x = output_tstart_x;	stitch->dst_encode_tile_rect[roi_img].end_x = output_tstart_x + (stitch->output_rgb_buffer_width - tstart_x);
				stitch->dst_encode_tile_rect[roi_img].start_y = output_tstart_y; stitch->dst_encode_tile_rect[roi_img].end_y = output_tstart_y + (tend_y - tstart_y);
				roi_img++;
				// overlap rectangle
				stitch->src_encode_tile_rect[roi_img].start_x = 0; stitch->src_encode_tile_rect[roi_img].end_x = src_tile_overlap;
				stitch->src_encode_tile_rect[roi_img].start_y = tstart_y; stitch->src_encode_tile_rect[roi_img].end_y = tend_y;
				// output image rectangle
				stitch->dst_encode_tile_rect[roi_img].start_x = output_tstart_x + (stitch->output_rgb_buffer_width - tstart_x); stitch->dst_encode_tile_rect[roi_img].end_x = stitch->dst_encode_tile_rect[roi_img].start_x + src_tile_overlap;
				stitch->dst_encode_tile_rect[roi_img].start_y = output_tstart_y; stitch->dst_encode_tile_rect[roi_img].end_y = output_tstart_y + (tend_y - tstart_y);
				roi_img++;
			}
			src_tile_start_x += src_tile_width;
			dst_tile_start_y += dst_tile_height;
		} 
		src_tile_start_y += src_tile_height;
	}
	stitch->num_encode_sections = roi_img;
	
	return VX_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////
// Stitch API implementation

//! \brief Get version.
LIVE_STITCH_API_ENTRY const char * VX_API_CALL lsGetVersion()
{
	return LS_VERSION;
}

//! \brief Set callback for log messages.
LIVE_STITCH_API_ENTRY void VX_API_CALL lsGlobalSetLogCallback(stitch_log_callback_f callback)
{
	g_live_stitch_log_message_callback = callback;
}

//! \brief Set global attributes. Note that current global attributes will become default attributes for when a stitch context is created.
LIVE_STITCH_API_ENTRY vx_status VX_API_CALL lsGlobalSetAttributes(vx_uint32 attr_offset, vx_uint32 attr_count, const vx_float32 * attr_ptr)
{
	// make sure global attributes are reset to default
	if (!g_live_stitch_attr_initialized) ResetLiveStitchGlobalAttributes();

	// bounding check
	if ((attr_offset + attr_count) > LIVE_STITCH_ATTR_MAX_COUNT)
		return VX_ERROR_INVALID_DIMENSION;

	// set global live_stitch_attr[]
	memcpy(&g_live_stitch_attr[attr_offset], attr_ptr, attr_count * sizeof(vx_float32));
	return VX_SUCCESS;
}

//! \brief Get global attributes. Note that current global attributes will become default attributes for when a stitch context is created.
LIVE_STITCH_API_ENTRY vx_status VX_API_CALL lsGlobalGetAttributes(vx_uint32 attr_offset, vx_uint32 attr_count, vx_float32 * attr_ptr)
{
	// make sure global attributes are reset to default
	if (!g_live_stitch_attr_initialized) ResetLiveStitchGlobalAttributes();

	// bounding check
	if ((attr_offset + attr_count) > LIVE_STITCH_ATTR_MAX_COUNT)
		return VX_ERROR_INVALID_DIMENSION;

	// get global live_stitch_attr[]
	memcpy(attr_ptr, &g_live_stitch_attr[attr_offset], attr_count * sizeof(vx_float32));
	return VX_SUCCESS;
}

//! \brief Create stitch context.
LIVE_STITCH_API_ENTRY ls_context VX_API_CALL lsCreateContext()
{
	PROFILER_INITIALIZE();
	// make sure global attributes are reset to default
	if (!g_live_stitch_attr_initialized) ResetLiveStitchGlobalAttributes();

	/////////////////////////////////////////////////////////
	// create stitch handle and initialize live_stitch_attr
	ls_context stitch = new ls_context_t;
	if (stitch) {
		memset(stitch, 0, sizeof(ls_context_t));
		memcpy(stitch->live_stitch_attr, g_live_stitch_attr, sizeof(stitch->live_stitch_attr));
		stitch->magic = LIVE_STITCH_MAGIC;
	}
	return stitch;
}

//! \brief Set context specific attributes.
LIVE_STITCH_API_ENTRY vx_status VX_API_CALL lsSetAttributes(ls_context stitch, vx_uint32 attr_offset, vx_uint32 attr_count, const vx_float32 * attr_ptr)
{
	ERROR_CHECK_STATUS_(IsValidContext(stitch));

	// bounding check
	if ((attr_offset + attr_count) > LIVE_STITCH_ATTR_MAX_COUNT)
		return VX_ERROR_INVALID_DIMENSION;

	for (vx_uint32 attr = attr_offset; attr < (attr_offset + attr_count); attr++) {
		if (attr == LIVE_STITCH_ATTR_SEAM_THRESHOLD) {
			// update scalar of seafind k0 kernel
			stitch->scene_threshold_value = (vx_uint32)stitch->live_stitch_attr[LIVE_STITCH_ATTR_SEAM_THRESHOLD];
			if (stitch->scene_threshold) {
				vx_status status = vxWriteScalarValue(stitch->scene_threshold, &stitch->scene_threshold_value);
				if (status != VX_SUCCESS)
					return status;
			}
		}
		else if (attr == LIVE_STITCH_ATTR_NOISE_FILTER_LAMBDA) {
			// update scalar of seafind k0 kernel
			stitch->noiseFilterLambda = (vx_float32)stitch->live_stitch_attr[LIVE_STITCH_ATTR_NOISE_FILTER_LAMBDA];
			if (stitch->filterLambda) {
				vx_status status = vxWriteScalarValue(stitch->filterLambda, &stitch->noiseFilterLambda);
				if (status != VX_SUCCESS)
					return status;
			}
		}
		else {
			// not all attributes are supported
			return VX_ERROR_NOT_SUPPORTED;
		}
	}

	// update live_stitch_attr to reflect recent changes
	memcpy(&stitch->live_stitch_attr[attr_offset], attr_ptr, attr_count * sizeof(vx_float32));
	return VX_SUCCESS;
}

//! \brief Get context specific attributes.
LIVE_STITCH_API_ENTRY vx_status VX_API_CALL lsGetAttributes(ls_context stitch, vx_uint32 attr_offset, vx_uint32 attr_count, vx_float32 * attr_ptr)
{
	ERROR_CHECK_STATUS_(IsValidContext(stitch));

	// bounding check
	if ((attr_offset + attr_count) > LIVE_STITCH_ATTR_MAX_COUNT)
		return VX_ERROR_INVALID_DIMENSION;

	// get live_stitch_attr
	memcpy(attr_ptr, &stitch->live_stitch_attr[attr_offset], attr_count * sizeof(vx_float32));
	return VX_SUCCESS;
}

//! \brief Set stitch configuration.
LIVE_STITCH_API_ENTRY vx_status VX_API_CALL lsSetOpenVXContext(ls_context stitch, vx_context  openvx_context)
{
	ERROR_CHECK_STATUS_(IsValidContextAndNotInitialized(stitch));
	if (stitch->context) {
		ls_printf("ERROR: lsSetOpenVXContext: OpenVX context already exists\n");
		return VX_ERROR_NOT_SUPPORTED;
	}
	stitch->context = openvx_context;
	stitch->context_is_external = true;
	if (stitch->opencl_context) {
		ERROR_CHECK_STATUS_(vxSetContextAttribute(stitch->context, VX_CONTEXT_ATTRIBUTE_AMD_OPENCL_CONTEXT, &stitch->opencl_context, sizeof(cl_context)));
	}
	return VX_SUCCESS;
}

LIVE_STITCH_API_ENTRY vx_status VX_API_CALL lsSetOpenCLContext(ls_context stitch, cl_context  opencl_context)
{
	ERROR_CHECK_STATUS_(IsValidContextAndNotInitialized(stitch));
	stitch->opencl_context = opencl_context;
	return VX_SUCCESS;
}

LIVE_STITCH_API_ENTRY vx_status VX_API_CALL lsSetRigParams(ls_context stitch, const rig_params * par)
{
	ERROR_CHECK_STATUS_(IsValidContext(stitch));
	if (stitch->initialized && !stitch->feature_enable_reinitialize) {
		ls_printf("ERROR: lsSetRigParams: lsReinitialize has been disabled\n");
		return VX_ERROR_NOT_SUPPORTED;
	}
	memcpy(&stitch->rig_par, par, sizeof(stitch->rig_par));
	// check and mark whether reinitialize is required
	if (stitch->initialized) {
		stitch->reinitialize_required = true;
		stitch->rig_params_updated = true;
	}
	return VX_SUCCESS;
}

LIVE_STITCH_API_ENTRY vx_status VX_API_CALL lsSetCameraConfig(ls_context stitch, vx_uint32 num_camera_rows, vx_uint32 num_camera_columns, vx_df_image buffer_format, vx_uint32 buffer_width, vx_uint32 buffer_height)
{
	ERROR_CHECK_STATUS_(IsValidContextAndNotInitialized(stitch));
	if (num_camera_rows * num_camera_columns > LIVE_STITCH_MAX_CAMERAS) {
		ls_printf("ERROR: this release supports upto %d cameras only\n", LIVE_STITCH_MAX_CAMERAS);
		return VX_ERROR_NOT_SUPPORTED;
	}
	if (buffer_format != VX_DF_IMAGE_UYVY && buffer_format != VX_DF_IMAGE_YUYV && buffer_format != VX_DF_IMAGE_RGB && buffer_format != VX_DF_IMAGE_NV12 && buffer_format != VX_DF_IMAGE_IYUV) {
		ls_printf("ERROR: lsSetCameraConfig: only UYVY/YUYV/RGB/NV12/IYUV buffer formats are allowed\n");
		return VX_ERROR_INVALID_FORMAT;
	}
	// check num rows and columns
	if (num_camera_rows < 1 || num_camera_columns < 1 ||
		(buffer_width % num_camera_columns) != 0 ||
		(buffer_height % num_camera_rows) != 0)
	{
		ls_printf("ERROR: lsSetCameraConfig: dimensions are is not multiple of camera rows & columns\n");
		return VX_ERROR_INVALID_DIMENSION;
	}
	// check that camera dimensions are multiples of 16x2 and width must be less than 8K
	if (((buffer_width / num_camera_columns) % 16) != 0 || ((buffer_height / num_camera_rows) % 2) != 0 || std::max(buffer_width, buffer_height / num_camera_rows) >= 8192) {
		ls_printf("ERROR: lsSetCameraConfig: camera dimensions are required to be multiple of 16x2 and width less than 8K\n");
		return VX_ERROR_INVALID_DIMENSION;
	}
	// set configuration parameters
	stitch->num_cameras = num_camera_rows * num_camera_columns;
	stitch->num_camera_rows = num_camera_rows;
	stitch->num_camera_columns = num_camera_columns;
	stitch->camera_buffer_format = buffer_format;
	stitch->camera_buffer_width = buffer_width;
	stitch->camera_buffer_height = buffer_height;
	if (buffer_format != VX_DF_IMAGE_NV12 && buffer_format != VX_DF_IMAGE_IYUV){ stitch->camera_buffer_stride_in_bytes = buffer_width * (buffer_format == VX_DF_IMAGE_RGB ? 3 : 2); }
	else{ stitch->camera_buffer_stride_in_bytes = buffer_width; }
	ERROR_CHECK_ALLOC_(stitch->camera_par = new camera_params[stitch->num_cameras]());
	stitch->camera_rgb_buffer_width = stitch->camera_buffer_width;
	stitch->camera_rgb_buffer_height = stitch->camera_buffer_height;
	// set default orientations
	for (vx_uint32 i = 0; i < stitch->num_cameras; i++) {
		stitch->camera_par[i].focal.yaw = -180.0f + 360.0f * (float)i / (float)stitch->num_cameras;
	}
	return VX_SUCCESS;
}

LIVE_STITCH_API_ENTRY vx_status VX_API_CALL lsSetOutputConfig(ls_context stitch, vx_df_image buffer_format, vx_uint32 buffer_width, vx_uint32 buffer_height)
{
	ERROR_CHECK_STATUS_(IsValidContextAndNotInitialized(stitch));
	if (buffer_format != VX_DF_IMAGE_UYVY && buffer_format != VX_DF_IMAGE_YUYV && buffer_format != VX_DF_IMAGE_RGB && buffer_format != VX_DF_IMAGE_NV12 && buffer_format != VX_DF_IMAGE_IYUV) {
		ls_printf("ERROR: lsSetOutputConfig: only UYVY/YUYV/RGB/NV12/IYUV buffer formats are allowed\n");
		return VX_ERROR_INVALID_FORMAT;
	}
	if (buffer_width != (buffer_height * 2)) {
		ls_printf("ERROR: lsSetOutputConfig: buffer_width should be 2 times buffer_height\n");
		return VX_ERROR_INVALID_DIMENSION;
	}
	// check that dimensions are multiples of 16x2
	if ((buffer_width % 16) != 0 || (buffer_height % 2) != 0) {
		ls_printf("ERROR: lsSetOutputConfig: output dimensions are required to be multiple of 16x2\n");
		return VX_ERROR_INVALID_DIMENSION;
	}
	// set configuration parameters
	stitch->output_buffer_format = buffer_format;
	stitch->output_buffer_width = buffer_width;
	stitch->output_buffer_height = buffer_height;
	if (buffer_format != VX_DF_IMAGE_NV12 && buffer_format != VX_DF_IMAGE_IYUV){ stitch->output_buffer_stride_in_bytes = buffer_width * (buffer_format == VX_DF_IMAGE_RGB ? 3 : 2); }
	else{ stitch->output_buffer_stride_in_bytes = buffer_width; }
	stitch->output_rgb_buffer_width = stitch->output_buffer_width;
	stitch->output_rgb_buffer_height = stitch->output_buffer_height;

	return VX_SUCCESS;
}

LIVE_STITCH_API_ENTRY vx_status VX_API_CALL lsSetOverlayConfig(ls_context stitch, vx_uint32 num_overlay_rows, vx_uint32 num_overlay_columns, vx_df_image buffer_format, vx_uint32 buffer_width, vx_uint32 buffer_height)
{
	ERROR_CHECK_STATUS_(IsValidContextAndNotInitialized(stitch));
	if (num_overlay_rows * num_overlay_columns > LIVE_STITCH_MAX_CAMERAS) {
		ls_printf("ERROR: this release supports upto %d cameras only\n", LIVE_STITCH_MAX_CAMERAS);
		return VX_ERROR_NOT_SUPPORTED;
	}
	if (buffer_format != VX_DF_IMAGE_RGBX) {
		ls_printf("ERROR: lsSetOverlayConfig: only RGBX buffer formats are allowed\n");
		return VX_ERROR_INVALID_FORMAT;
	}
	// check num rows and columns
	if (num_overlay_rows < 1 || num_overlay_columns < 1 ||
		(buffer_width % num_overlay_columns) != 0 ||
		(buffer_height % num_overlay_rows) != 0)
	{
		ls_printf("ERROR: lsSetOverlayConfig: dimensions are is not multiple of overlay rows and columns\n");
		return VX_ERROR_INVALID_DIMENSION;
	}
	// check that overlay dimensions are multiples of 16x2 and width less than 8K
	if (((buffer_width / num_overlay_columns) % 16) != 0 || ((buffer_height / num_overlay_rows) % 2) != 0 || std::max(buffer_width, buffer_height / num_overlay_columns) >= 8192) {
		ls_printf("ERROR: lsSetOverlayConfig: overlay dimensions are required to be multiple of 16x2 and width is less than 8K\n");
		return VX_ERROR_INVALID_DIMENSION;
	}
	// set configuration parameters
	stitch->num_overlays = num_overlay_rows * num_overlay_columns;
	stitch->num_overlay_rows = num_overlay_rows;
	stitch->num_overlay_columns = num_overlay_columns;
	stitch->overlay_buffer_width = buffer_width;
	stitch->overlay_buffer_height = buffer_height;
	stitch->overlay_buffer_stride_in_bytes = buffer_width * 4;
	ERROR_CHECK_ALLOC_(stitch->overlay_par = new camera_params[stitch->num_overlays]());
	// set default orientations
	stitch->overlay_par[0].focal.pitch = -90.0f;
	for (vx_uint32 i = 1; i < stitch->num_overlays; i++) {
		// set default overlay locations
		stitch->overlay_par[i].focal.pitch = -90.0f + 180.0f * (float)i / (float)(stitch->num_overlays-1);
	}
	return VX_SUCCESS;
}

LIVE_STITCH_API_ENTRY vx_status VX_API_CALL lsSetCameraParams(ls_context stitch, vx_uint32 cam_index, const camera_params * par)
{
	ERROR_CHECK_STATUS_(IsValidContext(stitch));
	if (cam_index >= stitch->num_cameras) {
		ls_printf("ERROR: lsSetCameraParams: invalid camera index (%d)\n", cam_index);
		return VX_ERROR_INVALID_VALUE;
	}
	if (stitch->initialized && !stitch->feature_enable_reinitialize) {
		ls_printf("ERROR: lsSetCameraParams: lsReinitialize has been disabled\n");
		return VX_ERROR_NOT_SUPPORTED;
	}
	memcpy(&stitch->camera_par[cam_index], par, sizeof(camera_params));
	// check and mark whether reinitialize is required
	if (stitch->initialized) {
		stitch->reinitialize_required = true;
		stitch->camera_params_updated = true;
	}
	return VX_SUCCESS;
}

LIVE_STITCH_API_ENTRY vx_status VX_API_CALL lsSetOverlayParams(ls_context stitch, vx_uint32 overlay_index, const camera_params * par)
{
	ERROR_CHECK_STATUS_(IsValidContext(stitch));
	if (overlay_index >= stitch->num_overlays) {
		ls_printf("ERROR: lsSetOverlayParams: invalid overlay index (%d)\n", overlay_index);
		return VX_ERROR_INVALID_VALUE;
	}
	if (stitch->initialized && !stitch->feature_enable_reinitialize) {
		ls_printf("ERROR: lsSetOverlayParams: lsReinitialize has been disabled\n");
		return VX_ERROR_NOT_SUPPORTED;
	}
	memcpy(&stitch->overlay_par[overlay_index], par, sizeof(camera_params));
	// check and mark whether reinitialize is required
	if (stitch->initialized) {
		stitch->reinitialize_required = true;
		stitch->overlay_params_updated = true;
	}
	return VX_SUCCESS;
}

LIVE_STITCH_API_ENTRY vx_status VX_API_CALL lsSetCameraBufferStride(ls_context stitch, vx_uint32 camera_buffer_stride_in_bytes)
{
	ERROR_CHECK_STATUS_(IsValidContextAndNotInitialized(stitch));
	if ((camera_buffer_stride_in_bytes % 16) != 0) {
		ls_printf("ERROR: lsSetCameraBufferStride: stride has to be a multiple of 16\n");
		return VX_ERROR_INVALID_DIMENSION;
	}
	stitch->camera_buffer_stride_in_bytes = camera_buffer_stride_in_bytes;
	return VX_SUCCESS;
}

LIVE_STITCH_API_ENTRY vx_status VX_API_CALL lsSetOutputBufferStride(ls_context stitch, vx_uint32 output_buffer_stride_in_bytes)
{
	ERROR_CHECK_STATUS_(IsValidContextAndNotInitialized(stitch));
	if ((output_buffer_stride_in_bytes % 16) != 0) {
		ls_printf("ERROR: lsSetOutputBufferStride: stride has to be a multiple of 16\n");
		return VX_ERROR_INVALID_DIMENSION;
	}
	stitch->output_buffer_stride_in_bytes = output_buffer_stride_in_bytes;
	return VX_SUCCESS;
}

LIVE_STITCH_API_ENTRY vx_status VX_API_CALL lsSetOverlayBufferStride(ls_context stitch, vx_uint32 overlay_buffer_stride_in_bytes)
{
	ERROR_CHECK_STATUS_(IsValidContextAndNotInitialized(stitch));
	if ((overlay_buffer_stride_in_bytes % 16) != 0) {
		ls_printf("ERROR: lsSetOverlayBufferStride: stride has to be a multiple of 16\n");
		return VX_ERROR_INVALID_DIMENSION;
	}
	stitch->overlay_buffer_stride_in_bytes = overlay_buffer_stride_in_bytes;
	return VX_SUCCESS;
}

LIVE_STITCH_API_ENTRY vx_status VX_API_CALL lsSetCameraModule(ls_context stitch, const char * module, const char * kernelName, const char * kernelArguments)
{
	ERROR_CHECK_STATUS_(IsValidContextAndNotInitialized(stitch));
	strncpy(stitch->loomio_camera.module, module, LOOMIO_MAX_LENGTH_MODULE_NAME-1);
	strncpy(stitch->loomio_camera.kernelName, kernelName, LOOMIO_MAX_LENGTH_KERNEL_NAME-1);
	strncpy(stitch->loomio_camera.kernelArguments, kernelArguments, LOOMIO_MAX_LENGTH_KERNEL_ARGUMENTS-1);
	return VX_SUCCESS;
}

LIVE_STITCH_API_ENTRY vx_status VX_API_CALL lsSetOutputModule(ls_context stitch, const char * module, const char * kernelName, const char * kernelArguments)
{
	ERROR_CHECK_STATUS_(IsValidContextAndNotInitialized(stitch));
	strncpy(stitch->loomio_output.module, module, LOOMIO_MAX_LENGTH_MODULE_NAME-1);
	strncpy(stitch->loomio_output.kernelName, kernelName, LOOMIO_MAX_LENGTH_KERNEL_NAME-1);
	strncpy(stitch->loomio_output.kernelArguments, kernelArguments, LOOMIO_MAX_LENGTH_KERNEL_ARGUMENTS-1);
	return VX_SUCCESS;
}

LIVE_STITCH_API_ENTRY vx_status VX_API_CALL lsSetOverlayModule(ls_context stitch, const char * module, const char * kernelName, const char * kernelArguments)
{
	ERROR_CHECK_STATUS_(IsValidContextAndNotInitialized(stitch));
	strncpy(stitch->loomio_overlay.module, module, LOOMIO_MAX_LENGTH_MODULE_NAME-1);
	strncpy(stitch->loomio_overlay.kernelName, kernelName, LOOMIO_MAX_LENGTH_KERNEL_NAME-1);
	strncpy(stitch->loomio_overlay.kernelArguments, kernelArguments, LOOMIO_MAX_LENGTH_KERNEL_ARGUMENTS-1);
	return VX_SUCCESS;
}

LIVE_STITCH_API_ENTRY vx_status VX_API_CALL lsSetViewingModule(ls_context stitch, const char * module, const char * kernelName, const char * kernelArguments)
{
	ERROR_CHECK_STATUS_(IsValidContextAndNotInitialized(stitch));
	strncpy(stitch->loomio_viewing.module, module, LOOMIO_MAX_LENGTH_MODULE_NAME - 1);
	strncpy(stitch->loomio_viewing.kernelName, kernelName, LOOMIO_MAX_LENGTH_KERNEL_NAME - 1);
	strncpy(stitch->loomio_viewing.kernelArguments, kernelArguments, LOOMIO_MAX_LENGTH_KERNEL_ARGUMENTS - 1);
	return VX_SUCCESS;
}

//! \brief initialize the stitch context.
LIVE_STITCH_API_ENTRY vx_status VX_API_CALL lsInitialize(ls_context stitch)
{
	PROFILER_START(LoomSL, InitializeGraph);
	ERROR_CHECK_STATUS_(IsValidContextAndNotInitialized(stitch));

	/////////////////////////////////////////////////////////
	// pick default stitch mode and aux data length
	stitch->stitching_mode = stitching_mode_normal;
	if (stitch->live_stitch_attr[LIVE_STITCH_ATTR_STITCH_MODE] == (float)stitching_mode_quick_and_dirty)
		stitch->stitching_mode = stitching_mode_quick_and_dirty;
	if (stitch->live_stitch_attr[LIVE_STITCH_ATTR_ENABLE_REINITIALIZE] == 1.0f)
		stitch->feature_enable_reinitialize = true;
	stitch->loomioOutputAuxSelection = (vx_uint32)stitch->live_stitch_attr[LIVE_STITCH_ATTR_IO_OUTPUT_AUX_SELECTION];
	stitch->loomioCameraAuxDataLength = std::min((vx_uint32)stitch->live_stitch_attr[LIVE_STITCH_ATTR_IO_CAMERA_AUX_DATA_SIZE], (vx_uint32)LOOMIO_MIN_AUX_DATA_CAPACITY);
	stitch->loomioOverlayAuxDataLength = std::min((vx_uint32)stitch->live_stitch_attr[LIVE_STITCH_ATTR_IO_OVERLAY_AUX_DATA_SIZE], (vx_uint32)LOOMIO_MIN_AUX_DATA_CAPACITY);
	stitch->loomioOutputAuxDataLength = std::min((vx_uint32)stitch->live_stitch_attr[LIVE_STITCH_ATTR_IO_OUTPUT_AUX_DATA_SIZE], (vx_uint32)LOOMIO_MIN_AUX_DATA_CAPACITY);

	/////////////////////////////////////////////////////////
	// create and initialize OpenVX context and graphs
	if (!stitch->context) {
		stitch->context = vxCreateContext();
		if (stitch->opencl_context) {
			ERROR_CHECK_STATUS_(vxSetContextAttribute(stitch->context, VX_CONTEXT_ATTRIBUTE_AMD_OPENCL_CONTEXT, &stitch->opencl_context, sizeof(cl_context)));
		}
	}
	ERROR_CHECK_OBJECT_(stitch->context);
	vxRegisterLogCallback(stitch->context, log_callback, vx_false_e);
	ERROR_CHECK_STATUS_(vxPublishKernels(stitch->context));
	ERROR_CHECK_OBJECT_(stitch->graphStitch = vxCreateGraph(stitch->context));
	if (stitch->live_stitch_attr[LIVE_STITCH_ATTR_PROFILER] == 2.0f) {
		ERROR_CHECK_STATUS_(vxDirective((vx_reference)stitch->graphStitch, VX_DIRECTIVE_AMD_ENABLE_PROFILE_CAPTURE));
	}

	// creating OpenVX image objects for input & output OpenCL buffers
	if (strlen(stitch->loomio_camera.kernelName) > 0) {
		// load OpenVX module (if specified)
		if (strlen(stitch->loomio_camera.module) > 0) {
			vx_status status = vxLoadKernels(stitch->context, stitch->loomio_camera.module);
			if (status != VX_SUCCESS) {
				ls_printf("ERROR: lsInitialize: vxLoadKernels(%s) failed (%d)\n", stitch->loomio_camera.module, status);
				return VX_ERROR_INVALID_PARAMETERS;
			}
		}
		// instantiate specified node into the graph
		ERROR_CHECK_OBJECT_(stitch->cameraMediaConfig = vxCreateScalar(stitch->context, VX_TYPE_STRING_AMD, stitch->loomio_camera.kernelArguments));
		ERROR_CHECK_OBJECT_(stitch->Img_input = vxCreateVirtualImage(stitch->graphStitch, stitch->camera_buffer_width, stitch->camera_buffer_height, stitch->camera_buffer_format));
		ERROR_CHECK_OBJECT_(stitch->loomioCameraAuxData = vxCreateArray(stitch->context, VX_TYPE_UINT8, stitch->loomioCameraAuxDataLength));
		vx_reference params[] = {
			(vx_reference)stitch->cameraMediaConfig,
			(vx_reference)stitch->Img_input,
			(vx_reference)stitch->loomioCameraAuxData,
		};
		ERROR_CHECK_OBJECT_(stitch->nodeLoomIoCamera = stitchCreateNode(stitch->graphStitch, stitch->loomio_camera.kernelName, params, dimof(params)));
	}
	else {
		// need image created from OpenCL handle
		if (stitch->camera_buffer_format == VX_DF_IMAGE_NV12 || stitch->camera_buffer_format == VX_DF_IMAGE_IYUV){
			vx_imagepatch_addressing_t addr_in[3] = { 0, 0, 0 };
			void *ptr[3] = { nullptr, nullptr, nullptr };
			addr_in[0].dim_x = stitch->camera_buffer_width;
			addr_in[0].dim_y = stitch->camera_buffer_height;
			addr_in[0].stride_x = 1;
			addr_in[0].stride_y = stitch->camera_buffer_stride_in_bytes;
			if (stitch->camera_buffer_format == VX_DF_IMAGE_NV12){
				addr_in[1].dim_x = stitch->camera_buffer_width;	addr_in[1].dim_y = stitch->camera_buffer_height >> 1;
				addr_in[1].stride_x = 2; addr_in[1].stride_y = stitch->camera_buffer_stride_in_bytes;
			}
			else{
				addr_in[1].dim_x = stitch->camera_buffer_width;	addr_in[1].dim_y = stitch->camera_buffer_height;
				addr_in[1].stride_x = 1; addr_in[1].stride_y = stitch->camera_buffer_stride_in_bytes;
				addr_in[2].dim_x = stitch->camera_buffer_width;	addr_in[2].dim_y = stitch->camera_buffer_height;
				addr_in[2].stride_x = 1; addr_in[2].stride_y = stitch->camera_buffer_stride_in_bytes;
			}
			ERROR_CHECK_OBJECT_(stitch->Img_input = vxCreateImageFromHandle(stitch->context, stitch->camera_buffer_format, &addr_in[0], ptr, VX_MEMORY_TYPE_OPENCL));
		}
		else{
			vx_imagepatch_addressing_t addr_in = { 0 };
			void *ptr[1] = { nullptr };
			addr_in.dim_x = stitch->camera_buffer_width;
			addr_in.dim_y = (stitch->camera_buffer_height);
			addr_in.stride_x = (stitch->camera_buffer_format == VX_DF_IMAGE_RGB) ? 3 : 2;
			addr_in.stride_y = stitch->camera_buffer_stride_in_bytes;
			if (addr_in.stride_y == 0) addr_in.stride_y = addr_in.stride_x * addr_in.dim_x;
			ERROR_CHECK_OBJECT_(stitch->Img_input = vxCreateImageFromHandle(stitch->context, stitch->camera_buffer_format, &addr_in, ptr, VX_MEMORY_TYPE_OPENCL));
		}
	}
	// check attribute for fast init code
	stitch->USE_CPU_INIT = (vx_uint32)stitch->live_stitch_attr[LIVE_STITCH_ATTR_USE_CPU_FOR_INIT];
	stitch->stitchInitData = nullptr;

	if (stitch->num_overlays > 0) {
		// create overlay image
		if (strlen(stitch->loomio_overlay.kernelName) > 0) {
			// load OpenVX module (if specified)
			if (strlen(stitch->loomio_overlay.module) > 0) {
				vx_status status = vxLoadKernels(stitch->context, stitch->loomio_overlay.module);
				if (status != VX_SUCCESS) {
					ls_printf("ERROR: lsInitialize: vxLoadKernels(%s) failed (%d)\n", stitch->loomio_overlay.module, status);
					return VX_ERROR_INVALID_PARAMETERS;
				}
			}
			// instantiate specified node into the graph
			ERROR_CHECK_OBJECT_(stitch->overlayMediaConfig = vxCreateScalar(stitch->context, VX_TYPE_STRING_AMD, stitch->loomio_overlay.kernelArguments));
			ERROR_CHECK_OBJECT_(stitch->Img_overlay = vxCreateVirtualImage(stitch->graphStitch, stitch->overlay_buffer_width, stitch->overlay_buffer_height, VX_DF_IMAGE_RGBX));
			ERROR_CHECK_OBJECT_(stitch->loomioOverlayAuxData = vxCreateArray(stitch->context, VX_TYPE_UINT8, stitch->loomioOverlayAuxDataLength));
			vx_reference params[] = {
				(vx_reference)stitch->overlayMediaConfig,
				(vx_reference)stitch->Img_overlay,
				(vx_reference)stitch->loomioOverlayAuxData,
			};
			ERROR_CHECK_OBJECT_(stitch->nodeLoomIoOverlay = stitchCreateNode(stitch->graphStitch, stitch->loomio_overlay.kernelName, params, dimof(params)));
		}
		else {
			// need image created from OpenCL handle
			vx_imagepatch_addressing_t addr_overlay = { 0 };
			void *ptr_overlay[1] = { nullptr };
			addr_overlay.dim_x = stitch->overlay_buffer_width;
			addr_overlay.dim_y = (stitch->overlay_buffer_height);
			addr_overlay.stride_x = 4;
			addr_overlay.stride_y = stitch->overlay_buffer_stride_in_bytes;
			if (addr_overlay.stride_y == 0) addr_overlay.stride_y = addr_overlay.stride_x * addr_overlay.dim_x;
			ERROR_CHECK_OBJECT_(stitch->Img_overlay = vxCreateImageFromHandle(stitch->context, VX_DF_IMAGE_RGBX, &addr_overlay, ptr_overlay, VX_MEMORY_TYPE_OPENCL));
		}
		// create remap table object and image for overlay warp
		ERROR_CHECK_OBJECT_(stitch->overlay_remap = vxCreateRemap(stitch->context, stitch->overlay_buffer_width, stitch->overlay_buffer_height, stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height));
		ERROR_CHECK_OBJECT_(stitch->Img_overlay_rgb = vxCreateVirtualImage(stitch->graphStitch, stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height, VX_DF_IMAGE_RGB));
		ERROR_CHECK_OBJECT_(stitch->Img_overlay_rgba = vxCreateVirtualImage(stitch->graphStitch, stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height, VX_DF_IMAGE_RGBX));
		// initialize remap using lens model
		ERROR_CHECK_STATUS_(AllocateLensModelBuffersForOverlay(stitch));
		ERROR_CHECK_STATUS_(InitializeInternalTablesForRemap(stitch, stitch->overlay_remap,
			stitch->num_overlays, stitch->num_overlay_columns,
			stitch->overlay_buffer_width / stitch->num_overlay_columns,
			stitch->overlay_buffer_height / stitch->num_overlay_rows,
			stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height,
			&stitch->rig_par, stitch->overlay_par, stitch->overlaySrcMap, stitch->validPixelOverlayMap,
			stitch->overlayIndexTmpBuf, stitch->overlayIndexBuf));
		if (!stitch->feature_enable_reinitialize) {
			if (stitch->overlaySrcMap) { delete[] stitch->overlaySrcMap; stitch->overlaySrcMap = nullptr; }
			if (stitch->validPixelOverlayMap) { delete[] stitch->validPixelOverlayMap; stitch->validPixelOverlayMap = nullptr; }
			if (stitch->overlayIndexTmpBuf) { delete[] stitch->overlayIndexTmpBuf; stitch->overlayIndexTmpBuf = nullptr; }
			if (stitch->overlayIndexBuf) { delete[] stitch->overlayIndexBuf; stitch->overlayIndexBuf = nullptr; }
		}
	}
	if (strlen(stitch->loomio_output.kernelName) > 0) {
		// load OpenVX module (if specified)
		if (strlen(stitch->loomio_output.module) > 0) {
			vx_status status = vxLoadKernels(stitch->context, stitch->loomio_output.module);
			if (status != VX_SUCCESS) {
				ls_printf("ERROR: lsInitialize: vxLoadKernels(%s) failed (%d)\n", stitch->loomio_output.module, status);
				return VX_ERROR_INVALID_PARAMETERS;
			}
		}
		// instantiate specified node into the graph
		stitch->output_encode_tiles = 1;
		ERROR_CHECK_OBJECT_(stitch->outputMediaConfig = vxCreateScalar(stitch->context, VX_TYPE_STRING_AMD, stitch->loomio_output.kernelArguments));
		if ((stitch->output_buffer_format == VX_DF_IMAGE_NV12 || stitch->output_buffer_format == VX_DF_IMAGE_IYUV) &&
			(stitch->live_stitch_attr[LIVE_STITCH_ATTR_OUTPUT_TILE_NUM_X] != 1.0f ||
				stitch->live_stitch_attr[LIVE_STITCH_ATTR_OUTPUT_TILE_NUM_Y] != 1.0f)){
				// set buffer to encode tile width and height
				stitch->output_encode_tiles = (vx_uint32)(stitch->live_stitch_attr[LIVE_STITCH_ATTR_OUTPUT_TILE_NUM_X] * stitch->live_stitch_attr[LIVE_STITCH_ATTR_OUTPUT_TILE_NUM_Y]);
				stitch->output_encode_buffer_width = (vx_uint32)(stitch->live_stitch_attr[LIVE_STITCH_ATTR_OUTPUT_ENCODER_WIDTH]
					/ stitch->live_stitch_attr[LIVE_STITCH_ATTR_OUTPUT_TILE_NUM_X]);
				stitch->output_encode_buffer_height = (vx_uint32)(stitch->live_stitch_attr[LIVE_STITCH_ATTR_OUTPUT_ENCODER_HEIGHT]
					/ stitch->live_stitch_attr[LIVE_STITCH_ATTR_OUTPUT_TILE_NUM_Y] * stitch->output_encode_tiles);
				// create virtual image
				stitch->Img_output = vxCreateVirtualImage(stitch->graphStitch,
					stitch->output_encode_buffer_width,
					stitch->output_encode_buffer_height,
					stitch->output_buffer_format);
		}
		else{
			stitch->Img_output = vxCreateVirtualImage(stitch->graphStitch, stitch->output_buffer_width,	stitch->output_buffer_height, stitch->output_buffer_format);
		}
		ERROR_CHECK_OBJECT_(stitch->Img_output);
		vx_uint32 zero = 0;
		ERROR_CHECK_OBJECT_(stitch->loomioOutputAuxData = vxCreateArray(stitch->context, VX_TYPE_UINT8, stitch->loomioOutputAuxDataLength));
		vx_array loomioOutputAuxDataSource = nullptr;
		if (stitch->loomioCameraAuxData && stitch->loomioOutputAuxSelection != 1) {
			loomioOutputAuxDataSource = stitch->loomioCameraAuxData;
		}
		else if (stitch->loomioOverlayAuxData && stitch->loomioOutputAuxSelection != 2) {
			loomioOutputAuxDataSource = stitch->loomioOverlayAuxData;
		}
		vx_reference params[] = {
			(vx_reference)stitch->outputMediaConfig,
			(vx_reference)stitch->Img_output,
			(vx_reference)loomioOutputAuxDataSource,
			(vx_reference)stitch->loomioOutputAuxData,
		};
		ERROR_CHECK_OBJECT_(stitch->nodeLoomIoOutput = stitchCreateNode(stitch->graphStitch, stitch->loomio_output.kernelName, params, dimof(params)));
	}
	else {
		// need image created from OpenCL handle
		stitch->output_encode_tiles = 1;
		if (stitch->output_buffer_format == VX_DF_IMAGE_NV12 || stitch->output_buffer_format == VX_DF_IMAGE_IYUV)
		{
			vx_imagepatch_addressing_t addr_out[3] = { 0, 0, 0 };
			void *ptr[3] = { nullptr, nullptr, nullptr };
			// encode activated
			if (stitch->live_stitch_attr[LIVE_STITCH_ATTR_OUTPUT_TILE_NUM_X] != 1.0f ||
				stitch->live_stitch_attr[LIVE_STITCH_ATTR_OUTPUT_TILE_NUM_Y] != 1.0f){		
				// create the encoded tiled output
				stitch->output_encode_tiles = (vx_uint32)(stitch->live_stitch_attr[LIVE_STITCH_ATTR_OUTPUT_TILE_NUM_X] * stitch->live_stitch_attr[LIVE_STITCH_ATTR_OUTPUT_TILE_NUM_Y]);
				stitch->output_encode_buffer_width = (vx_uint32)(stitch->live_stitch_attr[LIVE_STITCH_ATTR_OUTPUT_ENCODER_WIDTH]
					/ stitch->live_stitch_attr[LIVE_STITCH_ATTR_OUTPUT_TILE_NUM_X]);
				stitch->output_encode_buffer_height = (vx_uint32)(stitch->live_stitch_attr[LIVE_STITCH_ATTR_OUTPUT_ENCODER_HEIGHT]
					/ stitch->live_stitch_attr[LIVE_STITCH_ATTR_OUTPUT_TILE_NUM_Y]) ;

				// create output buffer
				addr_out[0].dim_x = stitch->output_encode_buffer_width;	addr_out[0].dim_y = stitch->output_encode_buffer_height;
				addr_out[0].stride_x = 1; addr_out[0].stride_y = (vx_int32)stitch->live_stitch_attr[LIVE_STITCH_ATTR_OUTPUT_ENCODER_STRIDE_Y];	// assuing every tile has the same stride
				if (stitch->output_buffer_format == VX_DF_IMAGE_NV12){
					addr_out[1].dim_x = stitch->output_encode_buffer_width;	addr_out[1].dim_y = stitch->output_encode_buffer_height >> 1;
					addr_out[1].stride_x = 1; addr_out[1].stride_y = addr_out[0].stride_y;
				}
				else{
					addr_out[1].dim_x = stitch->output_encode_buffer_width;	addr_out[1].dim_y = stitch->output_encode_buffer_height;
					addr_out[1].stride_x = 1; addr_out[1].stride_y = addr_out[0].stride_y >> 1;
					addr_out[2].dim_x = stitch->output_encode_buffer_width;	addr_out[2].dim_y = stitch->output_encode_buffer_height;
					addr_out[2].stride_x = 1; addr_out[2].stride_y = addr_out[0].stride_y >> 1;
				}

				for (vx_uint32 i = 0; i < stitch->output_encode_tiles; i++){
					ERROR_CHECK_OBJECT_(stitch->encodetileOutput[i] = vxCreateImageFromHandle(stitch->context, stitch->output_buffer_format, &addr_out[0], ptr, VX_MEMORY_TYPE_OPENCL));
				}
			}
			else{
				// create output buffer
				addr_out[0].dim_x = stitch->output_rgb_buffer_width;	addr_out[0].dim_y = stitch->output_rgb_buffer_height;
				addr_out[0].stride_x = 1; addr_out[0].stride_y = stitch->output_buffer_stride_in_bytes;
				if (stitch->output_buffer_format == VX_DF_IMAGE_NV12){
					addr_out[1].dim_x = stitch->output_rgb_buffer_width;	addr_out[1].dim_y = stitch->output_rgb_buffer_height >> 1;
					addr_out[1].stride_x = 1; addr_out[1].stride_y = stitch->output_buffer_stride_in_bytes;
				}
				else{
					addr_out[1].dim_x = stitch->output_rgb_buffer_width;	addr_out[1].dim_y = stitch->output_rgb_buffer_height;
					addr_out[1].stride_x = 1; addr_out[1].stride_y = stitch->output_buffer_stride_in_bytes;
					addr_out[2].dim_x = stitch->output_rgb_buffer_width;	addr_out[2].dim_y = stitch->output_rgb_buffer_height;
					addr_out[2].stride_x = 1; addr_out[2].stride_y = stitch->output_buffer_stride_in_bytes;
				}
				ERROR_CHECK_OBJECT_(stitch->Img_output = vxCreateImageFromHandle(stitch->context, stitch->output_buffer_format, &addr_out[0], ptr, VX_MEMORY_TYPE_OPENCL));
			}			
		}
		else{
			// create RGB/YUV buffer
			vx_imagepatch_addressing_t addr_out = { 0 };
			void *ptr[1] = { nullptr };
			addr_out.dim_x = stitch->output_buffer_width;
			addr_out.dim_y = stitch->output_buffer_height;
			addr_out.stride_x = (stitch->output_buffer_format == VX_DF_IMAGE_RGB) ? 3 : 2;
			addr_out.stride_y = stitch->output_buffer_stride_in_bytes;
			if (addr_out.stride_y == 0) addr_out.stride_y = addr_out.stride_x * addr_out.dim_x;
			ERROR_CHECK_OBJECT_(stitch->Img_output = vxCreateImageFromHandle(stitch->context, stitch->output_buffer_format, &addr_out, ptr, VX_MEMORY_TYPE_OPENCL));
		}
	}
	if (stitch->output_encode_tiles > 4){ ls_printf("ERROR: lsInitialize: Max Encode Tiles supported is 4\n"); return VX_ERROR_INVALID_PARAMETERS;}
	// create temporary images when extra color conversion is needed
	if (stitch->camera_buffer_format != VX_DF_IMAGE_RGB) {
		ERROR_CHECK_OBJECT_(stitch->Img_input_rgb = vxCreateVirtualImage(stitch->graphStitch, stitch->camera_rgb_buffer_width, stitch->camera_rgb_buffer_height, VX_DF_IMAGE_RGB));
	}
	if (stitch->output_buffer_format != VX_DF_IMAGE_RGB) {
		vx_uint32 output_img_width =  stitch->output_buffer_width;
		vx_uint32 output_img_height = stitch->output_buffer_height;
		ERROR_CHECK_OBJECT_(stitch->Img_output_rgb = vxCreateImage(stitch->context, output_img_width, output_img_height, VX_DF_IMAGE_RGB));
	}
	// process chroma key
	stitch->CHROMA_KEY = (vx_uint32)stitch->live_stitch_attr[LIVE_STITCH_ATTR_CHROMA_KEY];
	if (stitch->CHROMA_KEY){
		// create chroma key RGB buffer
		vx_imagepatch_addressing_t addr_out = { 0 };
		void *ptr[1] = { nullptr };
		addr_out.dim_x = stitch->output_buffer_width;
		addr_out.dim_y = stitch->output_buffer_height;
		addr_out.stride_x = 3;
		addr_out.stride_y = stitch->output_buffer_width * 3;
		if (addr_out.stride_y == 0) addr_out.stride_y = addr_out.stride_x * addr_out.dim_x;
		ERROR_CHECK_OBJECT_(stitch->chroma_key_input_img = vxCreateImageFromHandle(stitch->context, VX_DF_IMAGE_RGB, &addr_out, ptr, VX_MEMORY_TYPE_OPENCL));
		// create chroma key mask U8 buffer
		vx_uint32 output_img_width = stitch->output_buffer_width;
		vx_uint32 output_img_height = stitch->output_buffer_height;
		ERROR_CHECK_OBJECT_(stitch->chroma_key_mask_img = vxCreateVirtualImage(stitch->graphStitch, output_img_width, output_img_height, VX_DF_IMAGE_U8));
		ERROR_CHECK_OBJECT_(stitch->chroma_key_input_RGB_img = vxCreateVirtualImage(stitch->graphStitch, output_img_width, output_img_height, VX_DF_IMAGE_RGB));
		stitch->CHROMA_KEY_EED = (vx_uint32)stitch->live_stitch_attr[LIVE_STITCH_ATTR_CHROMA_KEY_EED];
		if (stitch->CHROMA_KEY_EED){
			ERROR_CHECK_OBJECT_(stitch->chroma_key_dilate_mask_img = vxCreateVirtualImage(stitch->graphStitch, output_img_width, output_img_height, VX_DF_IMAGE_U8));
			ERROR_CHECK_OBJECT_(stitch->chroma_key_erode_mask_img = vxCreateVirtualImage(stitch->graphStitch, output_img_width, output_img_height, VX_DF_IMAGE_U8));
		}
	}
	////////////////////////////////////////////////////////////////////////
	// build the input and output processing parts of stitch graph
	stitch->rgb_input = stitch->Img_input;
	if (stitch->camera_buffer_format != VX_DF_IMAGE_RGB) {
		// needs input color conversion
		if (stitch->camera_buffer_format == VX_DF_IMAGE_NV12 || stitch->camera_buffer_format == VX_DF_IMAGE_IYUV){
			stitch->InputColorConvertNode = vxColorConvertNode(stitch->graphStitch, stitch->rgb_input, stitch->Img_input_rgb);
		}
		else{
			stitch->InputColorConvertNode = stitchColorConvertNode(stitch->graphStitch, stitch->rgb_input, stitch->Img_input_rgb);
		}
		ERROR_CHECK_OBJECT_(stitch->InputColorConvertNode);
		stitch->rgb_input = stitch->Img_input_rgb;
	}
	// temporal filter for camera noise correction
	stitch->NOISE_FILTER = (vx_uint32)stitch->live_stitch_attr[LIVE_STITCH_ATTR_NOISE_FILTER];
	if (stitch->NOISE_FILTER){
		stitch->noiseFilterLambda = (vx_float32)stitch->live_stitch_attr[LIVE_STITCH_ATTR_NOISE_FILTER_LAMBDA];
		ERROR_CHECK_OBJECT_(stitch->filterLambda = vxCreateScalar(stitch->context, VX_TYPE_FLOAT32, &stitch->noiseFilterLambda));
		ERROR_CHECK_OBJECT_(stitch->noiseFilterInput_image = vxCreateImage(stitch->context, stitch->camera_buffer_width, stitch->camera_buffer_height, VX_DF_IMAGE_RGB));
		ERROR_CHECK_OBJECT_(stitch->noiseFilterImageDelay = vxCreateDelay(stitch->context, (vx_reference)stitch->noiseFilterInput_image, 2));
		stitch->noiseFilterNode = stitchNoiseFilterNode(stitch->graphStitch, stitch->filterLambda, stitch->rgb_input, (vx_image)vxGetReferenceFromDelay(stitch->noiseFilterImageDelay, -1), (vx_image)vxGetReferenceFromDelay(stitch->noiseFilterImageDelay, 0));
		ERROR_CHECK_OBJECT_(stitch->noiseFilterNode);
		stitch->rgb_input = (vx_image)vxGetReferenceFromDelay(stitch->noiseFilterImageDelay, 0);
	}
	stitch->rgb_output = stitch->Img_output;
	if (stitch->output_buffer_format != VX_DF_IMAGE_RGB) {
		// needs output color conversion
		if (stitch->output_buffer_format == VX_DF_IMAGE_NV12 || stitch->output_buffer_format == VX_DF_IMAGE_IYUV){
			if (stitch->output_encode_tiles == 1){ 
				stitch->OutputColorConvertNode = vxColorConvertNode(stitch->graphStitch, stitch->Img_output_rgb, stitch->rgb_output); 
				ERROR_CHECK_OBJECT_(stitch->OutputColorConvertNode);
			}
			else{
				// output needs to be encoded
				ERROR_CHECK_STATUS_(EncodeProcessImage(stitch));
				ERROR_CHECK_STATUS_(EncodeCreateImageFromROI(stitch));
			}
		}
		else{
			stitch->OutputColorConvertNode = stitchColorConvertNode(stitch->graphStitch, stitch->Img_output_rgb, stitch->rgb_output);
			ERROR_CHECK_OBJECT_(stitch->OutputColorConvertNode);
		}
		stitch->rgb_output = stitch->Img_output_rgb;
	}
	if (stitch->CHROMA_KEY){
		vx_uint32 ChromaKey_value = (vx_uint32)stitch->live_stitch_attr[LIVE_STITCH_ATTR_CHROMA_KEY_VALUE];
		vx_uint32 ChromaKey_Tol = (vx_uint32)stitch->live_stitch_attr[LIVE_STITCH_ATTR_CHROMA_KEY_TOL];
		stitch->chromaKey_mask_generation_node = stitchChromaKeyMaskGeneratorNode(stitch->graphStitch, ChromaKey_value, ChromaKey_Tol, stitch->chroma_key_input_RGB_img, stitch->chroma_key_mask_img);
		ERROR_CHECK_OBJECT_(stitch->chromaKey_mask_generation_node);
		if (stitch->CHROMA_KEY_EED){
			stitch->chromaKey_erode_node = vxErode3x3Node(stitch->graphStitch, stitch->chroma_key_mask_img, stitch->chroma_key_erode_mask_img);
			ERROR_CHECK_OBJECT_(stitch->chromaKey_erode_node);
			stitch->chromaKey_dilate_node = vxDilate3x3Node(stitch->graphStitch, stitch->chroma_key_erode_mask_img, stitch->chroma_key_dilate_mask_img);
			ERROR_CHECK_OBJECT_(stitch->chromaKey_dilate_node);
			stitch->chromaKey_merge_node = stitchChromaKeyMergeNode(stitch->graphStitch, stitch->chroma_key_input_RGB_img, stitch->chroma_key_input_img, stitch->chroma_key_dilate_mask_img, stitch->rgb_output);
			ERROR_CHECK_OBJECT_(stitch->chromaKey_merge_node);
		}
		else{
			stitch->chromaKey_merge_node = stitchChromaKeyMergeNode(stitch->graphStitch, stitch->chroma_key_input_RGB_img, stitch->chroma_key_input_img, stitch->chroma_key_mask_img, stitch->rgb_output);
			ERROR_CHECK_OBJECT_(stitch->chromaKey_merge_node);
		}
		stitch->rgb_output = stitch->chroma_key_input_RGB_img;
	}
	if (stitch->Img_overlay) {
		// need add overlay
		ERROR_CHECK_OBJECT_(stitch->nodeOverlayRemap = vxRemapNode(stitch->graphStitch, stitch->Img_overlay, stitch->overlay_remap, VX_INTERPOLATION_TYPE_BILINEAR, stitch->Img_overlay_rgba));
		ERROR_CHECK_OBJECT_(stitch->nodeOverlayBlend = stitchAlphaBlendNode(stitch->graphStitch, stitch->Img_overlay_rgb, stitch->Img_overlay_rgba, stitch->rgb_output));
		stitch->rgb_output = stitch->Img_overlay_rgb;
	}
	if (strlen(stitch->loomio_viewing.kernelName) > 0) {
		// load OpenVX module (if specified)
		if (strlen(stitch->loomio_viewing.module) > 0) {
			vx_status status = vxLoadKernels(stitch->context, stitch->loomio_viewing.module);
			if (status != VX_SUCCESS) {
				ls_printf("ERROR: lsInitialize: vxLoadKernels(%s) failed (%d)\n", stitch->loomio_viewing.module, status);
				return VX_ERROR_INVALID_PARAMETERS;
			}
		}
		// instantiate specified node into the graph
		vx_uint32 zero = 0;
		ERROR_CHECK_OBJECT_(stitch->viewingMediaConfig = vxCreateScalar(stitch->context, VX_TYPE_STRING_AMD, stitch->loomio_viewing.kernelArguments));
		ERROR_CHECK_OBJECT_(stitch->loomioViewingAuxData = vxCreateArray(stitch->context, VX_TYPE_UINT8, stitch->loomioOutputAuxDataLength));
		vx_reference params[] = {
			(vx_reference)stitch->viewingMediaConfig,
			(vx_reference)stitch->rgb_output,
			(vx_reference)stitch->loomioCameraAuxData,
			(vx_reference)stitch->loomioViewingAuxData,
		};
		ERROR_CHECK_OBJECT_(stitch->nodeLoomIoViewing = stitchCreateNode(stitch->graphStitch, stitch->loomio_viewing.kernelName, params, dimof(params)));
	}

	/***********************************************************************************************************************************
	Quick Stitch Mode -> Simple stitch
	************************************************************************************************************************************/
	if (stitch->stitching_mode == stitching_mode_quick_and_dirty)
	{
		// create remap table object
		ERROR_CHECK_OBJECT_(stitch->camera_remap = vxCreateRemap(stitch->context, stitch->camera_rgb_buffer_width, stitch->camera_rgb_buffer_height, stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height));
		// initialize remap using lens model
		ERROR_CHECK_STATUS_(AllocateLensModelBuffersForCamera(stitch));
		ERROR_CHECK_STATUS_(InitializeInternalTablesForRemap(stitch, stitch->camera_remap,
			stitch->num_cameras, stitch->num_camera_columns,
			stitch->camera_buffer_width / stitch->num_camera_columns,
			stitch->camera_buffer_height / stitch->num_camera_rows,
			stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height,
			&stitch->rig_par, stitch->camera_par, stitch->camSrcMap, stitch->validPixelCamMap,
			stitch->camIndexTmpBuf, stitch->camIndexBuf));
		if (!stitch->feature_enable_reinitialize) {
			if (stitch->camSrcMap) { delete[] stitch->camSrcMap; stitch->camSrcMap = nullptr; }
			if (stitch->validPixelCamMap) { delete[] stitch->validPixelCamMap; stitch->validPixelCamMap = nullptr; }
			if (stitch->camIndexTmpBuf) { delete[] stitch->camIndexTmpBuf; stitch->camIndexTmpBuf = nullptr; }
			if (stitch->camIndexBuf) { delete[] stitch->camIndexBuf; stitch->camIndexBuf = nullptr; }
		}

		////////////////////////////////////////////////////////////////////////
		// create and verify graphStitch using simple remap kernel
		////////////////////////////////////////////////////////////////////////
		ERROR_CHECK_OBJECT_(stitch->SimpleStitchRemapNode = vxRemapNode(stitch->graphStitch, stitch->rgb_input, stitch->camera_remap, VX_INTERPOLATION_TYPE_BILINEAR, stitch->rgb_output));
		ERROR_CHECK_STATUS_(vxVerifyGraph(stitch->graphStitch));
		ERROR_CHECK_STATUS_(SyncInternalTables(stitch));
	}
	/***********************************************************************************************************************************
	Normal Stitch Mode -> Full Stitch
	************************************************************************************************************************************/
	else if (stitch->stitching_mode == stitching_mode_normal)
	{
		////////////////////////////////////////////////////////////////////////
		// get configuration from environment variables
		////////////////////////////////////////////////////////////////////////
		if (stitch->num_cameras > 1) {
			stitch->EXPO_COMP = (vx_uint32)stitch->live_stitch_attr[LIVE_STITCH_ATTR_EXPCOMP];
			stitch->SEAM_FIND = (vx_uint32)stitch->live_stitch_attr[LIVE_STITCH_ATTR_SEAMFIND];
			stitch->SEAM_REFRESH = (vx_uint32)stitch->live_stitch_attr[LIVE_STITCH_ATTR_SEAM_REFRESH];
			stitch->SEAM_COST_SELECT = (vx_uint32)stitch->live_stitch_attr[LIVE_STITCH_ATTR_SEAM_COST_SELECT];
			stitch->SEAM_FLAGS = (vx_uint32)stitch->live_stitch_attr[LIVE_STITCH_ATTR_SEAM_FLAGS];
			stitch->scene_threshold_value = (vx_uint32)stitch->live_stitch_attr[LIVE_STITCH_ATTR_SEAM_THRESHOLD];
			stitch->SEAM_FIND_TARGET = 0;
			stitch->MULTIBAND_BLEND = (vx_uint32)stitch->live_stitch_attr[LIVE_STITCH_ATTR_MULTIBAND];
			stitch->num_bands = (vx_uint32)stitch->live_stitch_attr[LIVE_STITCH_ATTR_MULTIBAND_NUMBANDS];
			if (stitch->num_bands < 2) {
				// general protection
				stitch->live_stitch_attr[LIVE_STITCH_ATTR_MULTIBAND] = 0.0f;
				stitch->live_stitch_attr[LIVE_STITCH_ATTR_MULTIBAND_NUMBANDS] = 0.0f;
				stitch->MULTIBAND_BLEND = 0;
				stitch->num_bands = 0;
			}
			if (stitch->MULTIBAND_BLEND){
				// general protection for odd sized equirectangle
				vx_uint32 MAX_BAND = 10; // Max bands for multiband blend
				vx_uint32 MAX_BAND_ALLOWED = 0, half_width = stitch->output_rgb_buffer_width;
				for (vx_uint32 i = 0; i < MAX_BAND; i++){
					if (half_width % 2 == 0){ half_width = half_width / 2; MAX_BAND_ALLOWED++; }
					else{ break; }
				}
				if ((vx_uint32)stitch->num_bands > MAX_BAND_ALLOWED){
					stitch->num_bands = (vx_int32)MAX_BAND_ALLOWED;
					ls_printf("WARNING: Max allowed MULTIBAND BLEND bands for the set output equirectangle is %d\n", MAX_BAND_ALLOWED);
				}
			}
			stitch->EXPO_COMP_GAINW = 1;
			stitch->EXPO_COMP_GAINH = 1;
			stitch->EXPO_COMP_GAINC = (stitch->EXPO_COMP == 2) ? 3 : (stitch->EXPO_COMP == 4)? 12: 1;
			if (stitch->EXPO_COMP >= 3 ) {
				stitch->EXPO_COMP_GAINW = (vx_uint32)std::max(1.0f, stitch->live_stitch_attr[LIVE_STITCH_ATTR_EXPCOMP_GAIN_IMG_W]);
				stitch->EXPO_COMP_GAINH = (vx_uint32)std::max(1.0f, stitch->live_stitch_attr[LIVE_STITCH_ATTR_EXPCOMP_GAIN_IMG_H]);
				stitch->EXPO_COMP_GAINC = (vx_uint32)std::max(1.0f, stitch->live_stitch_attr[LIVE_STITCH_ATTR_EXPCOMP_GAIN_IMG_C]);
				if (stitch->EXPO_COMP == 4) stitch->EXPO_COMP_GAINC = 12;	// override the default value since kernel expects 12.
			}
			if (stitch->EXPO_COMP < 3) {
				stitch->alpha = (vx_float32)stitch->live_stitch_attr[LIVE_STITCH_ATTR_EXPCOMP_ALPHA_VALUE];
				stitch->beta = (vx_float32)stitch->live_stitch_attr[LIVE_STITCH_ATTR_EXPCOMP_BETA_VALUE];
			}
			// option to disable seam find with environment variable
			char value[256] = { 0 };
			if (StitchGetEnvironmentVariable("LOOM_SEAM_FIND_DISABLE", value, sizeof(value))) {
				if (!strcmp(value, "1") && stitch->SEAM_FIND) {
					stitch->SEAM_FIND = 0;
					ls_printf("WARNING: SeamFind has been disabled using environment variable: LOOM_SEAM_FIND_DISABLE=1\n");
				}
			}
			// quick setup files load
			stitch->SETUP_LOAD = (vx_uint32)stitch->live_stitch_attr[LIVE_STITCH_ATTR_SAVE_AND_LOAD_INIT];
			stitch->SETUP_LOAD_FILES_FOUND = vx_false_e;
			if (stitch->MULTIBAND_BLEND || stitch->EXPO_COMP){ stitch->SETUP_LOAD = 0; }
			if (stitch->SETUP_LOAD){ 	
				vx_status status = quickSetupFilesLookup(stitch);
				if (status != VX_SUCCESS) {
					vxAddLogEntry((vx_reference)stitch->context, status, "ERROR: lsInitialize: quickSetupFilesLookup() failed (%d)\n", status);
					return status;
				}
			}
		}

		// allocate internal tables
		vx_status status = AllocateInternalTablesForCamera(stitch);
		if (status != VX_SUCCESS)
			return status;

		////////////////////////////////////////////////////////////////////////
		// create and verify graphStitch using low-level kernels
		////////////////////////////////////////////////////////////////////////
		// warping
		ERROR_CHECK_OBJECT_(stitch->WarpNode = stitchWarpNode(stitch->graphStitch, 1, stitch->num_cameras, stitch->ValidPixelEntry, stitch->WarpRemapEntry, stitch->rgb_input, stitch->RGBY1, stitch->warp_luma_image, stitch->num_camera_columns));

		// exposure comp
		vx_image merge_input = stitch->RGBY1;
		vx_image merge_weight = stitch->weight_image;
		if (stitch->EXPO_COMP) {
			if (stitch->EXPO_COMP == 1) {
				ERROR_CHECK_OBJECT_(stitch->ExpcompComputeGainNode = stitchExposureCompCalcErrorFnNode(stitch->graphStitch, stitch->num_cameras, stitch->RGBY1, stitch->OverlapPixelEntry, stitch->valid_mask_image, stitch->A_matrix));
				ERROR_CHECK_OBJECT_(stitch->ExpcompSolveGainNode = stitchExposureCompSolveForGainNode(stitch->graphStitch, stitch->alpha, stitch->beta, stitch->A_matrix, stitch->overlap_matrix, stitch->gain_array));
			}
			else if (stitch->EXPO_COMP == 2) {
				ERROR_CHECK_OBJECT_(stitch->ExpcompComputeGainNode = stitchExposureCompCalcErrorFnRGBNode(stitch->graphStitch, stitch->num_cameras, stitch->RGBY1, stitch->OverlapPixelEntry, stitch->valid_mask_image, stitch->A_matrix));
				ERROR_CHECK_OBJECT_(stitch->ExpcompSolveGainNode = stitchExposureCompSolveForGainNode(stitch->graphStitch, stitch->alpha, stitch->beta, stitch->A_matrix, stitch->overlap_matrix, stitch->gain_array));
			}
			ERROR_CHECK_OBJECT_(stitch->ExpcompApplyGainNode = stitchExposureCompApplyGainNode(stitch->graphStitch, stitch->RGBY1, stitch->gain_array, stitch->valid_array, stitch->num_cameras, stitch->EXPO_COMP_GAINW, stitch->EXPO_COMP_GAINH, stitch->RGBY2));
			// update merge input
			merge_input = stitch->RGBY2;
		}
		if (stitch->SEAM_FIND) {
			if (stitch->SEAM_REFRESH)
			{
				//SeamFind Step 1: Seam Refresh 
				stitch->SeamfindStep1Node = stitchSeamFindSceneDetectNode(stitch->graphStitch, stitch->current_frame, stitch->scene_threshold,
					stitch->warp_luma_image, stitch->seamfind_info_array, stitch->seamfind_pref_array, stitch->seamfind_scene_array);
				ERROR_CHECK_OBJECT_(stitch->SeamfindStep1Node);
			}
			//SeamFind Step 2 - Cost Generation: 0:OpenVX Sobel 1:Optimized Sobel
			if (!stitch->SEAM_COST_SELECT) {
				vx_int32 zero = 0; vx_scalar shift;
				ERROR_CHECK_OBJECT_(shift = vxCreateScalar(stitch->context, VX_TYPE_INT32, &zero));
				ERROR_CHECK_OBJECT_(stitch->SobelNode = vxSobel3x3Node(stitch->graphStitch, stitch->warp_luma_image, stitch->sobelx_image, stitch->sobely_image));
				ERROR_CHECK_OBJECT_(stitch->MagnitudeNode = vxMagnitudeNode(stitch->graphStitch, stitch->sobelx_image, stitch->sobely_image, stitch->sobel_magnitude_s16_image));
				ERROR_CHECK_OBJECT_(stitch->PhaseNode = vxPhaseNode(stitch->graphStitch, stitch->sobelx_image, stitch->sobely_image, stitch->sobel_phase_image));
				ERROR_CHECK_OBJECT_(stitch->ConvertDepthNode = vxConvertDepthNode(stitch->graphStitch, stitch->sobel_magnitude_s16_image, stitch->sobel_magnitude_image, VX_CONVERT_POLICY_SATURATE, shift));
				ERROR_CHECK_STATUS_(vxReleaseScalar(&shift));
			}
			else {
				ERROR_CHECK_OBJECT_(stitch->SeamfindAnalyzeNode = stitchSeamFindAnalyzeNode(stitch->graphStitch, stitch->current_frame, stitch->seamfind_pref_array, stitch->seam_cost_enable));
				ERROR_CHECK_OBJECT_(stitch->SeamfindStep2Node = stitchSeamFindCostGenerateNode(stitch->graphStitch, stitch->seam_cost_enable, stitch->warp_luma_image, stitch->sobel_magnitude_image, stitch->sobel_phase_image));
			}
			//SeamFind Step 3 - Cost Accumulate
			stitch->SeamfindStep3Node = stitchSeamFindCostAccumulateNode(stitch->graphStitch, stitch->current_frame, stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height,
				stitch->sobel_magnitude_image, stitch->sobel_phase_image, stitch->valid_mask_image, stitch->seamfind_valid_array, stitch->seamfind_pref_array,
				stitch->seamfind_info_array, stitch->seamfind_accum_array);
			ERROR_CHECK_OBJECT_(stitch->SeamfindStep3Node);
			//SeamFind Step 4 - Path Trace
			stitch->SeamfindStep4Node = stitchSeamFindPathTraceNode(stitch->graphStitch, stitch->current_frame, stitch->weight_image, stitch->seamfind_info_array, 
				stitch->seamfind_accum_array, stitch->seamfind_pref_array, stitch->seamfind_path_array);
			ERROR_CHECK_OBJECT_(stitch->SeamfindStep4Node);
			//SeamFind Step 5 - Set Weights
			stitch->SeamfindStep5Node = stitchSeamFindSetWeightsNode(stitch->graphStitch, stitch->current_frame, stitch->num_cameras, stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height,
				stitch->seamfind_weight_array, stitch->seamfind_path_array, stitch->seamfind_pref_array, stitch->seamfind_weight_image, stitch->SEAM_FLAGS);
			ERROR_CHECK_OBJECT_(stitch->SeamfindStep5Node);
			// update merge weight image
			merge_weight = stitch->seamfind_weight_image;
		}
		// create data objects and nodes for multiband blending
		if (stitch->MULTIBAND_BLEND){
			// create Laplacian pyramids.
			for (int i = 1; i < stitch->num_bands; i++) {
				stitch->pStitchMultiband[i].WeightHSGNode = stitchMultiBandHalfScaleGaussianNode(stitch->graphStitch, stitch->num_cameras, stitch->pStitchMultiband[i].valid_array_offset,
					stitch->blend_offsets, stitch->pStitchMultiband[i - 1].WeightPyrImgGaussian, stitch->pStitchMultiband[i].WeightPyrImgGaussian);
				ERROR_CHECK_OBJECT_(stitch->pStitchMultiband[i].WeightHSGNode);
				stitch->pStitchMultiband[i].SourceHSGNode = stitchMultiBandHalfScaleGaussianNode(stitch->graphStitch, stitch->num_cameras, stitch->pStitchMultiband[i].valid_array_offset,
					stitch->blend_offsets, stitch->pStitchMultiband[i - 1].DstPyrImgGaussian, stitch->pStitchMultiband[i].DstPyrImgGaussian);
				ERROR_CHECK_OBJECT_(stitch->pStitchMultiband[i].SourceHSGNode);
				stitch->pStitchMultiband[i - 1].UpscaleSubtractNode = stitchMultiBandUpscaleGaussianSubtractNode(stitch->graphStitch, stitch->num_cameras, stitch->pStitchMultiband[i - 1].valid_array_offset,
					stitch->pStitchMultiband[i - 1].DstPyrImgGaussian, stitch->pStitchMultiband[i].DstPyrImgGaussian, stitch->blend_offsets, stitch->pStitchMultiband[i-1].WeightPyrImgGaussian, stitch->pStitchMultiband[i - 1].DstPyrImgLaplacian);
				ERROR_CHECK_OBJECT_(stitch->pStitchMultiband[i - 1].UpscaleSubtractNode);
			}
			// reconstruct Laplacian after blending with corresponding weights: for band = num_bands-1, laplacian and gaussian is the same
			int i = stitch->num_bands - 1;
			stitch->pStitchMultiband[i].BlendNode = stitchMultiBandMergeNode(stitch->graphStitch, stitch->num_cameras, stitch->pStitchMultiband[i].valid_array_offset,
				stitch->pStitchMultiband[i].DstPyrImgGaussian, stitch->pStitchMultiband[i].WeightPyrImgGaussian, stitch->blend_offsets, stitch->pStitchMultiband[i].DstPyrImgLaplacianRec);
			ERROR_CHECK_OBJECT_(stitch->pStitchMultiband[i].BlendNode);
			--i;
			for (; i > 0; --i){
				stitch->pStitchMultiband[i].UpscaleAddNode = stitchMultiBandUpscaleGaussianAddNode(stitch->graphStitch, stitch->num_cameras, stitch->pStitchMultiband[i].valid_array_offset,
					stitch->pStitchMultiband[i].DstPyrImgLaplacian, stitch->pStitchMultiband[i + 1].DstPyrImgLaplacianRec, stitch->blend_offsets, stitch->pStitchMultiband[i].DstPyrImgLaplacianRec);
				ERROR_CHECK_OBJECT_(stitch->pStitchMultiband[i].UpscaleAddNode);
			}
			// for the lowest level
			stitch->pStitchMultiband[0].UpscaleAddNode = stitchMultiBandLaplacianReconstructNode(stitch->graphStitch, stitch->num_cameras, stitch->pStitchMultiband[0].valid_array_offset,
				stitch->pStitchMultiband[0].DstPyrImgLaplacian, stitch->pStitchMultiband[1].DstPyrImgLaplacianRec, stitch->blend_offsets, stitch->pStitchMultiband[0].DstPyrImgLaplacianRec);
			ERROR_CHECK_OBJECT_(stitch->pStitchMultiband[0].UpscaleAddNode);
			// update merge input and weight images
			merge_input = stitch->pStitchMultiband[0].DstPyrImgLaplacianRec;
			merge_weight = stitch->blend_mask_image;
		}
		// merge node
		ERROR_CHECK_OBJECT_(stitch->MergeNode = stitchMergeNode(stitch->graphStitch,
			stitch->cam_id_image, stitch->group1_image, stitch->group2_image, merge_input, merge_weight, stitch->rgb_output));

		// verify the graph
		ERROR_CHECK_STATUS_(vxVerifyGraph(stitch->graphStitch));
		ERROR_CHECK_STATUS_(SyncInternalTables(stitch));
	}
	/***********************************************************************************************************************************
	Other Modes
	************************************************************************************************************************************/
	else {
		vxAddLogEntry((vx_reference)stitch->context, VX_ERROR_NO_RESOURCES, "Other Stitching Modes are under development\nMode-1 = Quick Stitch Mode Available\nMode-2 = Normal Stitch Mode Available\n");
		return VX_ERROR_NO_RESOURCES;
	}

	// mark that initialization is successful
	stitch->initialized = true;

	// debug: dump auxiliary data
	if (stitch->loomioCameraAuxData || stitch->loomioOverlayAuxData || stitch->loomioOutputAuxData || stitch->loomioViewingAuxData) {
		char fileName[1024] = { 0 };
		if (StitchGetEnvironmentVariable("LOOMIO_AUX_DUMP", fileName, sizeof(fileName))) {
			stitch->loomioAuxDumpFile = fopen(fileName, "wb");
			if (!stitch->loomioAuxDumpFile) { 
				ls_printf("ERROR: unable to create: %s\n", fileName); 
				if (stitch->loomioAuxDumpFile != NULL)	fclose(stitch->loomioAuxDumpFile);
				return VX_FAILURE;
			}
			ls_printf("OK: dumping auxiliary data into %s\n", fileName);
		}
	}
	PROFILER_STOP(LoomSL, InitializeGraph);
	return VX_SUCCESS;
}

//! \brief initialize the stitch context.
LIVE_STITCH_API_ENTRY vx_status VX_API_CALL lsReinitialize(ls_context stitch)
{
	PROFILER_START(LoomSL, ReinitializeGraph);
	if (!stitch->reinitialize_required) return VX_SUCCESS;
	ERROR_CHECK_STATUS_(IsValidContextAndInitialized(stitch));
	if (!stitch->feature_enable_reinitialize) {
		ls_printf("ERROR: lsReinitialize has been disabled\n");
		return VX_ERROR_NOT_SUPPORTED;
	}
	if (stitch->scheduled) {
		ls_printf("ERROR: lsReinitialize: can't reinitialize when already scheduled\n");
		return VX_ERROR_GRAPH_SCHEDULED;
	}

	if (stitch->rig_params_updated || stitch->camera_params_updated) {

		// Quick Initailize enabled
		if (stitch->stitchInitData && stitch->stitchInitData->graphInitialize){
			ERROR_CHECK_STATUS_(setupQuickInitializeParams(stitch)); 
		}

		// re-initialize tables for camera
		if (stitch->camera_remap){
			ERROR_CHECK_STATUS_(InitializeInternalTablesForRemap(stitch, stitch->camera_remap,
				stitch->num_cameras, stitch->num_camera_columns,
				stitch->camera_buffer_width / stitch->num_camera_columns,
				stitch->camera_buffer_height / stitch->num_camera_rows,
				stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height,
				&stitch->rig_par, stitch->camera_par, stitch->camSrcMap, stitch->validPixelCamMap,
				stitch->camIndexTmpBuf, stitch->camIndexBuf));
		}
		else{
			ERROR_CHECK_STATUS_(InitializeInternalTablesForCamera(stitch));
		}
		ERROR_CHECK_STATUS_(SyncInternalTables(stitch));
	}
	if (stitch->rig_params_updated || stitch->overlay_params_updated) {
		// re-initialize tables for overlay
		if (stitch->overlay_remap) {
			ERROR_CHECK_STATUS_(InitializeInternalTablesForRemap(stitch, stitch->overlay_remap,
				stitch->num_overlays, stitch->num_overlay_columns,
				stitch->overlay_buffer_width / stitch->num_overlay_columns,
				stitch->overlay_buffer_height / stitch->num_overlay_rows,
				stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height,
				&stitch->rig_par, stitch->overlay_par, stitch->overlaySrcMap, stitch->validPixelOverlayMap,
				stitch->overlayIndexTmpBuf, stitch->overlayIndexBuf));
			ERROR_CHECK_STATUS_(SyncInternalTables(stitch));
		}	
	}

	// clear flags
	stitch->reinitialize_required = false;
	stitch->rig_params_updated = false;
	stitch->camera_params_updated = false;
	stitch->overlay_params_updated = false;
	PROFILER_STOP(LoomSL, ReinitializeGraph);
	return VX_SUCCESS;
}

//! \brief Release stitch context. The ls_context will be reset to NULL.
SHARED_PUBLIC vx_status VX_API_CALL lsReleaseContext(ls_context * pStitch)
{
	if (!pStitch) {
		return VX_ERROR_INVALID_REFERENCE;
	}
	else {
		ls_context stitch = *pStitch;
		ERROR_CHECK_STATUS_(IsValidContext(stitch));
		// graph profile dump if requested
		if (stitch->live_stitch_attr[LIVE_STITCH_ATTR_PROFILER]) {
			if (stitch->graphStitch) {
				ls_printf("> stitch graph profile\n"); char fileName[] = "stdout";
				ERROR_CHECK_STATUS_(vxQueryGraph(stitch->graphStitch, VX_GRAPH_ATTRIBUTE_AMD_PERFORMANCE_INTERNAL_PROFILE, fileName, 0));
			}
		}

		// release configurations
		if (stitch->camera_par) delete[] stitch->camera_par;
		if (stitch->overlay_par) delete[] stitch->overlay_par;

		// release image objects
		if (stitch->Img_input) ERROR_CHECK_STATUS_(vxReleaseImage(&stitch->Img_input));
		if (stitch->Img_output) ERROR_CHECK_STATUS_(vxReleaseImage(&stitch->Img_output));
		if (stitch->Img_input_rgb) ERROR_CHECK_STATUS_(vxReleaseImage(&stitch->Img_input_rgb));
		if (stitch->Img_output_rgb) ERROR_CHECK_STATUS_(vxReleaseImage(&stitch->Img_output_rgb));
		if (stitch->Img_overlay) ERROR_CHECK_STATUS_(vxReleaseImage(&stitch->Img_overlay));
		if (stitch->Img_overlay_rgb) ERROR_CHECK_STATUS_(vxReleaseImage(&stitch->Img_overlay_rgb));
		if (stitch->Img_overlay_rgba) ERROR_CHECK_STATUS_(vxReleaseImage(&stitch->Img_overlay_rgba));
		if (stitch->RGBY1) ERROR_CHECK_STATUS_(vxReleaseImage(&stitch->RGBY1));
		if (stitch->RGBY2) ERROR_CHECK_STATUS_(vxReleaseImage(&stitch->RGBY2));
		if (stitch->weight_image) ERROR_CHECK_STATUS_(vxReleaseImage(&stitch->weight_image));
		if (stitch->cam_id_image) ERROR_CHECK_STATUS_(vxReleaseImage(&stitch->cam_id_image));
		if (stitch->group1_image) ERROR_CHECK_STATUS_(vxReleaseImage(&stitch->group1_image));
		if (stitch->group2_image) ERROR_CHECK_STATUS_(vxReleaseImage(&stitch->group2_image));
		if (stitch->valid_mask_image) ERROR_CHECK_STATUS_(vxReleaseImage(&stitch->valid_mask_image));
		if (stitch->warp_luma_image) ERROR_CHECK_STATUS_(vxReleaseImage(&stitch->warp_luma_image));
		if (stitch->sobel_magnitude_s16_image) ERROR_CHECK_STATUS_(vxReleaseImage(&stitch->sobel_magnitude_s16_image));
		if (stitch->sobelx_image) ERROR_CHECK_STATUS_(vxReleaseImage(&stitch->sobelx_image));
		if (stitch->sobely_image) ERROR_CHECK_STATUS_(vxReleaseImage(&stitch->sobely_image));
		if (stitch->sobel_magnitude_image) ERROR_CHECK_STATUS_(vxReleaseImage(&stitch->sobel_magnitude_image));
		if (stitch->sobel_phase_image) ERROR_CHECK_STATUS_(vxReleaseImage(&stitch->sobel_phase_image));
		if (stitch->seamfind_weight_image) ERROR_CHECK_STATUS_(vxReleaseImage(&stitch->seamfind_weight_image));
		if (stitch->blend_mask_image) ERROR_CHECK_STATUS_(vxReleaseImage(&stitch->blend_mask_image));
		if (stitch->blend_offsets) ERROR_CHECK_STATUS_(vxReleaseArray(&stitch->blend_offsets));
		if (stitch->chroma_key_input_img) ERROR_CHECK_STATUS_(vxReleaseImage(&stitch->chroma_key_input_img));
		if (stitch->chroma_key_mask_img) ERROR_CHECK_STATUS_(vxReleaseImage(&stitch->chroma_key_mask_img));
		if (stitch->chroma_key_dilate_mask_img) ERROR_CHECK_STATUS_(vxReleaseImage(&stitch->chroma_key_dilate_mask_img));
		if (stitch->chroma_key_erode_mask_img) ERROR_CHECK_STATUS_(vxReleaseImage(&stitch->chroma_key_erode_mask_img));
		if (stitch->chroma_key_input_RGB_img) ERROR_CHECK_STATUS_(vxReleaseImage(&stitch->chroma_key_input_RGB_img));
		if (stitch->noiseFilterInput_image) ERROR_CHECK_STATUS_(vxReleaseImage(&stitch->noiseFilterInput_image));

		// release scalar objects
		if (stitch->current_frame) ERROR_CHECK_STATUS_(vxReleaseScalar(&stitch->current_frame));
		if (stitch->scene_threshold) ERROR_CHECK_STATUS_(vxReleaseScalar(&stitch->scene_threshold));
		if (stitch->seam_cost_enable) ERROR_CHECK_STATUS_(vxReleaseScalar(&stitch->seam_cost_enable));
		if (stitch->filterLambda) ERROR_CHECK_STATUS_(vxReleaseScalar(&stitch->filterLambda));

		// release remap objects
		if (stitch->overlay_remap) ERROR_CHECK_STATUS_(vxReleaseRemap(&stitch->overlay_remap));
		if (stitch->camera_remap) ERROR_CHECK_STATUS_(vxReleaseRemap(&stitch->camera_remap));

		// release matrix
		if (stitch->overlap_matrix) ERROR_CHECK_STATUS_(vxReleaseMatrix(&stitch->overlap_matrix));
		if (stitch->A_matrix) ERROR_CHECK_STATUS_(vxReleaseMatrix(&stitch->A_matrix));
		if (stitch->A_matrix_initial_value) delete[] stitch->A_matrix_initial_value;

		// release arrays
		if (stitch->ValidPixelEntry) ERROR_CHECK_STATUS_(vxReleaseArray(&stitch->ValidPixelEntry));
		if (stitch->WarpRemapEntry) ERROR_CHECK_STATUS_(vxReleaseArray(&stitch->WarpRemapEntry));
		if (stitch->OverlapPixelEntry) ERROR_CHECK_STATUS_(vxReleaseArray(&stitch->OverlapPixelEntry));
		if (stitch->valid_array) ERROR_CHECK_STATUS_(vxReleaseArray(&stitch->valid_array));
		if (stitch->gain_array) ERROR_CHECK_STATUS_(vxReleaseArray(&stitch->gain_array));
		if (stitch->overlap_rect_array) ERROR_CHECK_STATUS_(vxReleaseArray(&stitch->overlap_rect_array));
		if (stitch->seamfind_valid_array) ERROR_CHECK_STATUS_(vxReleaseArray(&stitch->seamfind_valid_array));
		if (stitch->seamfind_weight_array) ERROR_CHECK_STATUS_(vxReleaseArray(&stitch->seamfind_weight_array));
		if (stitch->seamfind_accum_array) ERROR_CHECK_STATUS_(vxReleaseArray(&stitch->seamfind_accum_array));
		if (stitch->seamfind_pref_array) ERROR_CHECK_STATUS_(vxReleaseArray(&stitch->seamfind_pref_array));
		if (stitch->seamfind_path_array) ERROR_CHECK_STATUS_(vxReleaseArray(&stitch->seamfind_path_array));
		if (stitch->seamfind_scene_array) ERROR_CHECK_STATUS_(vxReleaseArray(&stitch->seamfind_scene_array));

		// release delay element
		if (stitch->noiseFilterImageDelay) ERROR_CHECK_STATUS_(vxReleaseDelay(&stitch->noiseFilterImageDelay));

		// release node objects
		if (stitch->InputColorConvertNode) ERROR_CHECK_STATUS_(vxReleaseNode(&stitch->InputColorConvertNode));
		if (stitch->SimpleStitchRemapNode) ERROR_CHECK_STATUS_(vxReleaseNode(&stitch->SimpleStitchRemapNode));
		if (stitch->OutputColorConvertNode) ERROR_CHECK_STATUS_(vxReleaseNode(&stitch->OutputColorConvertNode));
		if (stitch->nodeOverlayRemap) ERROR_CHECK_STATUS_(vxReleaseNode(&stitch->nodeOverlayRemap));
		if (stitch->nodeOverlayBlend) ERROR_CHECK_STATUS_(vxReleaseNode(&stitch->nodeOverlayBlend));
		if (stitch->WarpNode) ERROR_CHECK_STATUS_(vxReleaseNode(&stitch->WarpNode));
		if (stitch->ExpcompComputeGainNode) ERROR_CHECK_STATUS_(vxReleaseNode(&stitch->ExpcompComputeGainNode));
		if (stitch->ExpcompSolveGainNode) ERROR_CHECK_STATUS_(vxReleaseNode(&stitch->ExpcompSolveGainNode));
		if (stitch->ExpcompApplyGainNode) ERROR_CHECK_STATUS_(vxReleaseNode(&stitch->ExpcompApplyGainNode));
		if (stitch->MergeNode) ERROR_CHECK_STATUS_(vxReleaseNode(&stitch->MergeNode));
		if (stitch->SobelNode) ERROR_CHECK_STATUS_(vxReleaseNode(&stitch->SobelNode));
		if (stitch->MagnitudeNode) ERROR_CHECK_STATUS_(vxReleaseNode(&stitch->MagnitudeNode));
		if (stitch->PhaseNode) ERROR_CHECK_STATUS_(vxReleaseNode(&stitch->PhaseNode));
		if (stitch->ConvertDepthNode) ERROR_CHECK_STATUS_(vxReleaseNode(&stitch->ConvertDepthNode));
		if (stitch->SeamfindStep1Node) ERROR_CHECK_STATUS_(vxReleaseNode(&stitch->SeamfindStep1Node));
		if (stitch->SeamfindStep2Node) ERROR_CHECK_STATUS_(vxReleaseNode(&stitch->SeamfindStep2Node));
		if (stitch->SeamfindStep3Node) ERROR_CHECK_STATUS_(vxReleaseNode(&stitch->SeamfindStep3Node));
		if (stitch->SeamfindStep4Node) ERROR_CHECK_STATUS_(vxReleaseNode(&stitch->SeamfindStep4Node));
		if (stitch->SeamfindStep5Node) ERROR_CHECK_STATUS_(vxReleaseNode(&stitch->SeamfindStep5Node));
		if (stitch->SeamfindAnalyzeNode) ERROR_CHECK_STATUS_(vxReleaseNode(&stitch->SeamfindAnalyzeNode));
		if (stitch->chromaKey_mask_generation_node) ERROR_CHECK_STATUS_(vxReleaseNode(&stitch->chromaKey_mask_generation_node));
		if (stitch->chromaKey_dilate_node) ERROR_CHECK_STATUS_(vxReleaseNode(&stitch->chromaKey_dilate_node));
		if (stitch->chromaKey_erode_node) ERROR_CHECK_STATUS_(vxReleaseNode(&stitch->chromaKey_erode_node));
		if (stitch->chromaKey_merge_node) ERROR_CHECK_STATUS_(vxReleaseNode(&stitch->chromaKey_merge_node));
		if (stitch->noiseFilterNode) ERROR_CHECK_STATUS_(vxReleaseNode(&stitch->noiseFilterNode));
		if (stitch->MULTIBAND_BLEND && stitch->pStitchMultiband){
			for (int i = 0; i < stitch->num_bands; i++){
				if (stitch->pStitchMultiband[i].BlendNode)ERROR_CHECK_STATUS_(vxReleaseNode(&stitch->pStitchMultiband[i].BlendNode));
				if (stitch->pStitchMultiband[i].SourceHSGNode)ERROR_CHECK_STATUS_(vxReleaseNode(&stitch->pStitchMultiband[i].SourceHSGNode));
				if (stitch->pStitchMultiband[i].WeightHSGNode)ERROR_CHECK_STATUS_(vxReleaseNode(&stitch->pStitchMultiband[i].WeightHSGNode));
				if (stitch->pStitchMultiband[i].UpscaleSubtractNode)ERROR_CHECK_STATUS_(vxReleaseNode(&stitch->pStitchMultiband[i].UpscaleSubtractNode));
				if (stitch->pStitchMultiband[i].UpscaleAddNode)ERROR_CHECK_STATUS_(vxReleaseNode(&stitch->pStitchMultiband[i].UpscaleAddNode));
				if (stitch->pStitchMultiband[i].LaplacianReconNode)ERROR_CHECK_STATUS_(vxReleaseNode(&stitch->pStitchMultiband[i].LaplacianReconNode));
				if (stitch->pStitchMultiband[i].DstPyrImgGaussian) ERROR_CHECK_STATUS_(vxReleaseImage(&stitch->pStitchMultiband[i].DstPyrImgGaussian));
				if (stitch->pStitchMultiband[i].WeightPyrImgGaussian) ERROR_CHECK_STATUS_(vxReleaseImage(&stitch->pStitchMultiband[i].WeightPyrImgGaussian));
				if (stitch->pStitchMultiband[i].DstPyrImgLaplacian) ERROR_CHECK_STATUS_(vxReleaseImage(&stitch->pStitchMultiband[i].DstPyrImgLaplacian));
				if (stitch->pStitchMultiband[i].DstPyrImgLaplacianRec) ERROR_CHECK_STATUS_(vxReleaseImage(&stitch->pStitchMultiband[i].DstPyrImgLaplacianRec));
			}
			delete[] stitch->pStitchMultiband;
			delete[] stitch->multibandBlendOffsetIntoBuffer;
		}

		// release LoomIO objects
		if (stitch->cameraMediaConfig) ERROR_CHECK_STATUS_(vxReleaseScalar(&stitch->cameraMediaConfig));
		if (stitch->overlayMediaConfig) ERROR_CHECK_STATUS_(vxReleaseScalar(&stitch->overlayMediaConfig));
		if (stitch->outputMediaConfig) ERROR_CHECK_STATUS_(vxReleaseScalar(&stitch->outputMediaConfig));
		if (stitch->viewingMediaConfig) ERROR_CHECK_STATUS_(vxReleaseScalar(&stitch->viewingMediaConfig));
		if (stitch->loomioCameraAuxData) ERROR_CHECK_STATUS_(vxReleaseArray(&stitch->loomioCameraAuxData));
		if (stitch->loomioOverlayAuxData) ERROR_CHECK_STATUS_(vxReleaseArray(&stitch->loomioOverlayAuxData));
		if (stitch->loomioOutputAuxData) ERROR_CHECK_STATUS_(vxReleaseArray(&stitch->loomioOutputAuxData));
		if (stitch->loomioViewingAuxData) ERROR_CHECK_STATUS_(vxReleaseArray(&stitch->loomioViewingAuxData));
		if (stitch->nodeLoomIoCamera) ERROR_CHECK_STATUS_(vxReleaseNode(&stitch->nodeLoomIoCamera));
		if (stitch->nodeLoomIoOverlay) ERROR_CHECK_STATUS_(vxReleaseNode(&stitch->nodeLoomIoOverlay));
		if (stitch->nodeLoomIoOutput) ERROR_CHECK_STATUS_(vxReleaseNode(&stitch->nodeLoomIoOutput));
		if (stitch->nodeLoomIoViewing) ERROR_CHECK_STATUS_(vxReleaseNode(&stitch->nodeLoomIoViewing));

		// release tiled image elements
		if (stitch->num_encode_sections > 1){
			for (vx_uint32 i = 0; i < stitch->num_encode_sections; i++){
				if (stitch->encode_src_rgb_imgs[i]) ERROR_CHECK_STATUS_(vxReleaseImage(&stitch->encode_src_rgb_imgs[i]));
				if (stitch->encode_dst_imgs[i]) ERROR_CHECK_STATUS_(vxReleaseImage(&stitch->encode_dst_imgs[i]));
				if (stitch->encode_color_convert_nodes[i]) ERROR_CHECK_STATUS_(vxReleaseNode(&stitch->encode_color_convert_nodes[i]));
				if (stitch->encodetileOutput[i])ERROR_CHECK_STATUS_(vxReleaseImage(&stitch->encodetileOutput[i]));
			}
		}

		// release fast GPU initialize elements
		if (!stitch->USE_CPU_INIT){
			if (stitch->stitchInitData){
				if (stitch->stitchInitData->CameraParamsArr) ERROR_CHECK_STATUS_(vxReleaseArray(&stitch->stitchInitData->CameraParamsArr));
				if (stitch->stitchInitData->CameraZBuffArr) ERROR_CHECK_STATUS_(vxReleaseArray(&stitch->stitchInitData->CameraZBuffArr));
				if (stitch->stitchInitData->DefaultCamMap) ERROR_CHECK_STATUS_(vxReleaseImage(&stitch->stitchInitData->DefaultCamMap));
				if (stitch->stitchInitData->ValidPixelMap) ERROR_CHECK_STATUS_(vxReleaseImage(&stitch->stitchInitData->ValidPixelMap));
				if (stitch->stitchInitData->PaddedPixMap) ERROR_CHECK_STATUS_(vxReleaseImage(&stitch->stitchInitData->PaddedPixMap));
				if (stitch->stitchInitData->SrcCoordMap) ERROR_CHECK_STATUS_(vxReleaseImage(&stitch->stitchInitData->SrcCoordMap));
				if (stitch->stitchInitData->calc_warp_maps_node) ERROR_CHECK_STATUS_(vxReleaseNode(&stitch->stitchInitData->calc_warp_maps_node));
				if (stitch->stitchInitData->calc_default_idx_node) ERROR_CHECK_STATUS_(vxReleaseNode(&stitch->stitchInitData->calc_default_idx_node));
				if (stitch->stitchInitData->pad_dilate_node) ERROR_CHECK_STATUS_(vxReleaseNode(&stitch->stitchInitData->pad_dilate_node));
				if (stitch->stitchInitData->graphInitialize) ERROR_CHECK_STATUS_(vxReleaseGraph(&stitch->stitchInitData->graphInitialize));
				delete[] stitch->stitchInitData;
				stitch->stitchInitData = nullptr;
			}
		}

		// release internal buffers
		if (stitch->validPixelCamMap) { delete[] stitch->validPixelCamMap; stitch->validPixelCamMap = nullptr; }
		if (stitch->paddedPixelCamMap) { delete[] stitch->paddedPixelCamMap; stitch->paddedPixelCamMap = nullptr; }
		if (stitch->camSrcMap) { delete[] stitch->camSrcMap; stitch->camSrcMap = nullptr; }
		if (stitch->overlapRectBuf) { delete[] stitch->overlapRectBuf; stitch->overlapRectBuf = nullptr; }
		if (stitch->camIndexTmpBuf) { delete[] stitch->camIndexTmpBuf; stitch->camIndexTmpBuf = nullptr; }
		if (stitch->camIndexBuf) { delete[] stitch->camIndexBuf; stitch->camIndexBuf = nullptr; }
		if (stitch->overlapMatrixBuf) { delete[] stitch->overlapMatrixBuf; stitch->overlapMatrixBuf = nullptr; }
		if (stitch->overlaySrcMap) { delete[] stitch->overlaySrcMap; stitch->overlaySrcMap = nullptr; }
		if (stitch->validPixelOverlayMap) { delete[] stitch->validPixelOverlayMap; stitch->validPixelOverlayMap = nullptr; }
		if (stitch->overlayIndexTmpBuf) { delete[] stitch->overlayIndexTmpBuf; stitch->overlayIndexTmpBuf = nullptr; }
		if (stitch->overlayIndexBuf) { delete[] stitch->overlayIndexBuf; stitch->overlayIndexBuf = nullptr; }

		// debug aux dumps
		if (stitch->loomioAuxDumpFile) {
			fclose(stitch->loomioAuxDumpFile);
		}

		// clear the magic and destroy
		stitch->magic = ~LIVE_STITCH_MAGIC;

		//Graph & Context
		if (stitch->graphStitch) ERROR_CHECK_STATUS_(vxReleaseGraph(&stitch->graphStitch));
		if (stitch->context && !stitch->context_is_external) ERROR_CHECK_STATUS_(vxReleaseContext(&stitch->context));

		delete stitch;
		*pStitch = nullptr;
	}
	PROFILER_SHUTDOWN();
	return VX_SUCCESS;
}

//! \brief Set OpenCL buffers
//     input_buffer   - input opencl buffer with images from all cameras
//     overlay_buffer - overlay opencl buffer with all images
//     output_buffer  - output opencl buffer for output equirectangular image
//     chromaKey_buffer  - chroma key opencl buffer for equirectangular image
//   Use of nullptr will return the control of previously set buffer
LIVE_STITCH_API_ENTRY vx_status VX_API_CALL lsSetCameraBuffer(ls_context stitch, cl_mem * input_buffer)
{
	PROFILER_START(LoomSL, SetInputBuffer);
	ERROR_CHECK_STATUS_(IsValidContextAndInitialized(stitch));
	// check to make sure that LoomIO for camera is not active
	if (stitch->nodeLoomIoCamera) return VX_ERROR_NOT_ALLOCATED;

	// switch the user specified OpenCL buffer into image
	if (stitch->camera_buffer_format == VX_DF_IMAGE_NV12) {
		void * ptr_in[] = { input_buffer ? input_buffer[0] : nullptr, input_buffer ? input_buffer[1] : nullptr };
		ERROR_CHECK_STATUS_(vxSwapImageHandle(stitch->Img_input, ptr_in, nullptr, 2));
	}
	else {
		void * ptr_in[] = { input_buffer ? input_buffer[0] : nullptr };
		ERROR_CHECK_STATUS_(vxSwapImageHandle(stitch->Img_input, ptr_in, nullptr, 1));
	}
	PROFILER_STOP(LoomSL, SetInputBuffer);
	return VX_SUCCESS;
}
LIVE_STITCH_API_ENTRY vx_status VX_API_CALL lsSetOutputBuffer(ls_context stitch, cl_mem * output_buffer)
{
	PROFILER_START(LoomSL, SetOutputBuffer);
	ERROR_CHECK_STATUS_(IsValidContextAndInitialized(stitch));
	// check to make sure that LoomIO for output is not active
	if (stitch->nodeLoomIoOutput) return VX_ERROR_NOT_ALLOCATED;

	// switch the user specified OpenCL buffer into image
	if (stitch->output_buffer_format == VX_DF_IMAGE_NV12) {
		if (stitch->output_encode_tiles > 1) {
			for (vx_uint32 i = 0; i < stitch->output_encode_tiles; i++){
				void * ptr_out[] = { output_buffer ? output_buffer[(i * 2)] : nullptr, output_buffer ? output_buffer[(i * 2) + 1] : nullptr };
				ERROR_CHECK_STATUS_(vxSwapImageHandle(stitch->encodetileOutput[i], ptr_out, nullptr, 2));
			}
		}
		else {
			void * ptr_out[] = { output_buffer ? output_buffer[0] : nullptr, output_buffer ? output_buffer[1] : nullptr };
			ERROR_CHECK_STATUS_(vxSwapImageHandle(stitch->Img_output, ptr_out, nullptr, 2));
		}
	}
	else {
		void * ptr_out[] = { output_buffer ? output_buffer[0] : nullptr };
		ERROR_CHECK_STATUS_(vxSwapImageHandle(stitch->Img_output, ptr_out, nullptr, 1));
	}
	PROFILER_STOP(LoomSL, SetOutputBuffer);
	return VX_SUCCESS;
}
LIVE_STITCH_API_ENTRY vx_status VX_API_CALL lsSetOverlayBuffer(ls_context stitch, cl_mem * overlay_buffer)
{
	ERROR_CHECK_STATUS_(IsValidContextAndInitialized(stitch));
	// check to make sure that LoomIO for overlay is not active
	if (stitch->nodeLoomIoOverlay) return VX_ERROR_NOT_ALLOCATED;

	// switch the user specified OpenCL buffer into image
	void * ptr_overlay[] = { overlay_buffer ? overlay_buffer[0] : nullptr };
	ERROR_CHECK_STATUS_(vxSwapImageHandle(stitch->Img_overlay, ptr_overlay, nullptr, 1));

	return VX_SUCCESS;
}
LIVE_STITCH_API_ENTRY vx_status VX_API_CALL lsSetChromaKeyBuffer(ls_context stitch, cl_mem * chromaKey_buffer)
{
	ERROR_CHECK_STATUS_(IsValidContextAndInitialized(stitch));
	if (!stitch->CHROMA_KEY) return VX_ERROR_NOT_ALLOCATED;

	// switch the user specified OpenCL buffer into image
	void * ptr_chroma[] = { chromaKey_buffer ? chromaKey_buffer[0] : nullptr };
	ERROR_CHECK_STATUS_(vxSwapImageHandle(stitch->chroma_key_input_img, ptr_chroma, nullptr, 1));

	return VX_SUCCESS;
}

//! \brief Schedule next frame
LIVE_STITCH_API_ENTRY vx_status VX_API_CALL lsScheduleFrame(ls_context stitch)
{
	PROFILER_START(LoomSL, ScheduleGraph);
	ERROR_CHECK_STATUS_(IsValidContextAndInitialized(stitch));
	if (stitch->scheduled) {
		ls_printf("ERROR: lsScheduleFrame: already scheduled\n");
		return VX_ERROR_GRAPH_SCHEDULED;
	}
	if (stitch->reinitialize_required) {
		ls_printf("ERROR: lsScheduleFrame: reinitialize required\n");
		return VX_FAILURE;
	}

	// seamfind needs frame counter values to be incremented
	if (stitch->SEAM_FIND) {
		ERROR_CHECK_STATUS_(vxWriteScalarValue(stitch->current_frame, &stitch->current_frame_value));
		stitch->current_frame_value++;
	}

	// exposure comp expects A_matrix to be initialized to ZERO on GPU
	if ((stitch->EXPO_COMP <= 2) && stitch->A_matrix) {
		ERROR_CHECK_STATUS_(vxWriteMatrix(stitch->A_matrix, stitch->A_matrix_initial_value));
		ERROR_CHECK_STATUS_(vxDirective((vx_reference)stitch->A_matrix, VX_DIRECTIVE_AMD_COPY_TO_OPENCL));
	}

	// age delay element if temporal noise filter activated
	if (stitch->NOISE_FILTER){
		ERROR_CHECK_STATUS_(vxAgeDelay(stitch->noiseFilterImageDelay));
	}

	// start the graph schedule
	ERROR_CHECK_STATUS_(vxScheduleGraph(stitch->graphStitch));
	stitch->scheduled = true;
	PROFILER_STOP(LoomSL, ScheduleGraph);
	return VX_SUCCESS;
}

//! \brief Schedule next frame
LIVE_STITCH_API_ENTRY vx_status VX_API_CALL lsWaitForCompletion(ls_context stitch)
{
	PROFILER_START(LoomSL, WaitForCompletionGraph);
	ERROR_CHECK_STATUS_(IsValidContextAndInitialized(stitch));
	if (!stitch->scheduled) {
		ls_printf("ERROR: lsWaitForCompletion: not scheduled\n");
		return VX_ERROR_GRAPH_SCHEDULED;
	}

	// wait for graph completion
	ERROR_CHECK_STATUS_(vxWaitGraph(stitch->graphStitch));
	stitch->scheduled = false;

	// debug: dump auxiliary data
	if (stitch->loomioAuxDumpFile) {
		vx_array auxList[] = { stitch->loomioCameraAuxData, stitch->loomioOverlayAuxData, stitch->loomioOutputAuxData, stitch->loomioViewingAuxData };
		for (size_t i = 0; i < sizeof(auxList) / sizeof(auxList[0]); i++) {
			if (auxList[i]) {
				vx_size numItems = 0;
				ERROR_CHECK_STATUS_(vxQueryArray(auxList[i], VX_ARRAY_NUMITEMS, &numItems, sizeof(numItems)));
				if (numItems > 0) {
					vx_map_id map_id = 0;
					vx_size stride = 0;
					char * ptr = nullptr;
					ERROR_CHECK_STATUS_(vxMapArrayRange(auxList[i], 0, numItems, &map_id, &stride, (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
					fwrite(ptr, 1, numItems * stride, stitch->loomioAuxDumpFile);
					fflush(stitch->loomioAuxDumpFile);
					ERROR_CHECK_STATUS_(vxUnmapArrayRange(auxList[i], map_id));
				}
			}
		}
	}
	PROFILER_STOP(LoomSL, WaitForCompletionGraph);
	return VX_SUCCESS;
}

//! \brief query functions.
LIVE_STITCH_API_ENTRY vx_status VX_API_CALL lsGetOpenVXContext(ls_context stitch, vx_context  * openvx_context)
{
	ERROR_CHECK_STATUS_(IsValidContext(stitch));
	if (!stitch->context) {
		stitch->context = vxCreateContext();
		ERROR_CHECK_OBJECT_(stitch->context);
		if (stitch->opencl_context) {
			ERROR_CHECK_STATUS_(vxSetContextAttribute(stitch->context, VX_CONTEXT_ATTRIBUTE_AMD_OPENCL_CONTEXT, &stitch->opencl_context, sizeof(cl_context)));
		}
	}
	*openvx_context = stitch->context;
	return VX_SUCCESS;
}
LIVE_STITCH_API_ENTRY vx_status VX_API_CALL lsGetOpenCLContext(ls_context stitch, cl_context  * opencl_context)
{
	ERROR_CHECK_STATUS_(IsValidContext(stitch));
	if (!stitch->opencl_context) {
		vx_context openvx_context = nullptr;
		vx_status status = lsGetOpenVXContext(stitch, &openvx_context);
		if (status)
			return status;
		status = vxQueryContext(openvx_context, VX_CONTEXT_ATTRIBUTE_AMD_OPENCL_CONTEXT, &stitch->opencl_context, sizeof(cl_context));
		if (status)
			return status;
	}
	*opencl_context = stitch->opencl_context;
	return VX_SUCCESS;
}
LIVE_STITCH_API_ENTRY vx_status VX_API_CALL lsGetRigParams(ls_context stitch, rig_params * par)
{
	ERROR_CHECK_STATUS_(IsValidContext(stitch));
	memcpy(par, &stitch->rig_par, sizeof(rig_params));
	return VX_SUCCESS;
}
LIVE_STITCH_API_ENTRY vx_status VX_API_CALL lsGetCameraConfig(ls_context stitch, vx_uint32 * num_camera_rows, vx_uint32 * num_camera_columns, vx_df_image * buffer_format, vx_uint32 * buffer_width, vx_uint32 * buffer_height)
{
	ERROR_CHECK_STATUS_(IsValidContext(stitch));
	*num_camera_rows = stitch->num_camera_rows;
	*num_camera_columns = stitch->num_camera_columns;
	*buffer_format = stitch->camera_buffer_format;
	*buffer_width = stitch->camera_buffer_width;
	*buffer_height = stitch->camera_buffer_height;
	return VX_SUCCESS;
}
LIVE_STITCH_API_ENTRY vx_status VX_API_CALL lsGetOutputConfig(ls_context stitch, vx_df_image * buffer_format, vx_uint32 * buffer_width, vx_uint32 * buffer_height)
{
	ERROR_CHECK_STATUS_(IsValidContext(stitch));
	*buffer_format = stitch->output_buffer_format;
	*buffer_width = stitch->output_buffer_width;
	*buffer_height = stitch->output_buffer_height;
	return VX_SUCCESS;
}
LIVE_STITCH_API_ENTRY vx_status VX_API_CALL lsGetOverlayConfig(ls_context stitch, vx_uint32 * num_overlay_rows, vx_uint32 * num_overlay_columns, vx_df_image * buffer_format, vx_uint32 * buffer_width, vx_uint32 * buffer_height)
{
	ERROR_CHECK_STATUS_(IsValidContext(stitch));
	*num_overlay_rows = stitch->num_overlay_rows;
	*num_overlay_columns = stitch->num_overlay_columns;
	*buffer_format = VX_DF_IMAGE_RGBX;
	*buffer_width = stitch->overlay_buffer_width;
	*buffer_height = stitch->overlay_buffer_height;
	return VX_SUCCESS;
}
LIVE_STITCH_API_ENTRY vx_status VX_API_CALL lsGetCameraParams(ls_context stitch, vx_uint32 cam_index, camera_params * par)
{
	ERROR_CHECK_STATUS_(IsValidContext(stitch));
	if (cam_index >= stitch->num_cameras) {
		ls_printf("ERROR: lsGetCameraParams: invalid camera index (%d)\n", cam_index);
		return VX_ERROR_INVALID_VALUE;
	}
	memcpy(par, &stitch->camera_par[cam_index], sizeof(camera_params));
	return VX_SUCCESS;
}
LIVE_STITCH_API_ENTRY vx_status VX_API_CALL lsGetOverlayParams(ls_context stitch, vx_uint32 overlay_index, camera_params * par)
{
	ERROR_CHECK_STATUS_(IsValidContext(stitch));
	if (overlay_index >= stitch->num_overlays) {
		ls_printf("ERROR: lsGetOverlayParams: invalid camera index (%d)\n", overlay_index);
		return VX_ERROR_INVALID_VALUE;
	}
	memcpy(par, &stitch->overlay_par[overlay_index], sizeof(camera_params));
	return VX_SUCCESS;
}
LIVE_STITCH_API_ENTRY vx_status VX_API_CALL lsGetCameraBufferStride(ls_context stitch, vx_uint32 * camera_buffer_stride_in_bytes)
{
	ERROR_CHECK_STATUS_(IsValidContext(stitch));
	*camera_buffer_stride_in_bytes = stitch->camera_buffer_stride_in_bytes;
	return VX_SUCCESS;
}
LIVE_STITCH_API_ENTRY vx_status VX_API_CALL lsGetOutputBufferStride(ls_context stitch, vx_uint32 * output_buffer_stride_in_bytes)
{
	ERROR_CHECK_STATUS_(IsValidContext(stitch));
	*output_buffer_stride_in_bytes = stitch->output_buffer_stride_in_bytes;
	return VX_SUCCESS;
}
LIVE_STITCH_API_ENTRY vx_status VX_API_CALL lsGetOverlayBufferStride(ls_context stitch, vx_uint32 * overlay_buffer_stride_in_bytes)
{
	ERROR_CHECK_STATUS_(IsValidContext(stitch));
	*overlay_buffer_stride_in_bytes = stitch->overlay_buffer_stride_in_bytes;
	return VX_SUCCESS;
}
LIVE_STITCH_API_ENTRY vx_status VX_API_CALL lsGetCameraModule(ls_context stitch, char * openvx_module, size_t openvx_module_size, char * kernelName, size_t kernelName_size, char * kernelArguments, size_t kernelArguments_size)
{
	ERROR_CHECK_STATUS_(IsValidContext(stitch));
	strncpy(openvx_module, stitch->loomio_camera.module, openvx_module_size);
	strncpy(kernelName, stitch->loomio_camera.kernelName, kernelName_size);
	strncpy(kernelArguments, stitch->loomio_camera.kernelArguments, kernelArguments_size);
	openvx_module[openvx_module_size-1] = '\0';
	kernelName[kernelName_size - 1] = '\0';
	kernelArguments[kernelArguments_size - 1] = '\0';
	return VX_SUCCESS;
}
LIVE_STITCH_API_ENTRY vx_status VX_API_CALL lsGetOutputModule(ls_context stitch, char * openvx_module, size_t openvx_module_size, char * kernelName, size_t kernelName_size, char * kernelArguments, size_t kernelArguments_size)
{
	ERROR_CHECK_STATUS_(IsValidContext(stitch));
	strncpy(openvx_module, stitch->loomio_output.module, openvx_module_size);
	strncpy(kernelName, stitch->loomio_output.kernelName, kernelName_size);
	strncpy(kernelArguments, stitch->loomio_output.kernelArguments, kernelArguments_size);
	openvx_module[openvx_module_size - 1] = '\0';
	kernelName[kernelName_size - 1] = '\0';
	kernelArguments[kernelArguments_size - 1] = '\0';
	return VX_SUCCESS;
}
LIVE_STITCH_API_ENTRY vx_status VX_API_CALL lsGetOverlayModule(ls_context stitch, char * openvx_module, size_t openvx_module_size, char * kernelName, size_t kernelName_size, char * kernelArguments, size_t kernelArguments_size)
{
	ERROR_CHECK_STATUS_(IsValidContext(stitch));
	strncpy(openvx_module, stitch->loomio_overlay.module, openvx_module_size);
	strncpy(kernelName, stitch->loomio_overlay.kernelName, kernelName_size);
	strncpy(kernelArguments, stitch->loomio_overlay.kernelArguments, kernelArguments_size);
	openvx_module[openvx_module_size - 1] = '\0';
	kernelName[kernelName_size - 1] = '\0';
	kernelArguments[kernelArguments_size - 1] = '\0';
	return VX_SUCCESS;
}
LIVE_STITCH_API_ENTRY vx_status VX_API_CALL lsGetViewingModule(ls_context stitch, char * openvx_module, size_t openvx_module_size, char * kernelName, size_t kernelName_size, char * kernelArguments, size_t kernelArguments_size)
{
	ERROR_CHECK_STATUS_(IsValidContext(stitch));
	strncpy(openvx_module, stitch->loomio_viewing.module, openvx_module_size);
	strncpy(kernelName, stitch->loomio_viewing.kernelName, kernelName_size);
	strncpy(kernelArguments, stitch->loomio_viewing.kernelArguments, kernelArguments_size);
	openvx_module[openvx_module_size - 1] = '\0';
	kernelName[kernelName_size - 1] = '\0';
	kernelArguments[kernelArguments_size - 1] = '\0';
	return VX_SUCCESS;
}
LIVE_STITCH_API_ENTRY vx_status VX_API_CALL lsExportConfiguration(ls_context stitch, const char * exportType, const char * fileName)
{
	ERROR_CHECK_STATUS_(IsValidContext(stitch));
	if (!_stricmp(exportType, "loom_shell")) {
		FILE * fp = stdout;
		if (fileName) {
			fp = fopen(fileName, "w");
			if (!fp) {
				ls_printf("ERROR: lsExportConfiguration: unable to create: %s\n", fileName);
				if (fp != NULL)	fclose(fp);
				return VX_FAILURE;
			}
		}
		fprintf(fp, "ls_context context;\n");
		fprintf(fp, "context = lsCreateContext();\n");
		for (vx_uint32 i = 0; i < stitch->num_cameras; i++) {
			camera_params camera_par = stitch->camera_par[i];
			char lensType[64];
			if (camera_par.lens.lens_type == ptgui_lens_rectilinear) strcpy(lensType, "ptgui_lens_rectilinear");
			else if (camera_par.lens.lens_type == ptgui_lens_fisheye_ff) strcpy(lensType, "ptgui_lens_fisheye_ff");
			else if (camera_par.lens.lens_type == ptgui_lens_fisheye_circ) strcpy(lensType, "ptgui_lens_fisheye_circ");
			else if (camera_par.lens.lens_type == adobe_lens_rectilinear) strcpy(lensType, "adobe_lens_rectilinear");
			else if (camera_par.lens.lens_type == adobe_lens_fisheye) strcpy(lensType, "adobe_lens_fisheye");
			else {
				ls_printf("ERROR: lsExportConfiguration: loom_shell: (cam) lens_type=%d not supported\n", camera_par.lens.lens_type);
				return VX_ERROR_NOT_SUPPORTED;
			}
			fprintf(fp, "camera_params cam%02d_par = {{%8.3f,%8.3f,%8.3f,%.0f,%.0f,%.0f},{%.3f,%.1f,%.1f,%.3f,%.3f,%s,%f,%f,%f", i,
				camera_par.focal.yaw, camera_par.focal.pitch, camera_par.focal.roll,
				camera_par.focal.tx, camera_par.focal.ty, camera_par.focal.tz,
				camera_par.lens.hfov, camera_par.lens.haw, camera_par.lens.r_crop,
				camera_par.lens.du0, camera_par.lens.dv0,
				lensType, camera_par.lens.k1, camera_par.lens.k2, camera_par.lens.k3);
			vx_int32 lastNonZeroField = -1;
			for (vx_int32 i = 0; i < 7; i++) {
				if (camera_par.lens.reserved[i])
					lastNonZeroField = i;
			}
			for (vx_int32 i = 0; i <= lastNonZeroField; i++) {
				fprintf(fp, ",%f", camera_par.lens.reserved[i]);
			}
			fprintf(fp, "}};\n");
		}
		if (stitch->num_overlays > 0) {
			for (vx_uint32 i = 0; i < stitch->num_overlays; i++) {
				camera_params camera_par = stitch->overlay_par[i];
				char lensType[64];
				if (camera_par.lens.lens_type == ptgui_lens_rectilinear) strcpy(lensType, "ptgui_lens_rectilinear");
				else if (camera_par.lens.lens_type == ptgui_lens_fisheye_ff) strcpy(lensType, "ptgui_lens_fisheye_ff");
				else if (camera_par.lens.lens_type == ptgui_lens_fisheye_circ) strcpy(lensType, "ptgui_lens_fisheye_circ");
				else if (camera_par.lens.lens_type == adobe_lens_rectilinear) strcpy(lensType, "adobe_lens_rectilinear");
				else if (camera_par.lens.lens_type == adobe_lens_fisheye) strcpy(lensType, "adobe_lens_fisheye");
				else {
					ls_printf("ERROR: lsExportConfiguration: loom_shell: (ovr) lens_type=%d not supported\n", camera_par.lens.lens_type);
					return VX_ERROR_NOT_SUPPORTED;
				}
				fprintf(fp, "camera_params ovr%02d_par = {{%8.3f,%8.3f,%8.3f,%.0f,%.0f,%.0f},{%.3f,%.1f,%.1f,%.3f,%.3f,%s,%f,%f,%f", i,
					camera_par.focal.yaw, camera_par.focal.pitch, camera_par.focal.roll,
					camera_par.focal.tx, camera_par.focal.ty, camera_par.focal.tz,
					camera_par.lens.hfov, camera_par.lens.haw, camera_par.lens.r_crop,
					camera_par.lens.du0, camera_par.lens.dv0,
					lensType, camera_par.lens.k1, camera_par.lens.k2, camera_par.lens.k3);
				vx_int32 lastNonZeroField = -1;
				for (vx_int32 i = 0; i < 7; i++) {
					if (camera_par.lens.reserved[i])
						lastNonZeroField = i;
				}
				for (vx_int32 i = 0; i <= lastNonZeroField; i++) {
					fprintf(fp, ",%f", camera_par.lens.reserved[i]);
				}
				fprintf(fp, "}};\n");
			}
		}
		char formatName[64];
		if (stitch->output_buffer_format == VX_DF_IMAGE_RGB) strcpy(formatName, "VX_DF_IMAGE_RGB");
		else if (stitch->output_buffer_format == VX_DF_IMAGE_UYVY) strcpy(formatName, "VX_DF_IMAGE_UYVY");
		else if (stitch->output_buffer_format == VX_DF_IMAGE_YUYV) strcpy(formatName, "VX_DF_IMAGE_YUYV");
		else { memcpy(formatName, &stitch->output_buffer_format, 4); formatName[4] = 0; }
		fprintf(fp, "lsSetOutputConfig(context,%s,%d,%d);\n", formatName, stitch->output_buffer_width, stitch->output_buffer_height);
		if (stitch->camera_buffer_format == VX_DF_IMAGE_RGB) strcpy(formatName, "VX_DF_IMAGE_RGB");
		else if (stitch->camera_buffer_format == VX_DF_IMAGE_UYVY) strcpy(formatName, "VX_DF_IMAGE_UYVY");
		else if (stitch->camera_buffer_format == VX_DF_IMAGE_YUYV) strcpy(formatName, "VX_DF_IMAGE_YUYV");
		else { memcpy(formatName, &stitch->camera_buffer_format, 4); formatName[4] = 0; }
		fprintf(fp, "lsSetCameraConfig(context,%d,%d,%s,%d,%d);\n", stitch->num_camera_rows, stitch->num_camera_columns, formatName, stitch->camera_buffer_width, stitch->camera_buffer_height);
		for (vx_uint32 i = 0; i < stitch->num_cameras; i++) {
			camera_params camera_par = stitch->camera_par[i];
			fprintf(fp, "lsSetCameraParams(context,%2d,&cam%02d_par);\n", i, i);
		}
		if (stitch->num_overlays > 0) {
			fprintf(fp, "lsSetOverlayConfig(context,%d,%d,VX_DF_IMAGE_RGBX,%d,%d);\n", stitch->num_overlay_rows, stitch->num_overlay_columns, stitch->overlay_buffer_width, stitch->overlay_buffer_height);
			for (vx_uint32 i = 0; i < stitch->num_overlays; i++) {
				camera_params camera_par = stitch->overlay_par[i];
				fprintf(fp, "lsSetOverlayParams(context,%2d,&ovr%02d_par\n", i, i);
			}
		}
		fprintf(fp, "lsSetRigParams(context,{%.3f,%.3f,%.3f,%.3f});\n", stitch->rig_par.yaw, stitch->rig_par.pitch, stitch->rig_par.roll, stitch->rig_par.d);
		if (stitch->loomio_output.kernelName[0]) {
			fprintf(fp, "lsSetOutputModule(context,\"%s\",\"%s\",\"%s\");\n", stitch->loomio_output.module, stitch->loomio_output.kernelName, stitch->loomio_output.kernelArguments);
		}
		else if (stitch->output_buffer_stride_in_bytes) {
			fprintf(fp, "lsSetOutputBufferStride(context,%d);\n", stitch->output_buffer_stride_in_bytes);
		}
		if (stitch->loomio_camera.kernelName[0]) {
			fprintf(fp, "lsSetCameraModule(context,\"%s\",\"%s\",\"%s\");\n", stitch->loomio_camera.module, stitch->loomio_camera.kernelName, stitch->loomio_camera.kernelArguments);
		}
		else if (stitch->camera_buffer_stride_in_bytes) {
			fprintf(fp, "lsSetCameraBufferStride(context,%d);\n", stitch->camera_buffer_stride_in_bytes);
		}
		if (stitch->loomio_overlay.kernelName[0]) {
			fprintf(fp, "lsSetOverlayModule(context,\"%s\",\"%s\",\"%s\");\n", stitch->loomio_overlay.module, stitch->loomio_overlay.kernelName, stitch->loomio_overlay.kernelArguments);
		}
		else if (stitch->overlay_buffer_stride_in_bytes) {
			fprintf(fp, "lsSetOverlayBufferStride(context,%d);\n", stitch->overlay_buffer_stride_in_bytes);
		}
		if (stitch->loomio_viewing.kernelName[0]) {
			fprintf(fp, "lsSetViewingModule(context,\"%s\",\"%s\",\"%s\");\n", stitch->loomio_viewing.module, stitch->loomio_viewing.kernelName, stitch->loomio_viewing.kernelArguments);
		}
		if(fp != stdout)
			fclose(fp);
	}
	else if (!_strnicmp(exportType, "pts", 3)) {
		FILE * fp = stdout;
		if (fileName) {
			fp = fopen(fileName, "w");
			if (!fp) {
				ls_printf("ERROR: lsExportConfiguration: unable to create: %s\n", fileName);
				if (fp != NULL)	fclose(fp);
				return VX_FAILURE;
			}
		}
		const char * camFileFormat = exportType[3] == ':' ? &exportType[4] : "CAM%02d.bmp";
		bool gotCamFileList = strstr(camFileFormat, ",") ? true : false;
		if (stitch->num_cameras < 1) {
			ls_printf("ERROR: lsExportConfiguration: %s: no cameras detected in current configuration\n", exportType);
			return VX_ERROR_NOT_SUFFICIENT;
		}
		if (stitch->rig_par.d != 0.0f) {
			ls_printf("ERROR: lsExportConfiguration: %s: non-zero d field in rig_param is not supported\n", exportType);
			return VX_ERROR_NOT_SUPPORTED;
		}
		fprintf(fp, "# ptGui project file\n\n");
		fprintf(fp, "p w%d h%d f2 v360 u0 n\"JPEG g0 q95\"\n", stitch->output_buffer_width, stitch->output_buffer_height);
		fprintf(fp, "m g0 i0\n\n");
		fprintf(fp, "# input images:\n");
		for (vx_uint32 i = 0; i < stitch->num_cameras; i++) {
			int lens_type = -1;
			if (stitch->camera_par[i].lens.lens_type == ptgui_lens_rectilinear)
				lens_type = 0;
			else if (stitch->camera_par[i].lens.lens_type == ptgui_lens_fisheye_ff)
				lens_type = 3;
			else if (stitch->camera_par[i].lens.lens_type == ptgui_lens_fisheye_circ)
				lens_type = 2;
			else {
				ls_printf("ERROR: lsExportConfiguration: %s: lens_type of camera#%d not supported\n", exportType, i);
				return VX_ERROR_NOT_SUPPORTED;
			}
			if (stitch->camera_par[i].focal.tx != 0.0f || stitch->camera_par[i].focal.ty != 0.0f || stitch->camera_par[i].focal.tz != 0.0f) {
				ls_printf("ERROR: lsExportConfiguration: %s: non-zero tx/ty/tz fields in camera#%d is not supported\n", exportType, i);
				return VX_ERROR_NOT_SUPPORTED;
			}
			// get camFileName from camFileFormat
			char camFileName[1024];
			if (gotCamFileList) {
				strncpy(camFileName, camFileFormat, sizeof(camFileName));
				char * p = strstr(camFileName, ","); if (p) *p = '\0';
				const char * s = camFileFormat;
				while (*s && *s != ',')
					s++;
				if (*s == ',')
					camFileFormat = s + 1;
			}
			else {
				snprintf(camFileName, sizeof(camFileName), camFileFormat, i);
			}
			// add camera entry
			vx_uint32 width = stitch->camera_buffer_width / stitch->num_camera_columns;
			vx_uint32 height = stitch->camera_buffer_height / stitch->num_camera_rows;
			fprintf(fp, "#-imgfile %d %d \"%s\"\n", width, height, camFileName);
			vx_float32 d = stitch->camera_par[i].lens.du0;
			vx_float32 e = stitch->camera_par[i].lens.dv0;
			if ((stitch->camera_par[i].lens.haw > 0) && (fabsf(stitch->camera_par[i].lens.haw - width) >= 0.5f)) {
				vx_int32 left = 0, right = width, top = 0, bottom = height;
				// check reserved fields for hidden crop parameters from PtGui
				if (stitch->camera_par[i].lens.lens_type <= ptgui_lens_fisheye_circ &&
					(stitch->camera_par[i].lens.reserved[3] != 0 || stitch->camera_par[i].lens.reserved[4] != 0 ||
					 stitch->camera_par[i].lens.reserved[5] != 0 || stitch->camera_par[i].lens.reserved[6] != 0))
				{
					left = (vx_int32)stitch->camera_par[i].lens.reserved[3];
					top = (vx_int32)stitch->camera_par[i].lens.reserved[4];
					right = (vx_int32)stitch->camera_par[i].lens.reserved[5];
					bottom = (vx_int32)stitch->camera_par[i].lens.reserved[6];
					d -= 0.5f*(left + right - (vx_int32)width);
					e -= 0.5f*(top + bottom - (vx_int32)height);
				}
				else
				{
					// only use crop in horizontal direction
					left = (vx_int32)(floor(d) - 0.5f * stitch->camera_par[i].lens.haw);
					right = (vx_int32)(floor(d) + 0.5f * stitch->camera_par[i].lens.haw);
					d -= floorf(d);
				}
				fprintf(fp, "o f%d y%f r%f p%f v%f a%f b%f c%f d%f e%f g0 t0 C%d,%d,%d,%d\n",
					lens_type,
					stitch->camera_par[i].focal.yaw - stitch->rig_par.yaw,
					stitch->camera_par[i].focal.roll - stitch->rig_par.roll,
					stitch->camera_par[i].focal.pitch - stitch->rig_par.pitch,
					stitch->camera_par[i].lens.hfov,
					stitch->camera_par[i].lens.k1, stitch->camera_par[i].lens.k2, stitch->camera_par[i].lens.k3,
					d, e, left, right, top, bottom);
			}
			else {
				fprintf(fp, "o f%d y%f r%f p%f v%f a%f b%f c%f d%f e%f g0 t0\n",
					lens_type,
					stitch->camera_par[i].focal.yaw - stitch->rig_par.yaw,
					stitch->camera_par[i].focal.roll - stitch->rig_par.roll,
					stitch->camera_par[i].focal.pitch - stitch->rig_par.pitch,
					stitch->camera_par[i].lens.hfov * width / (stitch->camera_par[i].lens.haw > 0 ? stitch->camera_par[i].lens.haw : (float)width),
					stitch->camera_par[i].lens.k1, stitch->camera_par[i].lens.k2, stitch->camera_par[i].lens.k3,
					d, e);
			}
		}
		if(fp != stdout)
			fclose(fp);
	}
	else if (!_stricmp(exportType, "gdf")) {
		ERROR_CHECK_STATUS_(IsValidContextAndInitialized(stitch));
		if (_stricmp(fileName + strlen(fileName) - 4, ".gdf")) {
			ls_printf("ERROR: lsExportConfiguration: gdf: requires fileName extension to be .gdf\n");
			return VX_ERROR_INVALID_PARAMETERS;
		}
		char fileNamePrefixForTables[1024] = { 0 };
		std::string tableName = fileName;
		tableName.erase(tableName.find_last_of("."), std::string::npos);
		strncpy(fileNamePrefixForTables, tableName.c_str(), strlen(fileNamePrefixForTables));
		FILE * fp = fopen(fileName, "w");
		if (!fp) {
			ls_printf("ERROR: lsExportConfiguration: unable to create: %s\n", fileName);
			if (fp != NULL)	fclose(fp);
			return VX_FAILURE;
		}
		vx_node nodeObjList[] = {
			stitch->InputColorConvertNode, stitch->SimpleStitchRemapNode, stitch->OutputColorConvertNode,
			stitch->WarpNode, stitch->ExpcompComputeGainNode, stitch->ExpcompSolveGainNode, stitch->ExpcompApplyGainNode, stitch->MergeNode,
			stitch->SobelNode, stitch->MagnitudeNode, stitch->PhaseNode, stitch->ConvertDepthNode, 
			stitch->SeamfindStep1Node, stitch->SeamfindStep2Node, stitch->SeamfindStep3Node, stitch->SeamfindStep4Node, stitch->SeamfindStep5Node,
			stitch->nodeOverlayRemap, stitch->nodeOverlayBlend,
			stitch->nodeLoomIoCamera, stitch->nodeLoomIoOverlay, stitch->nodeLoomIoOutput, stitch->nodeLoomIoViewing,
		};
		const char * kernelNameList[] = {
			"com.amd.loomsl.color_convert", "org.khronos.openvx.remap", "com.amd.loomsl.color_convert",
			"com.amd.loomsl.warp", "com.amd.loomsl.expcomp_compute_gainmatrix", "com.amd.loomsl.expcomp_solvegains", "com.amd.loomsl.expcomp_applygains", "com.amd.loomsl.merge",
			"org.khronos.openvx.sobel_3x3", "org.khronos.openvx.magnitude", "org.khronos.openvx.phase", "org.khronos.openvx.convert_depth",
			"com.amd.loomsl.seamfind_scene_detect", "com.amd.loomsl.seamfind_cost_generate", "com.amd.loomsl.seamfind_cost_accumulate", "com.amd.loomsl.seamfind_path_trace", "com.amd.loomsl.seamfind_set_weights",
			"org.khronos.openvx.remap", "com.amd.loomsl.alpha_blend",
			stitch->loomio_camera.kernelName, stitch->loomio_overlay.kernelName, stitch->loomio_output.kernelName, stitch->loomio_viewing.kernelName,
		};
		std::map<vx_node, std::string> nodeMap;
		for (vx_size i = 0; i < dimof(nodeObjList); i++) {
			if (nodeObjList[i]) {
				nodeMap[nodeObjList[i]] = kernelNameList[i];
			}
		}
		if (stitch->MULTIBAND_BLEND){
			for (int i = 1; i < stitch->num_bands; i++) {
				nodeMap[stitch->pStitchMultiband[i].WeightHSGNode] = "com.amd.loomsl.half_scale_gaussian";
				nodeMap[stitch->pStitchMultiband[i].SourceHSGNode] = "com.amd.loomsl.half_scale_gaussian";
				nodeMap[stitch->pStitchMultiband[i].UpscaleSubtractNode] = "com.amd.loomsl.upscale_gaussian_subtract";
			}
			int i = stitch->num_bands - 1;
			nodeMap[stitch->pStitchMultiband[i].BlendNode] = "com.amd.loomsl.multiband_blend";
			--i;
			for (; i > 0; --i){
				nodeMap[stitch->pStitchMultiband[i].UpscaleAddNode] = "com.amd.loomsl.upscale_gaussian_add";
			}
			nodeMap[stitch->pStitchMultiband[0].UpscaleAddNode] = "com.amd.loomsl.laplacian_reconstruct";
		}
		std::map<vx_reference, std::string> refSuffixList;
		std::map<vx_reference, bool> refIsForCpuUseOnly;
		for (auto it = nodeMap.begin(); it != nodeMap.end(); it++) {
			vx_node node = it->first;
			if (node) {
				vx_uint32 paramCount;
				ERROR_CHECK_STATUS_(vxQueryNode(node, VX_NODE_PARAMETERS, &paramCount, sizeof(paramCount)));
				for (vx_uint32 paramIndex = 0; paramIndex < paramCount; paramIndex++) {
					vx_reference ref = avxGetNodeParamRef(node, paramIndex);
					if (vxGetStatus(ref) == VX_SUCCESS) {
						if (refSuffixList.find(ref) == refSuffixList.end()) {
							bool isIntermediateTmpData = false, isForCpuUseOnly = false;
							const char * fileNameSuffix = GetFileNameSuffix(stitch, ref, isIntermediateTmpData, isForCpuUseOnly);
							if (fileNameSuffix && !isIntermediateTmpData) {
								refSuffixList[ref] = fileNameSuffix;
								if (isForCpuUseOnly) {
									refIsForCpuUseOnly[ref] = true;
								}
							}
						}
					}
				}
			}
		}
		fprintf(fp, "import vx_loomsl\n");
		std::map<vx_reference, std::string> refNameList;
		if (stitch->stitching_mode == stitching_mode_normal) {
			fprintf(fp, "type WarpValidPixelEntryType userstruct:%d\n", (int)sizeof(StitchValidPixelEntry));
			fprintf(fp, "type WarpRemapEntryType userstruct:%d\n", (int)sizeof(StitchWarpRemapEntry));
			fprintf(fp, "data warpValidPixelTable = array:WarpValidPixelEntryType,%d\n", (int)stitch->table_sizes.warpTableSize);
			fprintf(fp, "data warpRemapTable = array:WarpRemapEntryType,%d\n", (int)stitch->table_sizes.warpTableSize);
			fprintf(fp, "data RGBY1 = image:%d,%d,RGBA\n", stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height * stitch->num_cameras);
			fprintf(fp, "data weight_image = image:%d,%d,U008\n", stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height * stitch->num_cameras);
			fprintf(fp, "data cam_id_image = image:%d,%d,U008\n", (stitch->output_rgb_buffer_width / 8), stitch->output_rgb_buffer_height);
			fprintf(fp, "data group1_image = image:%d,%d,U016\n", (stitch->output_rgb_buffer_width / 8), stitch->output_rgb_buffer_height);
			fprintf(fp, "data group2_image = image:%d,%d,U016\n", (stitch->output_rgb_buffer_width / 8), stitch->output_rgb_buffer_height);
			refNameList[(vx_reference)stitch->ValidPixelEntry] = "warpValidPixelTable";
			refNameList[(vx_reference)stitch->WarpRemapEntry] = "warpRemapTable";
			refNameList[(vx_reference)stitch->RGBY1] = "RGBY1";
			refNameList[(vx_reference)stitch->weight_image] = "weight_image";
			refNameList[(vx_reference)stitch->cam_id_image] = "cam_id_image";
			refNameList[(vx_reference)stitch->group1_image] = "group1_image";
			refNameList[(vx_reference)stitch->group2_image] = "group2_image";
			if (stitch->EXPO_COMP) {
				fprintf(fp, "type ExpCompValidEntryType userstruct:%d\n", (int)sizeof(StitchOverlapPixelEntry));
				fprintf(fp, "type ExpCompCalcEntryType userstruct:%d\n", (int)sizeof(StitchExpCompCalcEntry));
				fprintf(fp, "data expCompValidTable = array:ExpCompValidEntryType,%d\n", (int)stitch->table_sizes.expCompValidTableSize);
				if (stitch->EXPO_COMP < 3) fprintf(fp, "data expCompCalcTable = array:ExpCompCalcEntryType,%d\n", (int)stitch->table_sizes.expCompOverlapTableSize);
				fprintf(fp, "data expCompGain = array:VX_TYPE_FLOAT32,%d\n", (int)stitch->num_cameras);
				fprintf(fp, "data expCompAMat = matrix:VX_TYPE_INT32,%d,%d\n", stitch->num_cameras, stitch->num_cameras);
				fprintf(fp, "data expCompCountMat = matrix:VX_TYPE_INT32,%d,%d\n", stitch->num_cameras, stitch->num_cameras);
				fprintf(fp, "data RGBY2 = image:%d,%d,RGBA\n", stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height * stitch->num_cameras);
				refNameList[(vx_reference)stitch->valid_array] = "expCompValidTable";
				if (stitch->EXPO_COMP < 3) refNameList[(vx_reference)stitch->OverlapPixelEntry] = "expCompCalcTable";
				refNameList[(vx_reference)stitch->gain_array] = "expCompGain";
				refNameList[(vx_reference)stitch->A_matrix] = "expCompAMat";
				refNameList[(vx_reference)stitch->overlap_matrix] = "expCompCountMat";
				refNameList[(vx_reference)stitch->RGBY2] = "RGBY2";
			}
			if (stitch->SEAM_FIND) {
				fprintf(fp, "type SeamFindValidEntryType userstruct:%d\n", (int)sizeof(StitchSeamFindValidEntry));
				fprintf(fp, "type SeamFindWeightEntryType userstruct:%d\n", (int)sizeof(StitchSeamFindWeightEntry));
				fprintf(fp, "type SeamFindAccumEntryType userstruct:%d\n", (int)sizeof(StitchSeamFindAccumEntry));
				fprintf(fp, "type SeamFindPreferenceType userstruct:%d\n", (int)sizeof(StitchSeamFindPreference));
				fprintf(fp, "type SeamFindInformationType userstruct:%d\n", (int)sizeof(StitchSeamFindInformation));
				fprintf(fp, "type SeamFindPathEntryType userstruct:%d\n", (int)sizeof(StitchSeamFindPathEntry));
				fprintf(fp, "data seamFindValid = array:SeamFindValidEntryType,%d\n", (int)stitch->table_sizes.seamFindValidTableSize);
				fprintf(fp, "data seamFindWeight = array:SeamFindWeightEntryType,%d\n", (int)stitch->table_sizes.seamFindWeightTableSize);
				fprintf(fp, "data seamFindAccum = array:SeamFindAccumEntryType,%d\n", (int)stitch->table_sizes.seamFindAccumTableSize);
				fprintf(fp, "data seamFindPref = array:SeamFindPreferenceType,%d\n", (int)stitch->table_sizes.seamFindPrefInfoTableSize);
				fprintf(fp, "data seamFindInfo = array:SeamFindInformationType,%d\n", (int)stitch->table_sizes.seamFindPrefInfoTableSize);
				fprintf(fp, "data seamFindPath = array:SeamFindPathEntryType,%d\n", (int)stitch->table_sizes.seamFindPathTableSize);
				fprintf(fp, "data warpLuma = virtual-image:%d,%d,U008\n", stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height * stitch->num_cameras);
				refNameList[(vx_reference)stitch->seamfind_valid_array] = "seamFindValid";
				refNameList[(vx_reference)stitch->seamfind_weight_array] = "seamFindWeight";
				refNameList[(vx_reference)stitch->seamfind_accum_array] = "seamFindAccum";
				refNameList[(vx_reference)stitch->seamfind_pref_array] = "seamFindPref";
				refNameList[(vx_reference)stitch->seamfind_info_array] = "seamFindInfo";
				refNameList[(vx_reference)stitch->seamfind_path_array] = "seamFindPath";
				refNameList[(vx_reference)stitch->warp_luma_image] = "warpLuma";
				if (!stitch->SEAM_COST_SELECT) {
					fprintf(fp, "data seamFindSobelX = virtual-image:%d,%d,S016\n", stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height * stitch->num_cameras);
					fprintf(fp, "data seamFindSobelY = virtual-image:%d,%d,S016\n", stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height * stitch->num_cameras);
					fprintf(fp, "data seamFindMagS16 = virtual-image:%d,%d,S016\n", stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height * stitch->num_cameras);
					refNameList[(vx_reference)stitch->sobelx_image] = "seamFindSobelX";
					refNameList[(vx_reference)stitch->sobely_image] = "seamFindSobelY";
					refNameList[(vx_reference)stitch->sobel_magnitude_s16_image] = "seamFindMagS16";
				}
				fprintf(fp, "data seamFindMag = virtual-image:%d,%d,U008\n", stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height * stitch->num_cameras);
				fprintf(fp, "data seamFindPhase = virtual-image:%d,%d,U008\n", stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height * stitch->num_cameras);
				fprintf(fp, "data seamFindWeightImage = image:%d,%d,U008\n", stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height * stitch->num_cameras);
				fprintf(fp, "data seamFindCurFrame = scalar:VX_TYPE_UINT32,%d\n", stitch->current_frame_value);
				refNameList[(vx_reference)stitch->sobel_magnitude_image] = "seamFindMag";
				refNameList[(vx_reference)stitch->sobel_phase_image] = "seamFindPhase";
				refNameList[(vx_reference)stitch->seamfind_weight_image] = "seamFindWeightImage";
				refNameList[(vx_reference)stitch->current_frame] = "seamFindCurFrame";
				if (stitch->SEAM_REFRESH) {
					fprintf(fp, "type SeamFindSceneEntryType userstruct:%d\n", (int)sizeof(StitchSeamFindSceneEntry));
					fprintf(fp, "data seamFindSceneTable = array:SeamFindSceneEntryType,%d\n", (int)stitch->table_sizes.seamFindPrefInfoTableSize);
					fprintf(fp, "data seamFindSceneThreshold = scalar:VX_TYPE_UINT32,%d\n", stitch->scene_threshold_value);
					refNameList[(vx_reference)stitch->seamfind_scene_array] = "seamFindSceneTable";
					refNameList[(vx_reference)stitch->scene_threshold] = "seamFindSceneThreshold";
				}
				if (stitch->SEAM_COST_SELECT) {
					vx_uint32 cost_enable = 1;
					fprintf(fp, "data seamFindCost = scalar:VX_TYPE_UINT32,%d\n", cost_enable);
					refNameList[(vx_reference)stitch->seam_cost_enable] = "seamFindCost";
				}
			}
			if (stitch->MULTIBAND_BLEND) {
				fprintf(fp, "type BlendValidEntryType userstruct:%d\n", (int)sizeof(StitchBlendValidEntry));
				fprintf(fp, "data blendValidTable = array:BlendValidEntryType,%d\n", (int)stitch->table_sizes.blendOffsetTableSize);
				refNameList[(vx_reference)stitch->blend_offsets] = "blendValidTable";
			}
			if (stitch->SEAM_FIND || stitch->EXPO_COMP) {
				fprintf(fp, "data validMaskImage = image:%d,%d,U008\n", stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height * stitch->num_cameras);
				refNameList[(vx_reference)stitch->valid_mask_image] = "validMaskImage";
			}
		}
		fprintf(fp, "data Img_input = image:%d,%d,%4.4s\n", stitch->camera_buffer_width, stitch->camera_buffer_height, (const char *)&stitch->camera_buffer_format);
		fprintf(fp, "data Img_output = image:%d,%d,%4.4s\n", stitch->output_buffer_width, stitch->output_buffer_height, (const char *)&stitch->output_buffer_format);
		refNameList[(vx_reference)stitch->Img_input] = "Img_input";
		refNameList[(vx_reference)stitch->Img_output] = "Img_output";
		if (stitch->overlay_remap) {
			fprintf(fp, "data Img_overlay = image:%d,%d,RGBA\n", stitch->overlay_buffer_width, stitch->overlay_buffer_height);
			fprintf(fp, "data Img_overlay_rgba = virtual-image:%d,%d,RGBA\n", stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height);
			fprintf(fp, "data Img_overlay_rgb = virtual-image:%d,%d,RGB2\n", stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height);
			fprintf(fp, "data overlay_remap = remap:%d,%d,%d,%d\n", stitch->overlay_buffer_width, stitch->overlay_buffer_height, stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height);
			refNameList[(vx_reference)stitch->Img_overlay] = "Img_overlay";
			refNameList[(vx_reference)stitch->Img_overlay_rgba] = "Img_overlay_rgba";
			refNameList[(vx_reference)stitch->Img_overlay_rgb] = "Img_overlay_rgb";
			refNameList[(vx_reference)stitch->overlay_remap] = "overlay_remap";
		}
		if (stitch->camera_remap) {
			fprintf(fp, "data camera_remap = remap:%d,%d,%d,%d\n", stitch->camera_rgb_buffer_width, stitch->camera_rgb_buffer_height, stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height);
			refNameList[(vx_reference)stitch->camera_remap] = "camera_remap";
		}
		int genImageCount = 0, genScalarCount = 0;
		for (auto it = nodeMap.begin(); it != nodeMap.end(); it++) {
			vx_node node = it->first;
			if (node) {
				vx_uint32 paramCount;
				ERROR_CHECK_STATUS_(vxQueryNode(node, VX_NODE_PARAMETERS, &paramCount, sizeof(paramCount)));
				for (vx_uint32 paramIndex = 0; paramIndex < paramCount; paramIndex++) {
					vx_reference ref = avxGetNodeParamRef(node, paramIndex);
					if (vxGetStatus(ref) == VX_SUCCESS) {
						if (refNameList.find(ref) == refNameList.end()) {
							vx_enum type;
							ERROR_CHECK_STATUS_(vxQueryReference(ref, VX_REFERENCE_TYPE, &type, sizeof(type)));
							if (type == VX_TYPE_IMAGE) {
								vx_uint32 width, height; vx_df_image format;
								ERROR_CHECK_STATUS_(vxQueryImage((vx_image)ref, VX_IMAGE_WIDTH, &width, sizeof(width)));
								ERROR_CHECK_STATUS_(vxQueryImage((vx_image)ref, VX_IMAGE_HEIGHT, &height, sizeof(height)));
								ERROR_CHECK_STATUS_(vxQueryImage((vx_image)ref, VX_IMAGE_FORMAT, &format, sizeof(format)));
								char name[64]; snprintf(name, sizeof(name), "img_%02d", genImageCount++);
								fprintf(fp, "data %s = image:%d,%d,%4.4s\n", name, width, height, (const char *)&format);
								refNameList[ref] = name;
							}
							else if (type == VX_TYPE_SCALAR) {
								vx_enum data_type; char value[1024] = { 0 };
								ERROR_CHECK_STATUS_(vxQueryScalar((vx_scalar)ref, VX_SCALAR_TYPE, &data_type, sizeof(data_type)));
								ERROR_CHECK_STATUS_(vxReadScalarValue((vx_scalar)ref, value));
								char name[64]; snprintf(name, sizeof(name), "scalar_%02d", genScalarCount++);
								if (data_type == VX_TYPE_UINT32) fprintf(fp, "data %s = scalar:VX_TYPE_UINT32,%u\n", name, *(vx_uint32 *)value);
								else if (data_type == VX_TYPE_ENUM) fprintf(fp, "data %s = scalar:VX_TYPE_ENUM,0x%08x\n", name, *(vx_enum *)value);
								else if (data_type == VX_TYPE_INT32) fprintf(fp, "data %s = scalar:VX_TYPE_INT32,%d\n", name, *(vx_int32 *)value);
								else if (data_type == VX_TYPE_FLOAT32) fprintf(fp, "data %s = scalar:VX_TYPE_FLOAT32,%f\n", name, *(vx_float32 *)value);
								else if (data_type == VX_TYPE_STRING_AMD) fprintf(fp, "data %s = scalar:VX_TYPE_STRING_AMD,\"%s\"\n", name, value);
								else { ls_printf("ERROR: lsExportConfiguration: gdf: unsupported scalar data type found: %d\n", data_type); return VX_FAILURE; }
								refNameList[ref] = name;
							}
							else {
								ls_printf("ERROR: lsExportConfiguration: gdf: unsupported object type detected as arg#%d of node: %s\n", paramIndex, it->second.c_str());
								return VX_FAILURE;
							}
						}
					}
				}
			}
		}
		for (auto it = refSuffixList.begin(); it != refSuffixList.end(); it++) {
			if (refNameList.find(it->first) == refNameList.end()) {
				vx_enum type; char name[64] = { 0 }; const char * suffix = it->second.c_str();
				ERROR_CHECK_STATUS_(vxQueryReference(it->first, VX_REFERENCE_TYPE, &type, sizeof(type)));
				if (*suffix == '|') strcpy(name, suffix + 1);
				else {
					for (int i = 0; suffix[i]; i++) {
						name[i] = (suffix[i] == '.' || suffix[i] == '-') ? '_' : suffix[i];
					}
				}
				refNameList[it->first] = name;
				if (type == VX_TYPE_IMAGE) {
					vx_uint32 width, height; vx_df_image format;
					ERROR_CHECK_STATUS_(vxQueryImage((vx_image)it->first, VX_IMAGE_WIDTH, &width, sizeof(width)));
					ERROR_CHECK_STATUS_(vxQueryImage((vx_image)it->first, VX_IMAGE_HEIGHT, &height, sizeof(height)));
					ERROR_CHECK_STATUS_(vxQueryImage((vx_image)it->first, VX_IMAGE_FORMAT, &format, sizeof(format)));
					fprintf(fp, "data %s = image:%d,%d,%4.4s\n", name, width, height, (const char *)&format);
				}
				else if (type == VX_TYPE_SCALAR) {
					vx_enum data_type; char value[1024] = { 0 };
					ERROR_CHECK_STATUS_(vxQueryScalar((vx_scalar)it->first, VX_SCALAR_TYPE, &data_type, sizeof(data_type)));
					ERROR_CHECK_STATUS_(vxReadScalarValue((vx_scalar)it->first, value));
					if (data_type == VX_TYPE_UINT32) fprintf(fp, "data %s = scalar:VX_TYPE_UINT32,%u\n", name, *(vx_uint32 *)value);
					else if (data_type == VX_TYPE_ENUM) fprintf(fp, "data %s = scalar:VX_TYPE_ENUM,0x%08x\n", name, *(vx_enum *)value);
					else if (data_type == VX_TYPE_INT32) fprintf(fp, "data %s = scalar:VX_TYPE_INT32,%d\n", name, *(vx_int32 *)value);
					else if (data_type == VX_TYPE_FLOAT32) fprintf(fp, "data %s = scalar:VX_TYPE_FLOAT32,%f\n", name, *(vx_float32 *)value);
					else if (data_type == VX_TYPE_STRING_AMD) fprintf(fp, "data %s = scalar:VX_TYPE_STRING_AMD,\"%s\"\n", name, value);
					else { ls_printf("ERROR: lsExportConfiguration: gdf: unsupported scalar data type found: %d\n", data_type); return VX_FAILURE; }
				}
				else {
					ls_printf("ERROR: lsExportConfiguration: gdf: internal error: one of the object type is not supported: %d\n", type);
					return VX_FAILURE;
				}
			}
		}
		fprintf(fp, "\n");
		for (auto it = nodeMap.begin(); it != nodeMap.end(); it++) {
			vx_node node = it->first;
			if (node) {
				vx_uint32 paramCount;
				ERROR_CHECK_STATUS_(vxQueryNode(node, VX_NODE_PARAMETERS, &paramCount, sizeof(paramCount)));
				fprintf(fp, "node %s", it->second.c_str());
				for (vx_uint32 paramIndex = 0; paramIndex < paramCount; paramIndex++) {
					vx_reference ref = avxGetNodeParamRef(node, paramIndex);
					if (vxGetStatus(ref) == VX_SUCCESS) {
						if (refNameList.find(ref) == refNameList.end()) {
							vx_enum type;
							ERROR_CHECK_STATUS_(vxQueryReference(ref, VX_REFERENCE_TYPE, &type, sizeof(type)));
							if (type == VX_TYPE_SCALAR) {
								char value[1024];
								ERROR_CHECK_STATUS_(vxReadScalarValue((vx_scalar)ref, value));
								fprintf(fp, " !%d", *(int *)value);
							}
							else {
								ls_printf("ERROR: lsExportConfiguration: gdf: unsupported data object as arg#%d of node object: %s\n", paramIndex, it->second.c_str());
								return VX_FAILURE;
							}
						}
						else {
							fprintf(fp, " %s", refNameList[ref].c_str());
						}
					}
					else {
						fprintf(fp, " NULL");
					}
				}
				fprintf(fp, "\n");
			}
		}
		fprintf(fp, "\n");
		for (auto it = refSuffixList.begin(); it != refSuffixList.end(); it++) {
			fprintf(fp, "init %s %s-%s\n", refNameList[it->first].c_str(), fileNamePrefixForTables, it->second.c_str());
		}
		fprintf(fp, "read %s %s-camera-input.raw\n", refNameList[(vx_reference)stitch->Img_input].c_str(), fileNamePrefixForTables);
		if (stitch->Img_overlay) {
			fprintf(fp, "read %s %s-overlay-input.raw\n", refNameList[(vx_reference)stitch->Img_overlay].c_str(), fileNamePrefixForTables);
		}
		fprintf(fp, "write %s %s-stitch-output.raw\n", refNameList[(vx_reference)stitch->Img_output].c_str(), fileNamePrefixForTables);
		fprintf(fp, "\n");
		for (auto it = refSuffixList.begin(); it != refSuffixList.end(); it++) {
			if (refIsForCpuUseOnly.find(it->first) == refIsForCpuUseOnly.end() || !refIsForCpuUseOnly[it->first]) {
				fprintf(fp, "directive %s VX_DIRECTIVE_AMD_COPY_TO_OPENCL\n", refNameList[it->first].c_str());
			}
		}
		fclose(fp);
		ls_printf("OK: lsExportConfiguration: created %s\n", fileName);
		return DumpInternalTables(stitch, fileNamePrefixForTables, false);
	}
	else if (!_stricmp(exportType, "data")) {
		ERROR_CHECK_STATUS_(IsValidContextAndInitialized(stitch));
		return DumpInternalTables(stitch, fileName, true);
	}
	else {
		ls_printf("ERROR: lsExportConfiguration: unsupported exportType: %s\n", exportType);
		return VX_ERROR_NOT_SUPPORTED;
	}
	return VX_SUCCESS;
}
LIVE_STITCH_API_ENTRY vx_status VX_API_CALL lsImportConfiguration(ls_context stitch, const char * importType, const char * fileName)
{
	ERROR_CHECK_STATUS_(IsValidContext(stitch));
	if (!_stricmp(importType, "pts")) {
		if (stitch->num_cameras < 1) {
			ls_printf("ERROR: lsImportConfiguration: %s: needs more than one camera in the configuration\n", importType);
			return VX_ERROR_NOT_SUFFICIENT;
		}
		vx_uint32 camIndex = 0;
		camera_lens_type lens_type = ptgui_lens_fisheye_ff;
		vx_float32 a = 0, b = 0, c = 0, d = 0, e = 0, yaw = 0, pitch = 0, roll = 0, hfov = 0;
		vx_uint32 width = stitch->camera_buffer_width / stitch->num_camera_columns;
		vx_uint32 height = stitch->camera_buffer_height / stitch->num_camera_rows;
		bool isDummy = false;
		FILE * fp = fopen(fileName, "rb");
		if (!fp) {
			ls_printf("ERROR: lsImportConfiguration: unable to open: %s\n", fileName);
			if (fp != NULL)	fclose(fp);
			return VX_FAILURE;
		}
		fseek(fp, 0L, SEEK_END); long fileSize = ftell(fp); fseek(fp, 0L, SEEK_SET);
		char * textBuf = new char[fileSize + 1];
		ERROR_CHECK_FREAD_(fread(textBuf, 1, fileSize, fp),fileSize);
		fclose(fp);
		textBuf[fileSize] = 0;
		for (const char * text = textBuf; *text; ) {
			if (!strncmp(text, "#-dummyimage", 12)) {
				isDummy = true;
			}
			else if (*text == 'o') {
				bool cropSpecified = false;
				vx_int32 left = 0, top = 0, right = width, bottom = height;
				if (camIndex >= stitch->num_cameras) {
					ls_printf("ERROR: lsImportConfiguration: %s: PTS has more cameras than current configuration\n", importType);
					delete[] textBuf;
					return VX_ERROR_NOT_SUPPORTED;
				}
				while (*text && *text != '\n') {
					// skip till whitespace
					while (*text && *text != '\n' && *text != ' ' && *text != '\t') text++;
					while (*text && *text != '\n' && (*text == ' ' || *text == '\t')) text++;
					// process fields
					if (*text == 'f') {
						if (text[1] == '0') lens_type = ptgui_lens_rectilinear;
						else if (text[1] == '2') lens_type = ptgui_lens_fisheye_circ;
						else if (text[1] == '3') lens_type = ptgui_lens_fisheye_ff;
						else {
							ls_printf("ERROR: lsImportConfiguration: %s: lens_type f%c not supported\n", importType, text[1]);
							delete[] textBuf;
							return VX_ERROR_NOT_SUPPORTED;
						}
					}
					else if (*text == 'y') yaw = (float)atof(&text[1]);
					else if (*text == 'p') pitch = (float)atof(&text[1]);
					else if (*text == 'r') roll = (float)atof(&text[1]);
					else if (*text == 'v' && text[1] != '=') hfov = (float)atof(&text[1]);
					else if (*text == 'a' && text[1] != '=') a = (float)atof(&text[1]);
					else if (*text == 'b' && text[1] != '=') b = (float)atof(&text[1]);
					else if (*text == 'c' && text[1] != '=') c = (float)atof(&text[1]);
					else if (*text == 'd' && text[1] != '=') d = (float)atof(&text[1]);
					else if (*text == 'e' && text[1] != '=') e = (float)atof(&text[1]);
					else if (*text == 'C') {
						sscanf(&text[1], "%d,%d,%d,%d", &left, &right, &top, &bottom);
						cropSpecified = true;
					}
				}
				if (!isDummy) {
					camera_params * par = &stitch->camera_par[camIndex++];
					par->focal.yaw = yaw;
					par->focal.pitch = pitch;
					par->focal.roll = roll;
					par->focal.tx = 0.0f;
					par->focal.ty = 0.0f;
					par->focal.tz = 0.0f;
					par->lens.lens_type = lens_type;
					par->lens.hfov = hfov;
					par->lens.k1 = a;
					par->lens.k2 = b;
					par->lens.k3 = c;
					par->lens.du0 = d;
					par->lens.dv0 = e;
					par->lens.haw = (float)width;
					if (cropSpecified) {
						par->lens.haw = (float)(right - left);
						par->lens.du0 += 0.5f*(left + right - (vx_int32)width);
						par->lens.dv0 += 0.5f*(top + bottom - (vx_int32)height);
						par->lens.r_crop = par->lens.haw * 0.5f;
						// save PtGui crop values in reserved fields
						par->lens.reserved[3] = (float)left;
						par->lens.reserved[4] = (float)top;
						par->lens.reserved[5] = (float)right;
						par->lens.reserved[6] = (float)bottom;
					}
				}
				isDummy = false;
			}
			// skip till end-of-line
			while (*text && *text != '\n')
				text++;
			if (*text == '\n')
				text++;
		}
		delete[] textBuf;
		if (camIndex != stitch->num_cameras) {
			ls_printf("ERROR: lsImportConfiguration: %s: could not import for all %d cameras (found %d)\n", importType, stitch->num_cameras, camIndex);
			return VX_ERROR_NOT_SUFFICIENT;
		}
	}
	else {
		ls_printf("ERROR: lsImportConfiguration: unsupported importType: %s\n", importType);
		return VX_ERROR_NOT_SUPPORTED;
	}
	return VX_SUCCESS;
}

LIVE_STITCH_API_ENTRY vx_status VX_API_CALL lsSetExpCompGains(ls_context stitch, size_t num_entries, vx_float32 * gains)
{
	ERROR_CHECK_STATUS_(IsValidContextAndInitialized(stitch));
	if (!stitch->EXPO_COMP || !stitch->gain_array)
		return VX_ERROR_NOT_SUPPORTED;
	vx_size count;
	ERROR_CHECK_STATUS_(vxQueryArray(stitch->gain_array, VX_ARRAY_NUMITEMS, &count, sizeof(count)));
	if (num_entries != count) {
		ls_printf("ERROR: lsSetExpCompGains: expects num_entries to be %d: got %d\n", (vx_uint32)count, (vx_uint32)num_entries);
		return VX_ERROR_INVALID_PARAMETERS;
	}
	ERROR_CHECK_STATUS_(vxCopyArrayRange(stitch->gain_array, 0, num_entries, sizeof(vx_float32), gains, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
	return VX_SUCCESS;
}

LIVE_STITCH_API_ENTRY vx_status VX_API_CALL lsGetExpCompGains(ls_context stitch, size_t num_entries, vx_float32 * gains)
{
	ERROR_CHECK_STATUS_(IsValidContextAndInitialized(stitch));
	if (!stitch->EXPO_COMP || !stitch->gain_array) 
		return VX_ERROR_NOT_SUPPORTED;
	vx_size count;
	ERROR_CHECK_STATUS_(vxQueryArray(stitch->gain_array, VX_ARRAY_NUMITEMS, &count, sizeof(count)));
	if (num_entries != count) {
		ls_printf("ERROR: lsGetExpCompGains: expects num_entries to be %d: got %d\n", (vx_uint32)count, (vx_uint32)num_entries);
		return VX_ERROR_INVALID_PARAMETERS;
	}
	ERROR_CHECK_STATUS_(vxCopyArrayRange(stitch->gain_array, 0, num_entries, sizeof(vx_float32), gains, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
	return VX_SUCCESS;
}

LIVE_STITCH_API_ENTRY vx_status VX_API_CALL lsSetBlendWeights(ls_context stitch, vx_uint8 * weights, size_t size)
{
	ERROR_CHECK_STATUS_(IsValidContextAndInitialized(stitch));
	if (stitch->SEAM_FIND || !stitch->weight_image)
		return VX_ERROR_NOT_SUPPORTED;
	if (size != (size_t)(stitch->output_rgb_buffer_width * stitch->output_rgb_buffer_height * stitch->num_cameras))
		return VX_ERROR_INVALID_PARAMETERS;

	// copy weight image
	vx_rectangle_t rect = { 0, 0, stitch->output_rgb_buffer_width, stitch->output_rgb_buffer_height * stitch->num_cameras };
	vx_imagepatch_addressing_t addr;
	addr.dim_x = rect.end_x;
	addr.dim_y = rect.end_y;
	addr.stride_x = 1;
	addr.stride_y = rect.end_x;
	ERROR_CHECK_STATUS_(vxCopyImagePatch(stitch->weight_image, &rect, 0, &addr, weights, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));

	return VX_SUCCESS;
}
