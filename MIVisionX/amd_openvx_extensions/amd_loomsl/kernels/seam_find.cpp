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
#include "seam_find.h"

//developer settings
#define GET_TIMING     0
#define SHOW_MESSAGES  0

#if _WIN32
#include <windows.h>
#undef min
#undef max
#endif

#if SHOW_MESSAGES
#define PRINTF       printf
#else
#define PRINTF(...)
#endif

#if GET_TIMING
int64_t stitchGetClockCounter()
{
#if _WIN32
	LARGE_INTEGER v;
	QueryPerformanceCounter(&v);
	return v.QuadPart;
#else
	return chrono::high_resolution_clock::now().time_since_epoch().count();
#endif
}

int64_t stitchGetClockFrequency()
{
#if _WIN32
	LARGE_INTEGER v;
	QueryPerformanceFrequency(&v);
	return v.QuadPart;
#else
	return chrono::high_resolution_clock::period::den / chrono::high_resolution_clock::period::num;
#endif
}
#endif //GET_TIMING

/***********************************************************************************************************************************
												Seam Find CPU Model
***********************************************************************************************************************************/
//! \brief The input validator callback.
static vx_status VX_CALLBACK seamfind_model_input_validator(vx_node node, vx_uint32 index)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	// get reference for parameter at specified index
	vx_reference ref = avxGetNodeParamRef(node, index);
	ERROR_CHECK_OBJECT(ref);
	// validate each parameter
	if (index == 0)
	{//->Number of Camera
		vx_enum type = 0;	vx_uint32 value = 0;
		ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)ref, VX_SCALAR_ATTRIBUTE_TYPE, &type, sizeof(type)));
		ERROR_CHECK_STATUS(vxReadScalarValue((vx_scalar)ref, &value));
		if (value > 0 && type == VX_TYPE_UINT32)
			status = VX_SUCCESS;
		ERROR_CHECK_STATUS(vxReleaseScalar((vx_scalar*)&ref));
	}
	else if (index == 1)
	{ // array object of RECTANGLE type
		vx_enum itemtype = VX_TYPE_INVALID;
		vx_size capacity = 0;
		ERROR_CHECK_STATUS(vxQueryArray((vx_array)ref, VX_ARRAY_ATTRIBUTE_ITEMTYPE, &itemtype, sizeof(itemtype)));
		ERROR_CHECK_STATUS(vxQueryArray((vx_array)ref, VX_ARRAY_ATTRIBUTE_CAPACITY, &capacity, sizeof(capacity)));

		if (itemtype != VX_TYPE_RECTANGLE) {
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: seam_find array type should be an rectangle\n");
		}
		else if (capacity == 0) {
			status = VX_ERROR_INVALID_DIMENSION;
			vxAddLogEntry((vx_reference)node, status, "ERROR: seam_find array capacity should be positive\n");
		}
		else {
			status = VX_SUCCESS;
		}
		ERROR_CHECK_STATUS(vxReleaseArray((vx_array *)&ref));
	}
	else if (index == 2)
	{ // Overlap Matrix
		vx_enum type = VX_TYPE_INVALID;
		ERROR_CHECK_STATUS(vxQueryMatrix((vx_matrix)ref, VX_MATRIX_ATTRIBUTE_TYPE, &type, sizeof(type)));
		if (type == VX_TYPE_INT32)
			status = VX_SUCCESS;
	}
	else if (index == 3)
	{ // Image object	
		vx_int32 width_img = 0, height_img = 0;
		vx_df_image format = VX_DF_IMAGE_VIRT;
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_WIDTH, &width_img, sizeof(width_img)));
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_HEIGHT, &height_img, sizeof(height_img)));
		if (format == VX_DF_IMAGE_S16 || format == VX_DF_IMAGE_U8)
			status = VX_SUCCESS;
		ERROR_CHECK_STATUS(vxReleaseImage((vx_image*)&ref));
	}
	else if (index == 4 || index == 5)
	{ // Image object	
		vx_int32 width_img = 0, height_img = 0;
		vx_df_image format = VX_DF_IMAGE_VIRT;
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_WIDTH, &width_img, sizeof(width_img)));
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_HEIGHT, &height_img, sizeof(height_img)));
		if (format != VX_DF_IMAGE_U8)
			status = VX_ERROR_INVALID_FORMAT;

		else if (width_img < 0)
			status = VX_ERROR_INVALID_DIMENSION;

		else if (height_img < 0)
			status = VX_ERROR_INVALID_DIMENSION;

		else
			status = VX_SUCCESS;

		ERROR_CHECK_STATUS(vxReleaseImage((vx_image*)&ref));
	}

	return status;
}

//! \brief The output validator callback.
static vx_status VX_CALLBACK seamfind_model_output_validator(vx_node node, vx_uint32 index, vx_meta_format meta)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	if (index == 6)
	{ // Image object	
		//Query Weight Image
		vx_int32 width_img = 0, height_img = 0;
		vx_df_image format = VX_DF_IMAGE_VIRT;
		vx_image image = (vx_image)avxGetNodeParamRef(node, index);
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &width_img, sizeof(width_img)));
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &height_img, sizeof(height_img)));
		if (format != VX_DF_IMAGE_U8)
			status = VX_ERROR_INVALID_FORMAT;

		else if (width_img < 0)
			status = VX_ERROR_INVALID_DIMENSION;

		else if (height_img < 0)
			status = VX_ERROR_INVALID_DIMENSION;
		else
		{
			// set output image data
			ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_WIDTH, &width_img, sizeof(width_img)));
			ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_HEIGHT, &height_img, sizeof(height_img)));
			ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));
			status = VX_SUCCESS;
		}
		ERROR_CHECK_STATUS(vxReleaseImage(&image));
	}
	return status;
}

//! \brief The kernel execution.
static vx_status VX_CALLBACK seamfind_model_kernel(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
	//Number Of Cameras - Variable 0 
	vx_uint32 NumCam = 0;
	ERROR_CHECK_STATUS(vxReadScalarValue((vx_scalar)parameters[0], &NumCam));

	//ROI Array - Variable 1
	vx_array Array_ROI = (vx_array)parameters[1];
	vx_size max_roi = (vx_size)(NumCam * NumCam);
	vx_rectangle_t *Overlap_ROI = nullptr;
	vx_size stride = sizeof(VX_TYPE_RECTANGLE);
	ERROR_CHECK_STATUS(vxAccessArrayRange(Array_ROI, 0, max_roi, &stride, (void **)&Overlap_ROI, VX_READ_ONLY));

	//Overlap Matrix - Variable 2
	vx_matrix overlap_matrix = (vx_matrix)parameters[2];
	vx_int32 *Overlap_matrix = new int[max_roi];
	ERROR_CHECK_STATUS(vxReadMatrix(overlap_matrix, Overlap_matrix));

	//Input image - Variable 3
	vx_image input_image = (vx_image)parameters[3];
	void *input_image_ptr = nullptr; vx_rectangle_t input_rect;	vx_imagepatch_addressing_t input_addr;
	vx_uint32 input_width = 0, input_height = 0, plane = 0;

	ERROR_CHECK_STATUS(vxQueryImage(input_image, VX_IMAGE_ATTRIBUTE_WIDTH, &input_width, sizeof(input_width)));
	ERROR_CHECK_STATUS(vxQueryImage(input_image, VX_IMAGE_ATTRIBUTE_HEIGHT, &input_height, sizeof(input_height)));
	input_rect.start_x = input_rect.start_y = 0; input_rect.end_x = input_width; input_rect.end_y = input_height;
	ERROR_CHECK_STATUS(vxAccessImagePatch(input_image, &input_rect, plane, &input_addr, &input_image_ptr, VX_READ_ONLY));
	vx_int8 *input_ptr = (vx_int8*)input_image_ptr;

	//Simple MASK image - Variable 4
	vx_image mask_image = (vx_image)parameters[4];
	void *mask_image_ptr = nullptr; vx_rectangle_t mask_rect;	vx_imagepatch_addressing_t mask_addr;
	vx_uint32 width = 0, height = 0;

	ERROR_CHECK_STATUS(vxQueryImage(mask_image, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
	ERROR_CHECK_STATUS(vxQueryImage(mask_image, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
	mask_rect.start_x = mask_rect.start_y = 0; mask_rect.end_x = width; mask_rect.end_y = height;
	ERROR_CHECK_STATUS(vxAccessImagePatch(mask_image, &mask_rect, plane, &mask_addr, &mask_image_ptr, VX_READ_ONLY));
	vx_uint8 *MASK_ptr = (vx_uint8*)mask_image_ptr;

	//Equirectangular Image Width & Height
	vx_uint32 Img_width = width, Img_height = height / NumCam;

	//Input Weight image - Variable 5
	vx_image weight_image = (vx_image)parameters[5];
	void *weight_image_ptr = nullptr; vx_rectangle_t weight_rect;	vx_imagepatch_addressing_t weight_addr;
	weight_rect.start_x = weight_rect.start_y = 0; weight_rect.end_x = width; weight_rect.end_y = height;
	ERROR_CHECK_STATUS(vxAccessImagePatch(weight_image, &weight_rect, plane, &weight_addr, &weight_image_ptr, VX_READ_ONLY));
	vx_uint8 *weight_ptr = (vx_uint8*)weight_image_ptr;

	//Output Weight image - Variable 6
	vx_image new_weight_image = (vx_image)parameters[6];
	void *new_weight_image_ptr = nullptr; vx_rectangle_t output_weight_rect;	vx_imagepatch_addressing_t output_weight_addr;
	output_weight_rect.start_x = output_weight_rect.start_y = 0; output_weight_rect.end_x = width; output_weight_rect.end_y = height;
	ERROR_CHECK_STATUS(vxAccessImagePatch(new_weight_image, &output_weight_rect, plane, &output_weight_addr, &new_weight_image_ptr, VX_READ_AND_WRITE));
	vx_uint8 *output_weight_ptr = (vx_uint8*)new_weight_image_ptr;

	//Copy basic weight into output weight img
	void *ptr1 = nullptr;	void *ptr2 = nullptr;
	size_t len = output_weight_addr.stride_x * (output_weight_addr.dim_x * output_weight_addr.scale_x) / VX_SCALE_UNITY;

#pragma omp parallel for
	for (vx_uint32 y = 0; y < height; y += output_weight_addr.step_y)
	{
		ptr1 = vxFormatImagePatchAddress2d(weight_image_ptr, 0, y - output_weight_rect.start_y, &output_weight_addr);
		ptr2 = vxFormatImagePatchAddress2d(new_weight_image_ptr, 0, y - output_weight_rect.start_y, &output_weight_addr);
		memcpy(ptr2, ptr1, len);
	}
	//Cost Function array
	int Num_Overlap = 0;
	for (vx_uint32 i = 0; i < NumCam; i++)
		for (vx_uint32 j = i + 1; j < NumCam; j++)
		{
			vx_uint32 ID = (i * NumCam) + j;
			if (Overlap_matrix[ID] != 0)
				Num_Overlap++;
		}

	//SeamFind Accum Variable - Internal
	std::vector<StitchSeamFindAccum> cost_array;
	cost_array.resize(Img_width*Img_height*Num_Overlap);

	//Overlap Counter
	vx_uint32 overlap_count = 0;

	//Env Variable to Draw the Seam Found for verification
	int DRAW_SEAM = 0, SEAM_ADJUST = 0, PRINT_COST = 0;
	char textBuffer[256];
	if (StitchGetEnvironmentVariable("DRAW_SEAM", textBuffer, sizeof(textBuffer))){ DRAW_SEAM = atoi(textBuffer); }
	if (StitchGetEnvironmentVariable("SEAM_ADJUST", textBuffer, sizeof(textBuffer))){ SEAM_ADJUST = atoi(textBuffer); }
	if (StitchGetEnvironmentVariable("PRINT_COST", textBuffer, sizeof(textBuffer))){ PRINT_COST = atoi(textBuffer); }

	//Loop over all the overlap camera once
#pragma omp parallel for shared(overlap_count)
	for (vx_uint32 i = 0; i < NumCam; i++)
		for (vx_uint32 j = i + 1; j < NumCam; j++)
		{
			vx_uint32 ID = (i * NumCam) + j;
			if (Overlap_matrix[ID] != 0)
			{
				vx_uint32 output_offset = overlap_count * Img_height;
				vx_uint32 offset_1 = i * Img_height;
				vx_uint32 offset_2 = j * Img_height;

				vx_int32 min_cost = 0X7FFFFFFF;
				vx_uint32 min_x = -1, min_y = -1;
				int y_dir = Overlap_ROI[ID].end_y - Overlap_ROI[ID].start_y;
				int x_dir = Overlap_ROI[ID].end_x - Overlap_ROI[ID].start_x;
				/***********************************************************************************************************************************
				Vertical SeamCut
				************************************************************************************************************************************/
				if (y_dir >= x_dir)
				{
#if ENABLE_VERTICAL_SEAM
#pragma omp parallel for shared(cost_array)
					for (vx_uint32 ye = Overlap_ROI[ID].start_y; ye <= Overlap_ROI[ID].end_y; ye++)
						for (vx_uint32 xe = Overlap_ROI[ID].start_x; xe <= Overlap_ROI[ID].end_x; xe++)
						{
							vx_uint32 pixel_id_1 = ((ye + offset_1) * Img_width) + xe;
							vx_uint32 pixel_id_2 = ((ye + offset_2) * Img_width) + xe;

							vx_int32 pixel = 0x7F00FFFF;
							//Input overlap Pixel from first Image
							if (MASK_ptr[pixel_id_1] && MASK_ptr[pixel_id_2])
								pixel = (vx_int32)input_ptr[pixel_id_1];

							//Calculate the output Pixel ID
							vx_uint32 output_pixel_id = ((ye + output_offset) * Img_width) + xe;

							//Top row pixels value set and Parent X & Y set to Invalid
							if (ye == Overlap_ROI[ID].start_y)
							{
								cost_array[output_pixel_id].value = pixel;
								cost_array[output_pixel_id].parent_x = cost_array[output_pixel_id].parent_y = -1;
							}
							else
							{
								vx_int32 left = 0x7FFFFFFF, right = 0x7FFFFFFF, middle = 0x7FFFFFFF;

								//Fetch Left, Right & Middle pixel value for the top row
								if (((ye - 1) >= Overlap_ROI[ID].start_y) && ((xe - 1) >= Overlap_ROI[ID].start_x))
								{
									vx_uint32 ID_left = (((ye - 1) + output_offset) * Img_width) + (xe - 1);
									left = cost_array[ID_left].value;
								}
								if (((ye - 1) >= Overlap_ROI[ID].start_y) && ((xe + 1) <= Overlap_ROI[ID].end_x))
								{
									vx_uint32 ID_right = (((ye - 1) + output_offset) * Img_width) + (xe + 1);
									right = cost_array[ID_right].value;
								}
								if ((ye - 1) >= Overlap_ROI[ID].start_y)
								{
									vx_uint32 ID_middle = (((ye - 1) + output_offset) * Img_width) + xe;
									middle = cost_array[ID_middle].value;
								}

								//Select the least cost parent
								if (right < middle && right < left)
								{
									cost_array[output_pixel_id].value = pixel + right;
									cost_array[output_pixel_id].parent_x = (xe + 1);
									cost_array[output_pixel_id].parent_y = (ye - 1);
								}
								else if (left < right && left < middle)
								{
									cost_array[output_pixel_id].value = pixel + left;
									cost_array[output_pixel_id].parent_x = (xe - 1);
									cost_array[output_pixel_id].parent_y = (ye - 1);
								}
								else
								{
									cost_array[output_pixel_id].value = pixel + middle;
									cost_array[output_pixel_id].parent_x = xe;
									cost_array[output_pixel_id].parent_y = (ye - 1);
								}
							}
						}

					//Select the least cost pixel for the start of the seam
					vx_uint32 ye = Overlap_ROI[ID].end_y;
					min_y = ye;

					if (PRINT_COST)
						printf("CPU::Overlap %d,%d-->", i, j);

					for (vx_int32 xe = Overlap_ROI[ID].end_x; xe >= (vx_int32)Overlap_ROI[ID].start_x; xe--)
					{
						vx_uint32 pixel_id = ((ye + output_offset) * Img_width) + xe;

						if (min_cost > cost_array[pixel_id].value)
						{
							min_cost = cost_array[pixel_id].value;
							min_x = xe;

							if (PRINT_COST)
								printf("Xe:%d-->Cost:%d  ", xe, cost_array[pixel_id].value);

						}
					}

					if (PRINT_COST)
						printf("\n");

					//Selected Min Path 
					vx_uint32 min_path_start = ((min_y + output_offset) * Img_width) + min_x;

					//Traverse the path to obtain the seam
					while (cost_array[min_path_start].parent_x != -1)
					{
						//Set Initial Weight Values:TBD:
						int i_val = 0, j_val = 0;
						vx_uint32 weight_pixel_check = ((min_y + offset_1) * Img_width) + Overlap_ROI[ID].end_x;
						if (output_weight_ptr[weight_pixel_check] == 255){ i_val = 255; j_val = 0; }
						else{ i_val = 0; j_val = 255; }

						//Weights manipulation to match the seam
#pragma omp parallel for shared(i_val, j_val)
						for (vx_int32 xe = Overlap_ROI[ID].end_x; xe >= (vx_int32)Overlap_ROI[ID].start_x; xe--)
						{
							vx_uint32 pixel_id_1 = ((min_y + offset_1) * Img_width) + xe;
							vx_uint32 pixel_id_2 = ((min_y + offset_2) * Img_width) + xe;
							int seam_flag = 1;

							if (MASK_ptr[pixel_id_1] && MASK_ptr[pixel_id_2])
							{
#if !ENABLE_HORIZONTAL_SEAM
								for (vx_uint32 cam = 0; cam < NumCam; cam++)
									if (cam != i && cam != j)
									{
										vx_uint32 offset_pix = cam * Img_height;
										vx_uint32 pixel_id_pix = ((min_y + offset_pix) * Img_width) + xe;								
#if 1
										if (output_weight_ptr[pixel_id_pix])
											seam_flag = 0;
#else
										output_weight_ptr[pixel_id_pix] = 0;
#endif

									}
#endif
								if (seam_flag)
								{
									output_weight_ptr[pixel_id_1] = i_val;
									output_weight_ptr[pixel_id_2] = j_val;
								}
							}
							if (xe == min_x)
							{
								if (i_val == 255){ i_val = 0; j_val = 255; }
								else{ i_val = 255; j_val = 0; }
								if (DRAW_SEAM)
								{
									output_weight_ptr[pixel_id_1] = 0;
									output_weight_ptr[pixel_id_2] = 0;
								}
							}
						}
						min_y--;
						min_x = cost_array[min_path_start].parent_x;
						min_path_start = ((cost_array[min_path_start].parent_y + output_offset) * Img_width) + cost_array[min_path_start].parent_x;
					}
#endif
				}

				/***********************************************************************************************************************************
				Horizontal SeamCut
				************************************************************************************************************************************/
				else if (x_dir > y_dir)
				{
#if ENABLE_HORIZONTAL_SEAM
#pragma omp parallel for shared(cost_array)
					for (vx_uint32 xe = Overlap_ROI[ID].start_x; xe <= Overlap_ROI[ID].end_x; xe++)
						for (vx_uint32 ye = Overlap_ROI[ID].start_y; ye <= Overlap_ROI[ID].end_y; ye++)
						{
							vx_uint32 pixel_id_1 = ((ye + offset_1) * Img_width) + xe;
							vx_uint32 pixel_id_2 = ((ye + offset_2) * Img_width) + xe;

							vx_uint32 pixel = 0x7F0000FF;
							if (MASK_ptr[pixel_id_1] && MASK_ptr[pixel_id_2])
								pixel = input_ptr[pixel_id_1]; //Input overlap Pixel from first Image

							//Calculate the output Pixel ID
							vx_uint32 output_pixel_id = ((ye + output_offset) * Img_width) + xe;

							//Top Column pixels value set and Parent X & Y set to Invalid
							if (xe == Overlap_ROI[ID].start_x)
							{
								cost_array[output_pixel_id].value = pixel;
								cost_array[output_pixel_id].parent_x = cost_array[output_pixel_id].parent_y = -1;
							}
							else
							{
								vx_uint32 left = 0x7FFFFFFF, right = 0x7FFFFFFF, middle = 0x7FFFFFFF;

								//Fetch Left, Right & Middle pixel value for the top row
								if (((xe - 1) >= Overlap_ROI[ID].start_x) && ((ye - 1) >= Overlap_ROI[ID].start_y))
								{
									vx_uint32 ID_left = (((ye - 1) + output_offset) * Img_width) + (xe - 1);
									left = cost_array[ID_left].value;
								}
								if (((xe - 1) >= Overlap_ROI[ID].start_x) && ((ye + 1) <= Overlap_ROI[ID].end_y))
								{
									vx_uint32 ID_right = (((ye + 1) + output_offset) * Img_width) + (xe - 1);
									right = cost_array[ID_right].value;
								}
								if ((xe - 1) >= Overlap_ROI[ID].start_x)
								{
									vx_uint32 ID_middle = ((ye + output_offset) * Img_width) + (xe - 1);
									middle = cost_array[ID_middle].value;
								}

								//Select the least cost parent
								if (right < middle && right < left)
								{
									cost_array[output_pixel_id].value = pixel + right;
									cost_array[output_pixel_id].parent_x = (xe - 1);
									cost_array[output_pixel_id].parent_y = (ye + 1);
								}
								else if (left < right && left < middle)
								{
									cost_array[output_pixel_id].value = pixel + left;
									cost_array[output_pixel_id].parent_x = (xe - 1);
									cost_array[output_pixel_id].parent_y = (ye - 1);
								}
								else
								{
									cost_array[output_pixel_id].value = pixel + middle;
									cost_array[output_pixel_id].parent_x = xe - 1;
									cost_array[output_pixel_id].parent_y = ye;
								}
							}
						}

					//Select the least cost pixel for the start of the seam
					min_x = Overlap_ROI[ID].end_x;
					for (vx_int32 y = Overlap_ROI[ID].end_y; y >= (vx_int32)Overlap_ROI[ID].start_y; y--)
					{
						vx_uint32 pixel_id = ((y + output_offset) * Img_width) + min_x;
						if (min_cost > cost_array[pixel_id].value)
						{
							min_cost = cost_array[pixel_id].value;
							min_y = y;
						}
					}

					//Selected Min Path
					vx_uint32 min_path_start = ((min_y + output_offset) * Img_width) + min_x;

					//Traverse the path to obtain the seam
					while (cost_array[min_path_start].parent_y != -1 && (cost_array[min_path_start].parent_y != 0 || cost_array[min_path_start].parent_x != 0))
					{
						//Set Initial Weight Values
						int i_val = 0, j_val = 0;
						vx_uint32 weight_pixel_check = ((Overlap_ROI[ID].end_y + offset_1) * Img_width) + min_x;
						if (output_weight_ptr[weight_pixel_check] == 0){ i_val = 255; j_val = 0; }
						else{ i_val = 0; j_val = 255; }

						for (vx_int32 ye = Overlap_ROI[ID].end_y; ye >= (vx_int32)Overlap_ROI[ID].start_y; ye--)
						{
							vx_uint32 pixel_id_1 = ((ye + offset_1) * Img_width) + min_x;
							vx_uint32 pixel_id_2 = ((ye + offset_2) * Img_width) + min_x;
							int seam_flag = 1;

							if (MASK_ptr[pixel_id_1] && MASK_ptr[pixel_id_2])
							{
								for (vx_uint32 cam = 0; cam < NumCam; cam++)
									if (cam != i && cam != j)
									{
										vx_uint32 offset_pix = cam * Img_height;
										vx_uint32 pixel_id_pix = ((ye + offset_pix) * Img_width) + min_x;
#if 1
										if (output_weight_ptr[pixel_id_pix])
											seam_flag = 0;
#else
										output_weight_ptr[pixel_id_pix] = 0;
#endif
									}

								if (seam_flag)
								{
									output_weight_ptr[pixel_id_1] = i_val;
									output_weight_ptr[pixel_id_2] = j_val;
								}
							}

							if (ye == min_y)
							{
								if (i_val == 255){ i_val = 0; j_val = 255; }
								else{ i_val = 255; j_val = 0; }

								if (DRAW_SEAM)
								{
									output_weight_ptr[pixel_id_1] = 0;
									output_weight_ptr[pixel_id_2] = 0;
								}
							}
						}
						min_x--;
						min_y = cost_array[min_path_start].parent_y;
						min_path_start = ((cost_array[min_path_start].parent_y + output_offset) * Img_width) + cost_array[min_path_start].parent_x;
					}
#endif
				}
				overlap_count++;
			}
		}

	cost_array.clear();
	ERROR_CHECK_STATUS(vxCommitImagePatch(input_image, &input_rect, 0, &input_addr, input_image_ptr));
	ERROR_CHECK_STATUS(vxCommitImagePatch(mask_image, &mask_rect, 0, &mask_addr, mask_image_ptr));
	ERROR_CHECK_STATUS(vxCommitImagePatch(weight_image, &weight_rect, 0, &weight_addr, weight_image_ptr));
	ERROR_CHECK_STATUS(vxCommitImagePatch(new_weight_image, &output_weight_rect, 0, &output_weight_addr, new_weight_image_ptr));
	ERROR_CHECK_STATUS(vxCommitArrayRange(Array_ROI, 0, max_roi, Overlap_ROI));


	return VX_SUCCESS;
}

//! \brief The kernel publisher.
vx_status seamfind_model_publish(vx_context context)
{
	// add kernel to the context with callbacks
	vx_kernel kernel = vxAddKernel(context, "com.amd.loomsl.seamfind_model",
		AMDOVX_KERNEL_STITCHING_SEAMFIND_MODEL,
		seamfind_model_kernel,
		7,
		seamfind_model_input_validator,
		seamfind_model_output_validator,
		nullptr,
		nullptr);
	ERROR_CHECK_OBJECT(kernel);

	// set kernel parameters
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_MATRIX, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 6, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));

	// finalize and release kernel object
	ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
	ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

	return VX_SUCCESS;
}

/***********************************************************************************************************************************

														Seam Find GPU

***********************************************************************************************************************************/

/***********************************************************************************************************************************
Seam Find Kernel - 1 --- Set Seam Preference -- CPU/GPU - Seam Referesh
************************************************************************************************************************************/
//! \brief The input validator callback.
static vx_status VX_CALLBACK seamfind_scene_detect_input_validator(vx_node node, vx_uint32 index)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	// get reference for parameter at specified index
	vx_reference ref = avxGetNodeParamRef(node, index);
	ERROR_CHECK_OBJECT(ref);
	// validate each parameter
	if (index == 0 || index == 1)
	{//->Current Frame/NumCamera
		vx_enum type = 0;	vx_uint32 value = 0;
		ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)ref, VX_SCALAR_ATTRIBUTE_TYPE, &type, sizeof(type)));
		ERROR_CHECK_STATUS(vxReadScalarValue((vx_scalar)ref, &value));
		if (type == VX_TYPE_UINT32)
			status = VX_SUCCESS;
		ERROR_CHECK_STATUS(vxReleaseScalar((vx_scalar*)&ref));
	}
	else if (index == 2)
	{ // Image object	
		vx_int32 width_img = 0, height_img = 0;
		vx_df_image format = VX_DF_IMAGE_VIRT;
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_WIDTH, &width_img, sizeof(width_img)));
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_HEIGHT, &height_img, sizeof(height_img)));
		if (format == VX_DF_IMAGE_U8)
			status = VX_SUCCESS;
		ERROR_CHECK_STATUS(vxReleaseImage((vx_image*)&ref));
	}
	else if (index == 3)
	{ // array object
		vx_size itemsize = 0;
		vx_size capacity = 0;
		ERROR_CHECK_STATUS(vxQueryArray((vx_array)ref, VX_ARRAY_ATTRIBUTE_ITEMSIZE, &itemsize, sizeof(itemsize)));
		ERROR_CHECK_STATUS(vxQueryArray((vx_array)ref, VX_ARRAY_ATTRIBUTE_CAPACITY, &capacity, sizeof(capacity)));
		if (itemsize != sizeof(StitchSeamFindInformation)) {
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: seam_find array type should be an StitchSeamFindInformation\n");
		}
		else if (capacity == 0) {
			status = VX_ERROR_INVALID_DIMENSION;
			vxAddLogEntry((vx_reference)node, status, "ERROR: seam_find array capacity should be positive\n");
		}
		else {
			status = VX_SUCCESS;
		}
		ERROR_CHECK_STATUS(vxReleaseArray((vx_array *)&ref));
	}
	return status;
}

//! \brief The output validator callback.
static vx_status VX_CALLBACK seamfind_scene_detect_output_validator(vx_node node, vx_uint32 index, vx_meta_format meta)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	vx_reference ref = avxGetNodeParamRef(node, index);
	if (index == 4)
	{ // array object of StitchSeamFindPreference type
		vx_size itemsize = 0; vx_size arr_capacity = 0;
		vx_enum itemtype;
		ERROR_CHECK_STATUS(vxQueryArray((vx_array)ref, VX_ARRAY_ATTRIBUTE_CAPACITY, &arr_capacity, sizeof(arr_capacity)));
		ERROR_CHECK_STATUS(vxQueryArray((vx_array)ref, VX_ARRAY_ATTRIBUTE_ITEMSIZE, &itemsize, sizeof(itemsize)));
		ERROR_CHECK_STATUS(vxQueryArray((vx_array)ref, VX_ARRAY_ATTRIBUTE_ITEMTYPE, &itemtype, sizeof(itemtype)));

		if (itemsize == sizeof(StitchSeamFindPreference))
			status = VX_SUCCESS;
		else {
			status = VX_ERROR_INVALID_DIMENSION;
			vxAddLogEntry((vx_reference)node, status, "ERROR: SeamFind array element (StitchSeamFindPreference) size error\n");
		}
		// set output image meta data
		ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_ARRAY_ATTRIBUTE_ITEMTYPE, &itemtype, sizeof(itemtype)));
		ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_ARRAY_ATTRIBUTE_CAPACITY, &arr_capacity, sizeof(arr_capacity)));
		ERROR_CHECK_STATUS(vxReleaseArray((vx_array *)&ref));
	}
	if (index == 5)
	{ // array object of StitchSeamScene type
		vx_size itemsize = 0; vx_size arr_capacity = 0;
		vx_enum itemtype;
		ERROR_CHECK_STATUS(vxQueryArray((vx_array)ref, VX_ARRAY_ATTRIBUTE_CAPACITY, &arr_capacity, sizeof(arr_capacity)));
		ERROR_CHECK_STATUS(vxQueryArray((vx_array)ref, VX_ARRAY_ATTRIBUTE_ITEMSIZE, &itemsize, sizeof(itemsize)));
		ERROR_CHECK_STATUS(vxQueryArray((vx_array)ref, VX_ARRAY_ATTRIBUTE_ITEMTYPE, &itemtype, sizeof(itemtype)));

		if (itemsize == sizeof(StitchSeamFindSceneEntry))
			status = VX_SUCCESS;
		else {
			status = VX_ERROR_INVALID_DIMENSION;
			vxAddLogEntry((vx_reference)node, status, "ERROR: SeamFind array element (StitchSeamScene) size error\n");
		}
		// set output image meta data
		ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_ARRAY_ATTRIBUTE_ITEMTYPE, &itemtype, sizeof(itemtype)));
		ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_ARRAY_ATTRIBUTE_CAPACITY, &arr_capacity, sizeof(arr_capacity)));
		ERROR_CHECK_STATUS(vxReleaseArray((vx_array *)&ref));
	}
	return status;
}

//! \brief The kernel target support callback.
static vx_status VX_CALLBACK seamfind_scene_detect_query_target_support(vx_graph graph, vx_node node,
	vx_bool use_opencl_1_2,              // [input]  false: OpenCL driver is 2.0+; true: OpenCL driver is 1.2
	vx_uint32& supported_target_affinity // [output] must be set to AGO_TARGET_AFFINITY_CPU or AGO_TARGET_AFFINITY_GPU or (AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU)
	)
{
	char textBuffer[256];
	int SEAM_FIND_TARGET = 0;
	if (StitchGetEnvironmentVariable("SEAM_FIND_TARGET", textBuffer, sizeof(textBuffer))) { SEAM_FIND_TARGET = atoi(textBuffer); }

	if (!SEAM_FIND_TARGET)
		supported_target_affinity = AGO_TARGET_AFFINITY_GPU;
	else
		supported_target_affinity = AGO_TARGET_AFFINITY_CPU;

	return VX_SUCCESS;
}

//! \brief The OpenCL global work updater callback.
static vx_status VX_CALLBACK seamfind_scene_detect_opencl_global_work_update(
	vx_node node,                                  // [input] node
	const vx_reference parameters[],               // [input] parameters
	vx_uint32 num,                                 // [input] number of parameters
	vx_uint32 opencl_work_dim,                     // [input] work_dim for clEnqueueNDRangeKernel()
	vx_size opencl_global_work[],                  // [output] global_work[] for clEnqueueNDRangeKernel()
	const vx_size opencl_local_work[]              // [input] local_work[] for clEnqueueNDRangeKernel()
	)
{
	// Get the number of elements in the array
	vx_size arr_numitems = 0;
	vx_array arr = (vx_array)avxGetNodeParamRef(node, 3);				// input array
	ERROR_CHECK_OBJECT(arr);
	ERROR_CHECK_STATUS(vxQueryArray(arr, VX_ARRAY_ATTRIBUTE_NUMITEMS, &arr_numitems, sizeof(arr_numitems)));
	ERROR_CHECK_STATUS(vxReleaseArray(&arr));
	opencl_global_work[0] = (arr_numitems + opencl_local_work[0] - 1) & ~(opencl_local_work[0] - 1);

	return VX_SUCCESS;
}

//! \brief The OpenCL code generator callback.
static vx_status VX_CALLBACK seamfind_scene_detect_opencl_codegen(
	vx_node node,                                  // [input] node
	const vx_reference parameters[],               // [input] parameters
	vx_uint32 num,                                 // [input] number of parameters
	bool opencl_load_function,                     // [input]  false: normal OpenCL kernel; true: reserved
	char opencl_kernel_function_name[64],          // [output] kernel_name for clCreateKernel()
	std::string& opencl_kernel_code,               // [output] string for clCreateProgramWithSource()
	std::string& opencl_build_options,             // [output] options for clBuildProgram()
	vx_uint32& opencl_work_dim,                    // [output] work_dim for clEnqueueNDRangeKernel()
	vx_size opencl_global_work[],                  // [output] global_work[] for clEnqueueNDRangeKernel()
	vx_size opencl_local_work[],                   // [output] local_work[] for clEnqueueNDRangeKernel()
	vx_uint32& opencl_local_buffer_usage_mask,     // [output] reserved: must be ZERO
	vx_uint32& opencl_local_buffer_size_in_bytes   // [output] reserved: must be ZERO
	)
{
	// get input and output image configurations
	vx_size arr_capacity = 0;
	vx_array arr = (vx_array)avxGetNodeParamRef(node, 3);// input array
	ERROR_CHECK_OBJECT(arr);
	ERROR_CHECK_STATUS(vxQueryArray(arr, VX_ARRAY_ATTRIBUTE_CAPACITY, &arr_capacity, sizeof(arr_capacity)));
	ERROR_CHECK_STATUS(vxReleaseArray(&arr));

	char textBuffer[256];
	int VIEW_SCENE_CHANGE = 0;
	if (StitchGetEnvironmentVariable("VIEW_SCENE_CHANGE", textBuffer, sizeof(textBuffer))) { VIEW_SCENE_CHANGE = atoi(textBuffer); }
	// set kernel configuration
	vx_uint32 work_items = (vx_uint32)arr_capacity;
	strcpy(opencl_kernel_function_name, "seamfind_scene_detect");
	opencl_work_dim = 1;
	opencl_local_work[0] = 16;
	opencl_global_work[0] = (work_items + opencl_local_work[0] - 1) & ~(opencl_local_work[0] - 1);

	// Setting variables required by the interface
	opencl_local_buffer_usage_mask = 0;
	opencl_local_buffer_size_in_bytes = 0;

	// kernel header and reading
	char item[8192];
	snprintf(item, sizeof(item),
		"#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
		"#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable\n"
		"\n"
		"__kernel __attribute__((reqd_work_group_size(%d, 1, 1)))\n" // opencl_local_work[0]
		"\n"
		"void %s(uint current_frame, uint threshold,\n"				 // opencl_kernel_function_name
		"						uint ip_cost_width, uint ip_cost_height, __global uchar * ip_cost_buf, uint ip_cost_stride, uint ip_cost_offset,\n"
		"						__global char * seam_info_buf, uint seam_info_buf_offset, uint seam_info_num_items,\n"
		"						__global char * seam_pref_buf, uint seam_pref_buf_offset, uint seam_pref_num_items,\n"
		"						__global char * seam_scene_buf, uint seam_scene_buf_offset, uint seam_scene_num_items)\n"
		, (int)opencl_local_work[0], opencl_kernel_function_name);
	opencl_kernel_code = item;
	opencl_kernel_code +=
		"{\n"
		"\n"
		"int gid = get_global_id(0);\n"
		"\n"
		"if (gid < seam_info_num_items)\n"
		"{\n"
		"\n"
		"		seam_info_buf += seam_info_buf_offset + (gid * 16);\n"
		"		seam_pref_buf  =  seam_pref_buf + seam_pref_buf_offset;\n"
		"		seam_scene_buf =  seam_scene_buf + seam_scene_buf_offset;\n"
		"\n"
		"		ip_cost_buf =  ip_cost_buf + ip_cost_offset;\n"
		"		uint equi_height = (ip_cost_width >> 1);\n"
		"\n"
		"		short8 info, pref;\n"
		"		info = vload8(0, (__global short *)seam_info_buf);\n"
		"		pref = vload8(0, (__global short *)&seam_pref_buf[gid * 16]);\n"
		"		uint offset_1 = (info.s0 * equi_height);\n"
		"		uint offset_2 = (info.s1 * equi_height);\n"
		"		uint x_dir = (info.s3 - info.s2);\n"
		"		uint y_dir = (info.s5 - info.s4);\n"
		"		uint thresholdDefaultPercentage = 25;\n" // default if no threshold passed
		"		uint threshold_scene_vert = 0;\n"
		"		uint threshold_scene_hort = 0;\n"
		"\n";
	opencl_kernel_code +=
		"\n"
		"/* Vertical Seam */\n"
		"			if (y_dir >= x_dir)\n"
		"			{\n"
#if ENABLE_VERTICAL_SEAM
		"\n"
		"				if (pref.s7 != 0)\n"
		"				{\n"
		"\n"
		"					pref.s6 --;\n"
		"					if(pref.s6 == 0)\n"
		"					{\n"
		"						pref.s7 = 0;\n";
	if (VIEW_SCENE_CHANGE)
	{
		opencl_kernel_code +=
			"					pref.s2 = current_frame;\n";	// clear scene change display
	}
	opencl_kernel_code +=
		"					}\n"
		"\n"
		"				}\n"
		"\n"
		"				uint SAD = 0;\n"
		"				uint valid_pixel = 0;\n"
		"				uint changed_valid_pixel = 0;\n"
		"\n"
		"				if (pref.s7 == 0)\n"
		"				{\n"
		"					for (uint f = 0; f < 8; f++)\n"
		"					{\n"
		"						uint y_start = info.s4 + ((y_dir / 8) * f);\n"
		"						for (uint g = 0; g < 3; g++)\n"
		"						{\n"
		"							uint x_start = info.s2 + (((x_dir / 2) + ((x_dir / 10)*(g - 1))) - 4);\n"
		"							uint cost_id_1 = ((y_start + offset_1) * ip_cost_width) + x_start;\n"
		"							uint cost_id_2 = ((y_start + offset_2) * ip_cost_width) + x_start;\n"
		"							uint output_id = ((f*24)+(g*8));\n"
		"\n"
		"							for (uint k = 0; k < 8; k++)\n"
		"							{\n"
		"								uchar input_img_1 = *(__global uchar *)&ip_cost_buf[cost_id_1 + k];\n"
		"								uchar input_img_2 = *(__global uchar *)&ip_cost_buf[cost_id_2 + k];\n"
		"								uchar past_frame = 0;\n"
		"								uchar present_frame = input_img_1;\n"
		"								if(input_img_1 && input_img_2)\n"
		"								{\n"
		"									past_frame = *(__global uchar *)&seam_scene_buf[(gid * 192) + output_id + k];\n"
		"									*(__global uchar *)&seam_scene_buf[(gid * 192) + output_id + k] = present_frame;\n"
		"									SAD = abs_diff(present_frame, past_frame);\n"
		"									valid_pixel++;\n"
		"									if(SAD){ changed_valid_pixel++; SAD = 0; }\n"
		"								}\n"
		"							}\n"
		"						}\n"
		"					}\n"
		"\n"
		"					if (threshold > 0 && threshold <= 100 )\n" // if threshold: 1 - 100 in percentage
		"						thresholdDefaultPercentage = threshold; \n"
		"\n"
		"					threshold_scene_vert = (uint)(thresholdDefaultPercentage * valid_pixel * 0.01);\n"
		"\n"
		"					if(changed_valid_pixel > threshold_scene_vert && current_frame != 0 )\n"
		"					{\n"
		"						pref.s2 = current_frame;\n"
		"						pref.s6 = 1800;\n"
		"\n";
	if (!VIEW_SCENE_CHANGE)
	{
		opencl_kernel_code +=
			"						pref.s7 = 1;\n";
	}
	if (VIEW_SCENE_CHANGE == 1)
	{
		opencl_kernel_code +=
			"						pref.s7 = 2;\n";	// view Scene Change - Dark
	}
	if (VIEW_SCENE_CHANGE == 2)
	{
		opencl_kernel_code +=
			"						pref.s7 = 3;\n";	// view Scene Change - Bright
	}
	opencl_kernel_code +=
		"					}\n"
		"				}\n"
		"\n"
		"				*(__global short8 *)&seam_pref_buf[gid * 16] = pref;\n"
		"\n"
#endif
		"			}\n"

		"\n";
	opencl_kernel_code +=
		"/* Horizontal Seam */\n"
		"			else if(x_dir > y_dir)\n"
		"			{\n"
#if ENABLE_HORIZONTAL_SEAM
		"\n"
		"				if (pref.s7 != 0)\n"
		"				{\n"
		"\n"
		"					pref.s6 --;\n"
		"					if(pref.s6 == 0)\n"
		"					{\n"
		"						pref.s7 = 0;\n";
	if (VIEW_SCENE_CHANGE)
	{
		opencl_kernel_code +=
			"					pref.s2 = current_frame;\n";	// clear scene change display
	}
	opencl_kernel_code +=
		"					}\n"
		"\n"
		"				}\n"
		"\n"
		"				uint SAD = 0;\n"
		"				uint valid_pixel = 0;\n"
		"				uint changed_valid_pixel = 0;\n"
		"\n"
		"				if (pref.s7 == 0)\n"
		"				{\n"
		"					for (uint f = 0; f < 8; f++)\n"
		"					{\n"
		"						uint x_start = info.s2 + ((x_dir / 8) * f);\n"
		"						for (uint g = 0; g < 3; g++)\n"
		"						{\n"
		"							uint y_start = info.s4 + (((y_dir / 2) + ((y_dir / 10)*(g - 1))) - 4);\n"
		"							uint cost_id_1 = ((y_start + offset_1) * ip_cost_width) + x_start;\n"
		"							uint cost_id_2 = ((y_start + offset_2) * ip_cost_width) + x_start;\n"
		"							uint output_id = ((f*24)+(g*8));\n"
		"\n"
		"							for (uint k = 0; k < 8; k++)\n"
		"							{\n"
		"								uchar input_img_1 = *(__global uchar *)&ip_cost_buf[cost_id_1 + k];\n"
		"								uchar input_img_2 = *(__global uchar *)&ip_cost_buf[cost_id_2 + k];\n"
		"								uchar past_frame = 0;\n"
		"								uchar present_frame = input_img_1;\n"
		"								if(input_img_1 && input_img_2)\n"
		"								{\n"
		"									past_frame = *(__global uchar *)&seam_scene_buf[(gid * 192) + output_id + k];\n"
		"									*(__global uchar *)&seam_scene_buf[(gid * 192) + output_id + k] = present_frame;\n"
		"									SAD = abs_diff(present_frame, past_frame);\n"
		"									valid_pixel++;\n"
		"									if(SAD){ changed_valid_pixel++; SAD = 0; }\n"
		"								}\n"
		"							}\n"
		"						}\n"
		"					}\n"
		"\n"
		"					if (threshold > 0 && threshold <= 100 )\n" // if threshold: 1 - 100 in percentage
		"						thresholdDefaultPercentage = threshold; \n"
		"\n"
		"					threshold_scene_hort = (uint)(thresholdDefaultPercentage * valid_pixel * 0.01);\n"
		"\n"
		"					if(changed_valid_pixel > threshold_scene_hort && current_frame != 0 )\n"
		"					{\n"
		"						pref.s2 = current_frame;\n"
		"						pref.s6 = 1800;\n"
		"\n";
	if (!VIEW_SCENE_CHANGE)
	{
		opencl_kernel_code +=
			"						pref.s7 = 1;\n";
	}
	if (VIEW_SCENE_CHANGE == 1)
	{
		opencl_kernel_code +=
			"						pref.s7 = 2;\n";	// view Scene Change - Dark
	}
	if (VIEW_SCENE_CHANGE == 2)
	{
		opencl_kernel_code +=
			"						pref.s7 = 3;\n";	// view Scene Change - Bright
	}
	opencl_kernel_code +=
		"					}\n"
		"				}\n"
		"\n"
		"				*(__global short8 *)&seam_pref_buf[gid * 16] = pref;\n"
		"\n"
#endif
		"			}\n"
		"\n";

	opencl_kernel_code +=
		"	}\n"
		"}\n";

	return VX_SUCCESS;
}

//! \brief The kernel execution on the CPU.
static vx_status VX_CALLBACK seamfind_scene_detect_kernel(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
	//Number Of Cameras - Variable 0 & 1
	vx_uint32 current_frame = 0, width_eqr = 0, height_eqr = 0, Threshold_scalar = 0;
	ERROR_CHECK_STATUS(vxReadScalarValue((vx_scalar)parameters[0], &current_frame));
	ERROR_CHECK_STATUS(vxReadScalarValue((vx_scalar)parameters[1], &Threshold_scalar));

	//Input image - Variable 2
	vx_image input_image = (vx_image)parameters[2];
	void *input_image_ptr = nullptr; vx_rectangle_t input_rect;	vx_imagepatch_addressing_t input_addr;
	vx_uint32 input_width = 0, input_height = 0, plane = 0;
	ERROR_CHECK_STATUS(vxQueryImage(input_image, VX_IMAGE_ATTRIBUTE_WIDTH, &input_width, sizeof(input_width)));
	ERROR_CHECK_STATUS(vxQueryImage(input_image, VX_IMAGE_ATTRIBUTE_HEIGHT, &input_height, sizeof(input_height)));
	input_rect.start_x = input_rect.start_y = 0; input_rect.end_x = input_width; input_rect.end_y = input_height;
	ERROR_CHECK_STATUS(vxAccessImagePatch(input_image, &input_rect, plane, &input_addr, &input_image_ptr, VX_READ_ONLY));
	vx_uint8 *input_ptr = (vx_uint8*)input_image_ptr;
	width_eqr = input_width;
	height_eqr = (width_eqr >> 1);

	//SeamFindInfo Array - Variable 3
	vx_size arr_numitems = 0;
	vx_array SeamFindInfo = (vx_array)parameters[3];
	ERROR_CHECK_STATUS(vxQueryArray(SeamFindInfo, VX_ARRAY_ATTRIBUTE_NUMITEMS, &arr_numitems, sizeof(arr_numitems)));
	StitchSeamFindInformation *SeamFindInfo_ptr = nullptr;
	vx_size stride = sizeof(StitchSeamFindInformation);
	ERROR_CHECK_STATUS(vxAccessArrayRange(SeamFindInfo, 0, arr_numitems, &stride, (void **)&SeamFindInfo_ptr, VX_READ_ONLY));

	//Output Preference Array - Variable 4
	vx_array Array_Pref = (vx_array)parameters[4];
	StitchSeamFindPreference *Seam_Pref = nullptr;
	stride = sizeof(StitchSeamFindPreference);
	ERROR_CHECK_STATUS(vxAccessArrayRange(Array_Pref, 0, arr_numitems, &stride, (void **)&Seam_Pref, VX_READ_AND_WRITE));

	//Output SceneChange Array - Variable 5
	vx_array Array_previous_scene = (vx_array)parameters[5];
	StitchSeamFindSceneEntry *Seam_Previous_scene = nullptr;
	stride = sizeof(StitchSeamFindSceneEntry);
	ERROR_CHECK_STATUS(vxAccessArrayRange(Array_previous_scene, 0, arr_numitems, &stride, (void **)&Seam_Previous_scene, VX_READ_AND_WRITE));

	//Local SeamScene Data
	std::vector<StitchSeamFindSceneEntry> current_seam_scene;
	current_seam_scene.resize(arr_numitems);

	//Get ENV Variables to override/set pref
	char textBuffer[256];
	int SEAM_THRESHOLD = 1500, VIEW_SCENE_CHANGE = 0, SCENE_DURATION = 150, SEAM_FREQUENCY = 300;
	if (StitchGetEnvironmentVariable("SEAM_THRESHOLD", textBuffer, sizeof(textBuffer))) { SEAM_THRESHOLD = atoi(textBuffer); }
	if (StitchGetEnvironmentVariable("VIEW_SCENE_CHANGE", textBuffer, sizeof(textBuffer))) { VIEW_SCENE_CHANGE = atoi(textBuffer); }
	if (StitchGetEnvironmentVariable("SCENE_DURATION", textBuffer, sizeof(textBuffer))) { SCENE_DURATION = atoi(textBuffer); }
	if (StitchGetEnvironmentVariable("SEAM_FREQUENCY", textBuffer, sizeof(textBuffer))){ SEAM_FREQUENCY = atoi(textBuffer); }

	//Live Updated Threshold value
	if (Threshold_scalar){ SEAM_THRESHOLD = (int)((Threshold_scalar * (192 * 255)) * 0.01); }

	//Loop over all the overlap camera once
	for (vx_uint32 i = 0; i < arr_numitems; i++)
	{
		vx_uint32 offset_1 = SeamFindInfo_ptr[i].cam_id_1 * height_eqr;
		vx_uint32 offset_2 = SeamFindInfo_ptr[i].cam_id_2 * height_eqr;
		int y_dir = SeamFindInfo_ptr[i].end_y - SeamFindInfo_ptr[i].start_y;
		int x_dir = SeamFindInfo_ptr[i].end_x - SeamFindInfo_ptr[i].start_x;
		/***********************************************************************************************************************************
		Vertical SeamCut
		************************************************************************************************************************************/
		if (y_dir >= x_dir)
		{
#if	ENABLE_VERTICAL_SEAM
			//count down previous scene change
			if (Seam_Pref[i].scene_flag != 0)
			{
				Seam_Pref[i].seam_lock--;
				if (Seam_Pref[i].seam_lock == 0)
				{
					Seam_Pref[i].scene_flag = 0;
					if (VIEW_SCENE_CHANGE == 1 || VIEW_SCENE_CHANGE == 2)
						Seam_Pref[i].start_frame = current_frame;
				}
			}

			//Find current frame segement values
			for (int f = 0; f < 8; f++)
			{
				int y_start = SeamFindInfo_ptr[i].start_y + ((y_dir / 8) * f);
				for (int g = 0; g < 3; g++)
				{
					int x_start = SeamFindInfo_ptr[i].start_x + (((x_dir / 2) + ((x_dir / 10)*(g - 1))) - 4);
					for (int k = 0; k < MAX_SEAM_BYTES; k++)
					{
						int cost_id_1 = ((y_start + offset_1)*width_eqr) + (x_start + k);
						int cost_id_2 = ((y_start + offset_2)*width_eqr) + (x_start + k);
						if (input_ptr[cost_id_1] && input_ptr[cost_id_2])
						{
#if 1
							current_seam_scene[i].segment[(f * 3) + g][k] = /*abs*/(input_ptr[cost_id_1]); //Check cost from one image
#else
							current_seam_scene[i].segment[(f * 3) + g][k] = /*abs*/((input_ptr[cost_id_1] + input_ptr[cost_id_2]) / 2);//Check cost from two images
#endif
						}
					}
				}
			}
			//First Frame reference value stored
			if (current_frame == 0)
			{
				for (int f = 0; f < MAX_SEGMENTS; f++)
					for (int k = 0; k < MAX_SEAM_BYTES; k++)
						Seam_Previous_scene[i].segment[f][k] = current_seam_scene[i].segment[f][k];
			}
			//Else calculate SAD and store the values
			else
			{
				int SAD = 0;
				for (int f = 0; f < MAX_SEGMENTS; f++)
					for (int k = 0; k < MAX_SEAM_BYTES; k++)
					{
						SAD += abs(Seam_Previous_scene[i].segment[f][k] - current_seam_scene[i].segment[f][k]);
						Seam_Previous_scene[i].segment[f][k] = current_seam_scene[i].segment[f][k];
					}
				//if scene change detected, set seam to be found in the current frame
				if (SAD > SEAM_THRESHOLD && Seam_Pref[i].scene_flag == 0)
				{
					Seam_Pref[i].start_frame = current_frame;
					Seam_Pref[i].scene_flag = 1;
					Seam_Pref[i].seam_lock = SCENE_DURATION;
					if (VIEW_SCENE_CHANGE == 1 || VIEW_SCENE_CHANGE == 2)
						Seam_Pref[i].scene_flag = (VIEW_SCENE_CHANGE + 1);
				}
			}
#endif
		}
		/***********************************************************************************************************************************
		Horizontal SeamCut
		************************************************************************************************************************************/
		else if (x_dir > y_dir)
		{
#if ENABLE_HORIZONTAL_SEAM
			//count down previous scene change
			if (Seam_Pref[i].scene_flag != 0)
			{
				Seam_Pref[i].seam_lock--;
				if (Seam_Pref[i].seam_lock == 0)
				{
					Seam_Pref[i].scene_flag = 0;
					if (VIEW_SCENE_CHANGE == 1 || VIEW_SCENE_CHANGE == 2)
						Seam_Pref[i].start_frame = current_frame;
				}
			}

			//Find current frame segement value
			for (int f = 0; f < 8; f++)
			{
				int x_start = SeamFindInfo_ptr[i].start_x + ((x_dir / 8) * f);
				for (int g = 0; g < 3; g++)
				{
					int y_start = SeamFindInfo_ptr[i].start_y + (((y_dir / 2) + ((y_dir / 10)*(g - 1))) - 4);
					for (int k = 0; k < MAX_SEAM_BYTES; k++)
					{
						int cost_id_1 = ((y_start + offset_1)*width_eqr) + (x_start + k);
						int cost_id_2 = ((y_start + offset_2)*width_eqr) + (x_start + k);
						if (input_ptr[cost_id_1] && input_ptr[cost_id_2])
						{
#if 1
							current_seam_scene[i].segment[(f * 3) + g][k] = /*abs*/(input_ptr[cost_id_1]);//Check cost from one image
#else
							current_seam_scene[i].segment[(f * 3) + g][k] = /*abs*/((input_ptr[cost_id_1] + input_ptr[cost_id_2]) / 2);//Check cost from two images
#endif
						}
					}
				}
			}
			//First Frame reference value stored
			if (current_frame == 0)
			{
				for (int f = 0; f < MAX_SEGMENTS; f++)
					for (int k = 0; k < MAX_SEAM_BYTES; k++)
						Seam_Previous_scene[i].segment[f][k] = current_seam_scene[i].segment[f][k];
			}
			//Else calculate SAD and store the values
			else
			{
				int SAD = 0;
				for (int f = 0; f < MAX_SEGMENTS; f++)
					for (int k = 0; k < MAX_SEAM_BYTES; k++)
					{
						SAD += abs(Seam_Previous_scene[i].segment[f][k] - current_seam_scene[i].segment[f][k]);
						Seam_Previous_scene[i].segment[f][k] = current_seam_scene[i].segment[f][k];
					}
				//if scene change detected, set seam to be found in the current frame
				if (SAD > SEAM_THRESHOLD && Seam_Pref[i].scene_flag == 0)
				{
					Seam_Pref[i].start_frame = current_frame;
					Seam_Pref[i].scene_flag = 1;
					Seam_Pref[i].seam_lock = SCENE_DURATION;
					if (VIEW_SCENE_CHANGE == 1 || VIEW_SCENE_CHANGE == 2)
						Seam_Pref[i].scene_flag = (VIEW_SCENE_CHANGE + 1);
				}
			}
#endif
		}
	}

	ERROR_CHECK_STATUS(vxCommitImagePatch(input_image, &input_rect, 0, &input_addr, input_image_ptr));
	ERROR_CHECK_STATUS(vxCommitArrayRange(SeamFindInfo, 0, arr_numitems, SeamFindInfo_ptr));
	ERROR_CHECK_STATUS(vxCommitArrayRange(Array_Pref, 0, arr_numitems, Seam_Pref));
	ERROR_CHECK_STATUS(vxCommitArrayRange(Array_previous_scene, 0, arr_numitems, Seam_Previous_scene));

	//Clear Memory
	current_seam_scene.clear();

	return VX_SUCCESS;
}

//! \brief The kernel publisher.
vx_status seamfind_scene_detect_publish(vx_context context)
{
	// add kernel to the context with callbacks
	vx_kernel kernel = vxAddKernel(context, "com.amd.loomsl.seamfind_scene_detect",
		AMDOVX_KERNEL_STITCHING_SEAMFIND_SCENE_DETECT,
		seamfind_scene_detect_kernel,
		6,
		seamfind_scene_detect_input_validator,
		seamfind_scene_detect_output_validator,
		nullptr,
		nullptr);
	ERROR_CHECK_OBJECT(kernel);
	amd_kernel_query_target_support_f query_target_support_f = seamfind_scene_detect_query_target_support;
	amd_kernel_opencl_codegen_callback_f opencl_codegen_callback_f = seamfind_scene_detect_opencl_codegen;
	amd_kernel_opencl_global_work_update_callback_f opencl_global_work_update_callback_f = seamfind_scene_detect_opencl_global_work_update;
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_CODEGEN_CALLBACK, &opencl_codegen_callback_f, sizeof(opencl_codegen_callback_f)));
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_GLOBAL_WORK_UPDATE_CALLBACK, &opencl_global_work_update_callback_f, sizeof(opencl_global_work_update_callback_f)));

	// set kernel parameters
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 4, VX_OUTPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 5, VX_OUTPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));

	// finalize and release kernel object
	ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
	ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

	return VX_SUCCESS;
}

/***********************************************************************************************************************************

Seam Find Kernel: 2 - GPU Cost Calculator

************************************************************************************************************************************/
//! \brief The input validator callback.
static vx_status VX_CALLBACK seamfind_cost_generate_input_validator(vx_node node, vx_uint32 index)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	// get reference for parameter at specified index
	vx_reference ref = avxGetNodeParamRef(node, index);
	ERROR_CHECK_OBJECT(ref);
	// validate each parameter
	if (index == 0)
	{ // object of SCALAR type
		vx_enum itemtype = VX_TYPE_INVALID;
		ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)ref, VX_SCALAR_ATTRIBUTE_TYPE, &itemtype, sizeof(itemtype)));

		if (itemtype == VX_TYPE_UINT32) {
			status = VX_SUCCESS;
		}
		else {
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: SeamFind Flag scalar type should be a UINT32\n");
		}
		ERROR_CHECK_STATUS(vxReleaseScalar((vx_scalar *)&ref));
	}
	else if (index == 1)
	{ // image of format U008
		// check input image format and dimensions
		vx_uint32 input_width = 0, input_height = 0;
		vx_df_image input_format = VX_DF_IMAGE_VIRT;
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_WIDTH, &input_width, sizeof(input_width)));
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_HEIGHT, &input_height, sizeof(input_height)));
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_FORMAT, &input_format, sizeof(input_format)));

		if (input_format != VX_DF_IMAGE_U8)
		{
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: SeamFind doesn't support Cost image format: %4.4s\n", &input_format);
		}
		else
			status = VX_SUCCESS;
		ERROR_CHECK_STATUS(vxReleaseImage((vx_image *)&ref));
	}

	return status;
}

//! \brief The output validator callback.
static vx_status VX_CALLBACK seamfind_cost_generate_output_validator(vx_node node, vx_uint32 index, vx_meta_format meta)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	if (index == 2 || index == 3)
	{ // Image object	
		//Query Weight Image
		vx_int32 width_img = 0, height_img = 0;
		vx_df_image format = VX_DF_IMAGE_VIRT;
		vx_image image = (vx_image)avxGetNodeParamRef(node, index);
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &width_img, sizeof(width_img)));
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &height_img, sizeof(height_img)));
		if (format != VX_DF_IMAGE_U8)
			status = VX_ERROR_INVALID_FORMAT;

		else if (width_img < 0)
			status = VX_ERROR_INVALID_DIMENSION;

		else if (height_img < 0)
			status = VX_ERROR_INVALID_DIMENSION;
		else
		{
			// set output image data
			ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_WIDTH, &width_img, sizeof(width_img)));
			ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_HEIGHT, &height_img, sizeof(height_img)));
			ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));
			status = VX_SUCCESS;
		}
		ERROR_CHECK_STATUS(vxReleaseImage(&image));
	}
	return status;
}

//! \brief The kernel target support callback.
static vx_status VX_CALLBACK seamfind_cost_generate_query_target_support(vx_graph graph, vx_node node,
	vx_bool use_opencl_1_2,              // [input]  false: OpenCL driver is 2.0+; true: OpenCL driver is 1.2
	vx_uint32& supported_target_affinity // [output] must be set to AGO_TARGET_AFFINITY_CPU or AGO_TARGET_AFFINITY_GPU or (AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU)
	)
{
	supported_target_affinity = AGO_TARGET_AFFINITY_GPU;
	return VX_SUCCESS;
}

//! \brief The OpenCL global work updater callback.
static vx_status VX_CALLBACK seamfind_cost_generate_opencl_global_work_update(
	vx_node node,                                  // [input] node
	const vx_reference parameters[],               // [input] parameters
	vx_uint32 num,                                 // [input] number of parameters
	vx_uint32 opencl_work_dim,                     // [input] work_dim for clEnqueueNDRangeKernel()
	vx_size opencl_global_work[],                  // [output] global_work[] for clEnqueueNDRangeKernel()
	const vx_size opencl_local_work[]              // [input] local_work[] for clEnqueueNDRangeKernel()
	)
{
	return VX_SUCCESS;
}

//! \brief The OpenCL code generator callback.
static vx_status VX_CALLBACK seamfind_cost_generate_opencl_codegen(
	vx_node node,                                  // [input] node
	const vx_reference parameters[],               // [input] parameters
	vx_uint32 num,                                 // [input] number of parameters
	bool opencl_load_function,                     // [input]  false: normal OpenCL kernel; true: reserved
	char opencl_kernel_function_name[64],          // [output] kernel_name for clCreateKernel()
	std::string& opencl_kernel_code,               // [output] string for clCreateProgramWithSource()
	std::string& opencl_build_options,             // [output] options for clBuildProgram()
	vx_uint32& opencl_work_dim,                    // [output] work_dim for clEnqueueNDRangeKernel()
	vx_size opencl_global_work[],                  // [output] global_work[] for clEnqueueNDRangeKernel()
	vx_size opencl_local_work[],                   // [output] local_work[] for clEnqueueNDRangeKernel()
	vx_uint32& opencl_local_buffer_usage_mask,     // [output] reserved: must be ZERO
	vx_uint32& opencl_local_buffer_size_in_bytes   // [output] reserved: must be ZERO
	)
{
	// read the input and output configuration
	vx_uint32 width = 0, height = 0;
	vx_image image = (vx_image)avxGetNodeParamRef(node, 1);				// input image
	ERROR_CHECK_OBJECT(image);
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
	ERROR_CHECK_STATUS(vxReleaseImage(&image));

	// set kernel configuration
	vx_uint32 work_items[2] = { (width + 7) / 8, height };
	strcpy(opencl_kernel_function_name, "seamfind_cost_generate");
	opencl_work_dim = 2;
	opencl_local_work[0] = 16;
	opencl_local_work[1] = 16;
	opencl_global_work[0] = (work_items[0] + opencl_local_work[0] - 1) & ~(opencl_local_work[0] - 1);
	opencl_global_work[1] = (work_items[1] + opencl_local_work[1] - 1) & ~(opencl_local_work[1] - 1);

	// Setting variables required by the interface
	opencl_local_buffer_usage_mask = 0;
	opencl_local_buffer_size_in_bytes = 0;

	// kernel header and reading
	char item[8192];
	snprintf(item, sizeof(item),
		"#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
		"#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable\n\n"
		"#define T1 ((float8)(0.4142135623730950488016887242097f))\n"
		"#define T2 ((float8)(2.4142135623730950488016887242097f))\n"
		"\n"
		"__kernel __attribute__((reqd_work_group_size(%d, %d, 1)))\n"					// opencl_local_work[0], opencl_local_work[1]
		"\n"
		"void %s(uint flag,\n"				// opencl_kernel_function_name
		"		 uint ip_image_width, uint ip_image_height, __global uchar * ip_image_buf, uint ip_image_stride, uint ip_image_offset,\n"
		"		 uint op_mag_width, uint op_mag_height, __global uchar * op_mag_buf, uint op_mag_stride, uint op_mag_offset,\n"
		"		 uint op_phase_width, uint op_phase_height, __global uchar * op_phase_buf, uint op_phase_stride, uint op_phase_offset)\n"
		"{\n"
		"  if (flag) {\n"
		"    uint x = get_global_id(0) * 8;\n"
		"    uint y = get_global_id(1);\n"
		"    int lx = get_local_id(0);\n"
		"    int ly = get_local_id(1);\n"
		"    bool valid = (x < %d) && (y < %d);\n"	// width, height
		, (int)opencl_local_work[0], (int)opencl_local_work[1], opencl_kernel_function_name, width, height);
	opencl_kernel_code = item;
	opencl_kernel_code +=
		"    ip_image_buf += ip_image_offset;\n"
		"    op_mag_buf += (op_mag_offset + y * op_mag_stride + x);\n"
		"    op_phase_buf += (op_phase_offset + y * op_phase_stride + x);\n\n"
		"    __local uchar lbuf[2448];   // 136x18 pixels\n"
		"    int lstride = 136;\n"
		"    { // Load 136x18 pixels into LDS using 16x16 workgroup\n"
		"      int gstride = (int) ip_image_stride;\n"
		"      int goffset = (y - 1) * gstride + x - 4;\n"
		"      int loffset = ly * lstride + (lx << 3);\n"
		"      *(__local uint2 *)(lbuf + loffset) = vload2(0, (__global uint *)(ip_image_buf + goffset));\n"
		"      bool doExtraLoad = false;\n"
		"      if (ly < 2) {\n"
		"        loffset += 16 * lstride;\n"
		"        goffset += 16 * gstride;\n"
		"        doExtraLoad = true;\n"
		"      }\n"
		"      else {\n"
		"        int lid = (ly - 2) * 16 + lx;\n"
		"        loffset = lid * lstride + 128;\n"
		"        goffset = (y - ly + lid - 1) * gstride + (((x >> 3) - lx) << 3) + 124;\n"
		"        doExtraLoad = true;\n"
		"      }\n"
		"      if (doExtraLoad) {\n"
		"        *(__local uint2 *)(lbuf + loffset) = vload2(0, (__global uint *)(ip_image_buf + goffset));\n"
		"      }\n"
		"      barrier(CLK_LOCAL_MEM_FENCE);\n"
		"    }\n\n"
		"    float8 Gx = (float8)(0), Gy = (float8)(0), tempf;\n"
		"    __local uint * lbufptr = (__local uint *)(lbuf + ly * lstride + (lx << 3));\n"
		"    // Filter row 0\n"
		"    uint4 pix = vload4(0, lbufptr);"
		"    tempf = (float8)(amd_unpack3(pix.s0), amd_unpack0(pix.s1), amd_unpack1(pix.s1), amd_unpack2(pix.s1), amd_unpack3(pix.s1), amd_unpack0(pix.s2), amd_unpack1(pix.s2), amd_unpack2(pix.s2));\n"
		"    Gx -= tempf; Gy -= tempf;\n"
		"    tempf = (float8)(amd_unpack0(pix.s1), amd_unpack1(pix.s1), amd_unpack2(pix.s1), amd_unpack3(pix.s1), amd_unpack0(pix.s2), amd_unpack1(pix.s2), amd_unpack2(pix.s2), amd_unpack3(pix.s2));\n"
		"    Gy = mad(tempf, (float8)(-2.0f), Gy);\n"
		"    tempf = (float8)(amd_unpack1(pix.s1), amd_unpack2(pix.s1), amd_unpack3(pix.s1), amd_unpack0(pix.s2), amd_unpack1(pix.s2), amd_unpack2(pix.s2), amd_unpack3(pix.s2), amd_unpack0(pix.s3));\n"
		"    Gx += tempf; Gy -= tempf;\n"
		"    // Filter row 1\n"
		"    pix = vload4(0, lbufptr + (lstride >> 2));\n"
		"    tempf = (float8)(amd_unpack3(pix.s0), amd_unpack0(pix.s1), amd_unpack1(pix.s1), amd_unpack2(pix.s1), amd_unpack3(pix.s1), amd_unpack0(pix.s2), amd_unpack1(pix.s2), amd_unpack2(pix.s2));\n"
		"    Gx = mad(tempf, (float8)(-2.0f), Gx);\n"
		"    tempf = (float8)(amd_unpack1(pix.s1), amd_unpack2(pix.s1), amd_unpack3(pix.s1), amd_unpack0(pix.s2), amd_unpack1(pix.s2), amd_unpack2(pix.s2), amd_unpack3(pix.s2), amd_unpack0(pix.s3));\n"
		"    Gx = mad(tempf, (float8)(2.0f), Gx);\n"
		"    // Filter row 2\n"
		"    pix = vload4(0, lbufptr + (lstride >> 1));\n"
		"    tempf = (float8)(amd_unpack3(pix.s0), amd_unpack0(pix.s1), amd_unpack1(pix.s1), amd_unpack2(pix.s1), amd_unpack3(pix.s1), amd_unpack0(pix.s2), amd_unpack1(pix.s2), amd_unpack2(pix.s2));\n"
		"    Gx -= tempf; Gy += tempf;\n"
		"    tempf = (float8)(amd_unpack0(pix.s1), amd_unpack1(pix.s1), amd_unpack2(pix.s1), amd_unpack3(pix.s1), amd_unpack0(pix.s2), amd_unpack1(pix.s2), amd_unpack2(pix.s2), amd_unpack3(pix.s2));\n"
		"    Gy = mad(tempf, (float8)(2.0f), Gy);\n"
		"    tempf = (float8)(amd_unpack1(pix.s1), amd_unpack2(pix.s1), amd_unpack3(pix.s1), amd_unpack0(pix.s2), amd_unpack1(pix.s2), amd_unpack2(pix.s2), amd_unpack3(pix.s2), amd_unpack0(pix.s3));\n"
		"    Gx += tempf; Gy += tempf;\n"
		"    // Compute mag & phase image\n"
		"    uint2 mag, ph;\n"
		"    int8 quad = select(select((int8)0, (int8)3, signbit(Gy)), select((int8)1, (int8)2, signbit(Gy)), signbit(Gx));\n"
		"    Gx = fabs(Gx); Gy = fabs(Gy);\n"
		"    tempf = Gx + Gy;\n"
		"    mag.s0 = amd_pack(tempf.s0123); mag.s1 = amd_pack(tempf.s4567);\n"
		"    tempf = select(select((float8)2, (float8)1, Gy < T2*Gx), (float8)0, Gy < T1*Gx);\n"
		"    tempf += (2*(convert_float8)(quad));\n"
		"    tempf = select(tempf, (float8)0, tempf > (float8)(7.0f));\n"
		"    ph.s0 = amd_pack(tempf.s0123); ph.s1 = amd_pack(tempf.s4567); ph <<= 5;\n"
		"    if (valid) {\n"
		"      *(__global uint2 *) op_mag_buf = mag;\n"
		"      *(__global uint2 *) op_phase_buf = ph;\n"
		"    }\n"
		"  }\n"
		"}\n";

	return VX_SUCCESS;
}

//! \brief The kernel execution.
static vx_status VX_CALLBACK seamfind_cost_generate_kernel(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
	return VX_ERROR_NOT_SUPPORTED;
}

//! \brief The kernel publisher.
vx_status seamfind_cost_generate_publish(vx_context context)
{
	// add kernel to the context with callbacks
	vx_kernel kernel = vxAddKernel(context, "com.amd.loomsl.seamfind_cost_generate",
		AMDOVX_KERNEL_STITCHING_SEAMFIND_COST_GENERATE,
		seamfind_cost_generate_kernel,
		4,
		seamfind_cost_generate_input_validator,
		seamfind_cost_generate_output_validator,
		nullptr,
		nullptr);
	ERROR_CHECK_OBJECT(kernel);
	amd_kernel_query_target_support_f query_target_support_f = seamfind_cost_generate_query_target_support;
	amd_kernel_opencl_codegen_callback_f opencl_codegen_callback_f = seamfind_cost_generate_opencl_codegen;
	amd_kernel_opencl_global_work_update_callback_f opencl_global_work_update_callback_f = seamfind_cost_generate_opencl_global_work_update;
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_CODEGEN_CALLBACK, &opencl_codegen_callback_f, sizeof(opencl_codegen_callback_f)));
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_GLOBAL_WORK_UPDATE_CALLBACK, &opencl_global_work_update_callback_f, sizeof(opencl_global_work_update_callback_f)));

	// set kernel parameters
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));

	// finalize and release kernel object
	ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
	ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

	return VX_SUCCESS;
}

/***********************************************************************************************************************************

Seam Find Kernel -- 3: Cost Accumulate with edgeness - Vertical & Horizontal Seam

************************************************************************************************************************************/
//! \brief The input validator callback.
static vx_status VX_CALLBACK seamfind_cost_accumulate_input_validator(vx_node node, vx_uint32 index)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	// get reference for parameter at specified index
	vx_reference ref = avxGetNodeParamRef(node, index);
	ERROR_CHECK_OBJECT(ref);
	// validate each parameter
	if (index == 0 || index == 1 || index == 2)
	{ // object of SCALAR type
		vx_enum itemtype = VX_TYPE_INVALID;
		ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)ref, VX_SCALAR_ATTRIBUTE_TYPE, &itemtype, sizeof(itemtype)));

		if (itemtype == VX_TYPE_UINT32) {
			status = VX_SUCCESS;
		}
		else {
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: SeamFind Equi Width/Height scalar type should be a UINT32\n");
		}
		ERROR_CHECK_STATUS(vxReleaseScalar((vx_scalar *)&ref));
	}
	else if (index == 3)
	{ // image of format U008
		// check input image format and dimensions
		vx_uint32 input_width = 0, input_height = 0;
		vx_df_image input_format = VX_DF_IMAGE_VIRT;
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_WIDTH, &input_width, sizeof(input_width)));
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_HEIGHT, &input_height, sizeof(input_height)));
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_FORMAT, &input_format, sizeof(input_format)));

		if (input_format != VX_DF_IMAGE_U8)
		{
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: SeamFind doesn't support Cost image format: %4.4s\n", &input_format);
		}
		else
			status = VX_SUCCESS;
		ERROR_CHECK_STATUS(vxReleaseImage((vx_image *)&ref));
	}
	else if (index == 4)
	{ // image of format U008
		// check input image format and dimensions
		vx_uint32 input_width = 0, input_height = 0;
		vx_df_image input_format = VX_DF_IMAGE_VIRT;
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_WIDTH, &input_width, sizeof(input_width)));
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_HEIGHT, &input_height, sizeof(input_height)));
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_FORMAT, &input_format, sizeof(input_format)));

		if (input_format != VX_DF_IMAGE_U8) {
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: SeamFind doesn't support phase image format: %4.4s\n", &input_format);
		}
		else
			status = VX_SUCCESS;
		ERROR_CHECK_STATUS(vxReleaseImage((vx_image *)&ref));
	}
	else if (index == 5)
	{ // image of format U008
		// check input image format and dimensions
		vx_uint32 input_width = 0, input_height = 0;
		vx_df_image input_format = VX_DF_IMAGE_VIRT;
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_WIDTH, &input_width, sizeof(input_width)));
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_HEIGHT, &input_height, sizeof(input_height)));
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_FORMAT, &input_format, sizeof(input_format)));

		if (input_format != VX_DF_IMAGE_U8) {
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: SeamFind doesn't support Mask image format: %4.4s\n", &input_format);
		}
		else
			status = VX_SUCCESS;
		ERROR_CHECK_STATUS(vxReleaseImage((vx_image *)&ref));
	}
	else if (index == 6)
	{ // array object of StitchSeamFindValidEntry type
		vx_size itemsize = 0;
		ERROR_CHECK_STATUS(vxQueryArray((vx_array)ref, VX_ARRAY_ATTRIBUTE_ITEMSIZE, &itemsize, sizeof(itemsize)));
		if (itemsize == sizeof(StitchSeamFindValidEntry)) {
			status = VX_SUCCESS;
		}
		else {
			status = VX_ERROR_INVALID_DIMENSION;
			vxAddLogEntry((vx_reference)node, status, "ERROR: SeamFind array element (StitchSeamFindValidEntry) size should be 16 bytes\n");
		}
		ERROR_CHECK_STATUS(vxReleaseArray((vx_array *)&ref));
	}
	else if (index == 7)
	{ // array object of StitchSeamFindPreference type
		vx_size itemsize = 0;
		ERROR_CHECK_STATUS(vxQueryArray((vx_array)ref, VX_ARRAY_ATTRIBUTE_ITEMSIZE, &itemsize, sizeof(itemsize)));
		if (itemsize == sizeof(StitchSeamFindPreference)) {
			status = VX_SUCCESS;
		}
		else {
			status = VX_ERROR_INVALID_DIMENSION;
			vxAddLogEntry((vx_reference)node, status, "ERROR: SeamFind array element (StitchSeamFindPreference) size should be 16 bytes\n");
		}
		ERROR_CHECK_STATUS(vxReleaseArray((vx_array *)&ref));
	}
	else if (index == 8)
	{ // array object of StitchSeamFindInformation type
		vx_size itemsize = 0;
		ERROR_CHECK_STATUS(vxQueryArray((vx_array)ref, VX_ARRAY_ATTRIBUTE_ITEMSIZE, &itemsize, sizeof(itemsize)));
		if (itemsize == sizeof(StitchSeamFindInformation)) {
			status = VX_SUCCESS;
		}
		else {
			status = VX_ERROR_INVALID_DIMENSION;
			vxAddLogEntry((vx_reference)node, status, "ERROR: SeamFind array element (StitchSeamFindPreference) size should be 16 bytes\n");
		}
		ERROR_CHECK_STATUS(vxReleaseArray((vx_array *)&ref));
	}

	return status;
}

//! \brief The output validator callback.
static vx_status VX_CALLBACK seamfind_cost_accumulate_output_validator(vx_node node, vx_uint32 index, vx_meta_format meta)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	vx_reference ref = avxGetNodeParamRef(node, index);
	if (index == 9)
	{ // array object of StitchSeamFindAccumEntry type
		vx_size itemsize = 0; vx_size arr_capacity = 0;
		vx_enum itemtype;
		ERROR_CHECK_STATUS(vxQueryArray((vx_array)ref, VX_ARRAY_ATTRIBUTE_CAPACITY, &arr_capacity, sizeof(arr_capacity)));
		ERROR_CHECK_STATUS(vxQueryArray((vx_array)ref, VX_ARRAY_ATTRIBUTE_ITEMSIZE, &itemsize, sizeof(itemsize)));
		ERROR_CHECK_STATUS(vxQueryArray((vx_array)ref, VX_ARRAY_ATTRIBUTE_ITEMTYPE, &itemtype, sizeof(itemtype)));

		if (itemsize == sizeof(StitchSeamFindAccumEntry))
			status = VX_SUCCESS;
		else {
			status = VX_ERROR_INVALID_DIMENSION;
			vxAddLogEntry((vx_reference)node, status, "ERROR: SeamFind array element (StitchSeamFindAccumEntry) size should be 12 bytes\n");
		}
		// set output image meta data
		ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_ARRAY_ATTRIBUTE_ITEMTYPE, &itemtype, sizeof(itemtype)));
		ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_ARRAY_ATTRIBUTE_CAPACITY, &arr_capacity, sizeof(arr_capacity)));
		ERROR_CHECK_STATUS(vxReleaseArray((vx_array *)&ref));
	}
	return status;
}

//! \brief The kernel target support callback.
static vx_status VX_CALLBACK seamfind_cost_accumulate_query_target_support(vx_graph graph, vx_node node,
	vx_bool use_opencl_1_2,              // [input]  false: OpenCL driver is 2.0+; true: OpenCL driver is 1.2
	vx_uint32& supported_target_affinity // [output] must be set to AGO_TARGET_AFFINITY_CPU or AGO_TARGET_AFFINITY_GPU or (AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU)
	)
{
	supported_target_affinity = AGO_TARGET_AFFINITY_GPU;
	return VX_SUCCESS;
}

//! \brief The OpenCL global work updater callback.
static vx_status VX_CALLBACK seamfind_cost_accumulate_opencl_global_work_update(
	vx_node node,                                  // [input] node
	const vx_reference parameters[],               // [input] parameters
	vx_uint32 num,                                 // [input] number of parameters
	vx_uint32 opencl_work_dim,                     // [input] work_dim for clEnqueueNDRangeKernel()
	vx_size opencl_global_work[],                  // [output] global_work[] for clEnqueueNDRangeKernel()
	const vx_size opencl_local_work[]              // [input] local_work[] for clEnqueueNDRangeKernel()
	)
{
	// Get the number of elements in the array
	vx_size arr_numitems = 0;
	vx_array arr = (vx_array)avxGetNodeParamRef(node, 6);				// input array
	ERROR_CHECK_OBJECT(arr);
	ERROR_CHECK_STATUS(vxQueryArray(arr, VX_ARRAY_ATTRIBUTE_NUMITEMS, &arr_numitems, sizeof(arr_numitems)));
	ERROR_CHECK_STATUS(vxReleaseArray(&arr));
	opencl_global_work[0] = (arr_numitems + opencl_local_work[0] - 1) & ~(opencl_local_work[0] - 1);

	return VX_SUCCESS;
}

//! \brief The OpenCL code generator callback.
static vx_status VX_CALLBACK seamfind_cost_accumulate_opencl_codegen(
	vx_node node,                                  // [input] node
	const vx_reference parameters[],               // [input] parameters
	vx_uint32 num,                                 // [input] number of parameters
	bool opencl_load_function,                     // [input]  false: normal OpenCL kernel; true: reserved
	char opencl_kernel_function_name[64],          // [output] kernel_name for clCreateKernel()
	std::string& opencl_kernel_code,               // [output] string for clCreateProgramWithSource()
	std::string& opencl_build_options,             // [output] options for clBuildProgram()
	vx_uint32& opencl_work_dim,                    // [output] work_dim for clEnqueueNDRangeKernel()
	vx_size opencl_global_work[],                  // [output] global_work[] for clEnqueueNDRangeKernel()
	vx_size opencl_local_work[],                   // [output] local_work[] for clEnqueueNDRangeKernel()
	vx_uint32& opencl_local_buffer_usage_mask,     // [output] reserved: must be ZERO
	vx_uint32& opencl_local_buffer_size_in_bytes   // [output] reserved: must be ZERO
	)
{
	// get input and output image configurations
	vx_size arr_capacity = 0;
	vx_array arr = (vx_array)avxGetNodeParamRef(node, 6); // input array
	ERROR_CHECK_OBJECT(arr);
	ERROR_CHECK_STATUS(vxQueryArray(arr, VX_ARRAY_ATTRIBUTE_CAPACITY, &arr_capacity, sizeof(arr_capacity)));
	ERROR_CHECK_STATUS(vxReleaseArray(&arr));

	// get developer configurations
	int SEAM_FIND_MODE = 0, COST_SELECT = 0, SEAM_QUALITY = 1; // TBD: use scalar flags ** DANGER **
	char textBuffer[256];
	if (StitchGetEnvironmentVariable("SEAM_FIND_MODE", textBuffer, sizeof(textBuffer)))	{ SEAM_FIND_MODE = atoi(textBuffer); }
	if (StitchGetEnvironmentVariable("COST_SELECT", textBuffer, sizeof(textBuffer)))	{ COST_SELECT = atoi(textBuffer); }
	if (StitchGetEnvironmentVariable("SEAM_QUALITY", textBuffer, sizeof(textBuffer)))	{ SEAM_QUALITY = atoi(textBuffer); }

	// set kernel configuration
	vx_uint32 work_items = (vx_uint32)arr_capacity;
	strcpy(opencl_kernel_function_name, "seamfind_cost_accumulate");
	opencl_work_dim = 1;
	opencl_local_work[0] = 256; //512;//256;//128;//64;
	opencl_global_work[0] = (work_items + opencl_local_work[0] - 1) & ~(opencl_local_work[0] - 1);

	// Setting variables required by the interface
	opencl_local_buffer_usage_mask = 0;
	opencl_local_buffer_size_in_bytes = 0;

	// kernel header and reading
	char item[8192];
	snprintf(item, sizeof(item),
		"#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
		"#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable\n"
		"\n"
		"__kernel __attribute__((reqd_work_group_size(%d, 1, 1)))\n"					// opencl_local_work[0]
		"\n"
		"void %s(uint current_frame,uint equi_width, uint equi_height,\n"				// opencl_kernel_function_name
		"						uint ip_cost_width, uint ip_cost_height, __global uchar * ip_cost_buf, uint ip_cost_stride, uint ip_cost_offset,\n"
		"						uint ip_phase_width, uint ip_phase_height, __global uchar * ip_phase_buf, uint ip_phase_stride, uint ip_phase_offset,\n"
		"						uint ip_mask_width, uint ip_mask_height, __global uchar * ip_mask_buf, uint ip_mask_stride, uint ip_mask_offset,\n"
		"						__global char * seam_valid_buf, uint seam_valid_buf_offset, uint valid_pix_num_items,\n"
		"						__global char * seam_pref_buf, uint seam_pref_buf_offset, uint seam_pref_num_items,\n"
		"						__global char * seam_info_buf, uint seam_info_buf_offset, uint seam_info_num_items,\n"
		"						__global char * seam_accum_buf, uint seam_accum_buf_offset, uint seam_num_items)\n"
		, (int)opencl_local_work[0], opencl_kernel_function_name);
	opencl_kernel_code = item;
	opencl_kernel_code +=
		"{\n"
		"\n"
		"int gid = get_global_id(0);\n"
		"\n"
		"if (gid < valid_pix_num_items)\n"
		"{\n"
		"\n"
		"	seam_valid_buf += seam_valid_buf_offset + (gid * 16);\n"
		"	seam_pref_buf =  seam_pref_buf + seam_pref_buf_offset;\n"
		"	seam_info_buf =  seam_info_buf + seam_info_buf_offset;\n"
		"	seam_accum_buf =  seam_accum_buf + seam_accum_buf_offset;\n"
		"\n"
		"	ip_cost_buf =  ip_cost_buf + ip_cost_offset;\n"
		"	ip_phase_buf =  ip_phase_buf + ip_phase_offset;\n"
		"	ip_mask_buf =  ip_mask_buf + ip_mask_offset;\n"
		"\n"
		"	int4 accum;\n"
		"	short8 dim, pref, info;\n"
		"	dim = vload8(0, (__global short *)seam_valid_buf);\n"
		"	pref = vload8(0, (__global short *)&seam_pref_buf[dim.s7 * 16]);\n"
		"	info = vload8(0, (__global short *)&seam_info_buf[dim.s7 * 16]);\n"
		"	uint overlap_offset = ((info.s7 << 16) & 0xFFFF0000) | (info.s6  & 0x0000FFFF);\n"
		"\n"
		"	if (pref.s5 != -1 && ( (pref.s2 == current_frame) || ((current_frame + 1) % (pref.s3 + pref.s1) == 0)))\n"
		"	{\n"
		"\n"
		"/*	Vertical Seam */\n"
		"		if (dim.s2 >= dim.s3)\n"
		"		{\n"
#if ENABLE_VERTICAL_SEAM
		"			uint input_offset = dim.s6 * equi_height;\n"
		"			for (uint i = 0; i < dim.s2; i++)\n"
		"			{\n"
		"				uint ID1 = (((dim.s1 + i) + input_offset) * equi_width) + dim.s0;\n"
		"				uint ID2 = ((dim.s5 + i) * equi_width) + dim.s4;\n"
		"				uint output_ID = overlap_offset + (((dim.s1 - info.s4) + i) * dim.s3) + (dim.s0 - info.s2);\n"
		"\n"
		"				uchar mask_img_1 = *(__global uchar *)&ip_mask_buf[ID1];\n"
		"				uchar mask_img_2 = *(__global uchar *)&ip_mask_buf[ID2];\n"
		"\n";
	if (!COST_SELECT)
	{
		opencl_kernel_code +=
			"				uchar cost_img = *(__global uchar *)&ip_cost_buf[ID1];\n"
			"\n"
			"				uchar phase_img_R = *(__global uchar *)&ip_phase_buf[ID1+1];\n"
			"				uchar phase_img_L = *(__global uchar *)&ip_phase_buf[ID1-1];\n"
			"				uchar magnitude_img_R = *(__global uchar *)&ip_cost_buf[ID1+1];\n"
			"				uchar magnitude_img_L = *(__global uchar *)&ip_cost_buf[ID1-1];\n"
			"\n"
			"				int Pixel = select(0x7F00FFFF, (int)cost_img, mask_img_1 && mask_img_2);\n"
			"\n";
	}
	else
	{
		opencl_kernel_code +=
			"				uchar cost_img_1 = *(__global uchar *)&ip_cost_buf[ID1];\n"
			"				uchar cost_img_2 = *(__global uchar *)&ip_cost_buf[ID2];\n"
			"				int cost_img = (int)((cost_img_1 + cost_img_2)/2) ;\n"
			"\n"
			"				uchar phase_img_R = *(__global uchar *)&ip_phase_buf[ID1+1];\n"
			"				uchar phase_img_L = *(__global uchar *)&ip_phase_buf[ID1-1];\n"
			"				uchar magnitude_img_R = *(__global uchar *)&ip_cost_buf[ID1+1];\n"
			"				uchar magnitude_img_L = *(__global uchar *)&ip_cost_buf[ID1-1];\n"
			"\n"
			"				int Pixel = 0x7F00FFFF;\n"
			"				if(mask_img_1 && mask_img_2)\n"
			"					Pixel = (int)cost_img;\n"
			"\n";
	}
	opencl_kernel_code +=
		"				//Quantize the phase image\n"
		"				phase_img_R = phase_img_R >> 5;\n"
		"				phase_img_L = phase_img_L >> 5;\n"
		"				/* Parent at the start of the seam set to control value */\n"
		"				if (i == 0)\n"
		"				{\n"
		"					accum.s0 = -1;\n"
		"					accum.s1 = Pixel;\n"
		"					accum.s2 = 0;\n"
		"					if (Pixel != 0x7F00FFFF && (dim.s0 > info.s2 && dim.s0 < info.s3))\n"
		"						accum.s2 = 1;\n"
		"				}\n"
		"				else\n"
		"				{\n"
		"					int left = 0x7FFFFFFF , right = 0x7FFFFFFF, middle = 0x7FFFFFFF;\n"
		"					int left_prop = 0, right_prop = 0, middle_prop = 0;\n"
		"					uchar mask_1 = 0, mask_2 = 0;\n"
		"\n"
		"					/* Finding parent right, left & middle values */\n"
		"					if(dim.s0 > 0 && dim.s0 > info.s2)\n"
		"					{\n"
		"\n"
		"						uint ID_left = overlap_offset + ((dim.s1 - info.s4 + i - 1) * dim.s3) + (dim.s0 - info.s2 - 1);\n"
		"						int4 left_accum = vload4(0, (__global int *)&seam_accum_buf[ID_left * 12]);\n"
		"						mask_1 = *(__global uchar *)&ip_mask_buf[((((dim.s1 + i) -1 ) + input_offset) * equi_width) + (dim.s0 - 1)];\n"
		"						mask_2 = *(__global uchar *)&ip_mask_buf[(((dim.s5 + i) -1 ) * equi_width) + (dim.s4 - 1)];\n"
		"						if(mask_1 && mask_2)\n"
		"						{\n"
		"							left = left_accum.s1;\n"
		"							left_prop = left_accum.s2;\n"
		"						}\n"
		"\n"
		"					}\n"
		"\n"
		"					if(dim.s0 < equi_width - 1 && dim.s0 < info.s3)\n"
		"					{\n"
		"\n"
		"						uint ID_right = overlap_offset + ((dim.s1 - info.s4 + i - 1) * dim.s3) + (dim.s0 - info.s2 + 1);\n"
		"						int4 right_accum = vload4(0, (__global int *)&seam_accum_buf[ID_right * 12]);\n"
		"						mask_1 = *(__global uchar *)&ip_mask_buf[((((dim.s1 + i) -1 ) + input_offset) * equi_width) + (dim.s0 + 1)];\n"
		"						mask_2 = *(__global uchar *)&ip_mask_buf[(((dim.s5 + i) -1 ) * equi_width) + (dim.s4 + 1)];\n"
		"						if(mask_1 && mask_2)\n"
		"						{\n"
		"							right = right_accum.s1;\n"
		"							right_prop = right_accum.s2;\n"
		"						}\n"
		"\n"
		"					}\n"
		"\n"
		"					uint ID_middle = overlap_offset + ((dim.s1 - info.s4 + i - 1) * dim.s3) + (dim.s0 - info.s2);\n"
		"					int4 middle_accum = vload4(0, (__global int *)&seam_accum_buf[ID_middle * 12]);\n"
		"					mask_1 = *(__global uchar *)&ip_mask_buf[((((dim.s1 + i) -1 ) + input_offset) * equi_width) + (dim.s0)];\n"
		"					mask_2 = *(__global uchar *)&ip_mask_buf[(((dim.s5 + i) -1 ) * equi_width) + (dim.s4)];\n"
		"					if(mask_1 && mask_2)\n"
		"					{\n"
		"						middle = middle_accum.s1;\n"
		"						middle_prop = middle_accum.s2;\n"
		"					}\n"
		"\n"
		"\n"
		"\n"
		"					/* Adding Bonus to the path next to an Edge */\n"
		"					int BONUS = 0, WINNER_L = 0,  WINNER_R = 0, multi = 2;\n"
		"\n";
	if (SEAM_QUALITY == 1)
	{
		opencl_kernel_code +=
			"\n"
			"				if(magnitude_img_R > 225){ WINNER_R = 50; multi = 2; }\n"
			"				if(magnitude_img_L > 225){ WINNER_L = 50; multi = 2; }\n"
			"\n"
			"				if(magnitude_img_R > 75)\n"
			"					if (phase_img_R == 0 || phase_img_R == 4)\n"
			"						BONUS = magnitude_img_R + WINNER_R;\n"
			"\n"
			"				if(magnitude_img_L > 75)\n"
			"					if (phase_img_L == 0 || phase_img_L == 4)\n"
			"						BONUS += magnitude_img_L + WINNER_L;\n"
			"\n";
	}
	else if (SEAM_QUALITY == 2)
	{
		opencl_kernel_code +=
			"\n"
			"				if(magnitude_img_R > 225) WINNER_R = 50;\n"
			"				if(magnitude_img_L > 225) WINNER_L = 50;\n"
			"\n"
			"				if(magnitude_img_R > 128)\n"
			"					if (phase_img_R == 0 || phase_img_R == 4)\n"
			"						BONUS = magnitude_img_R + WINNER_R;\n"
			"\n"
			"				if(magnitude_img_L > 128)\n"
			"					if (phase_img_L == 0 || phase_img_L == 4)\n"
			"						BONUS += magnitude_img_L + WINNER_L;\n"
			"\n";
	}
	opencl_kernel_code +=
		"\n"
		"					/* Select Right, left or middle parent path */\n"
		"\n";
	opencl_kernel_code +=
		"\n"
		"					if( (mask_img_1 && mask_img_2) && (right_prop || left_prop || middle_prop))\n"
		"					{\n"
		"\n"
		"						int valid_child = 0x7FFFFFFF;\n"
		"\n"
		"						if ((right < valid_child) && right_prop)\n"
		"						{\n"
		"							valid_child = right;\n"
		"							accum.s0 = ((((dim.s1 + i) - 1) << 16 ) & 0xFFFF0000) | ((dim.s0 + 1) & 0x0000FFFF) ;\n"
		"							accum.s1 = (right + Pixel) + (multi*BONUS);\n"
		"							accum.s2 = 1;\n"
		"						}\n"
		"						if ((left < valid_child) && left_prop)\n"
		"						{\n"
		"							valid_child = left;\n"
		"							accum.s0 = ((((dim.s1 + i) - 1) << 16 ) & 0xFFFF0000) | ((dim.s0 - 1) & 0x0000FFFF) ;\n"
		"							accum.s1 = (left + Pixel)  + (multi*BONUS);\n"
		"							accum.s2 = 1;\n"
		"						}\n"
		"						if ((middle < valid_child) && middle_prop)\n"
		"						{\n"
		"							accum.s0 = ((((dim.s1 + i) - 1) << 16 ) & 0xFFFF0000) | (dim.s0 & 0x0000FFFF) ;\n"
		"							accum.s1 = (middle + Pixel) - (multi*BONUS);\n"
		"							accum.s2 = 1;\n"
		"						}\n"
		"					}\n"
		"					else\n"
		"					{\n"
		"						if (right < middle && right < left)\n"
		"						{\n"
		"							accum.s0 = ((((dim.s1 + i) - 1) << 16 ) & 0xFFFF0000) | ((dim.s0 + 1) & 0x0000FFFF) ;\n"
		"							accum.s1 = (right + Pixel) + (multi*BONUS);\n"
		"							accum.s2 = 0;\n"
		"						}\n"
		"						else if (left < right && left < middle)\n"
		"						{\n"
		"							accum.s0 = ((((dim.s1 + i) - 1) << 16 ) & 0xFFFF0000) | ((dim.s0 - 1) & 0x0000FFFF) ;\n"
		"							accum.s1 = (left + Pixel)  + (multi*BONUS);\n"
		"							accum.s2 = 0;\n"
		"						}\n"
		"						else\n"
		"						{\n"
		"							accum.s0 = ((((dim.s1 + i) - 1) << 16 ) & 0xFFFF0000) | (dim.s0 & 0x0000FFFF) ;\n"
		"							accum.s1 = (middle + Pixel) - (multi*BONUS);\n"
		"							accum.s2 = 0;\n"
		"						}\n"
		"					}\n"
		"\n";
	opencl_kernel_code +=
		"				}\n"
		"\n"
		"				*(__global int2 *) &seam_accum_buf[output_ID * 12] = accum.s01;\n"
		"				*(__global int *) &seam_accum_buf[output_ID * 12 + 8] = accum.s2;\n"
		"				barrier(CLK_GLOBAL_MEM_FENCE);\n"
		"\n"
		"			}\n"
#endif
		"		}\n"
		"\n";
	opencl_kernel_code +=
		" /* Horizontal Seam */\n"
		"	else if(dim.s3 > dim.s2)\n"
		"	{\n"
#if ENABLE_HORIZONTAL_SEAM
		"		uint input_offset = dim.s6 * equi_height;\n"
		"		for (uint i = 0; i < dim.s3; i++)\n"
		"		{\n"
		"			uint ID1 = ((dim.s1 + input_offset) * equi_width) + (dim.s0 + i);\n"
		"			uint ID2 = ((dim.s5 * equi_width)) + (dim.s4 + i);\n"
		"			uint output_ID = overlap_offset + ((dim.s0 - info.s2 + i) * dim.s2) + (dim.s1 - info.s4) ;\n"
		"\n"
		"			uchar mask_img_1 = *(__global uchar *)&ip_mask_buf[ID1];\n"
		"			uchar mask_img_2 = *(__global uchar *)&ip_mask_buf[ID2];\n"
		"			uchar cost_img = *(__global uchar *)&ip_cost_buf[ID1];\n"
		"\n"
		"			uchar phase_img_R = 0, magnitude_img_R = 0, phase_img_L = 0, magnitude_img_L = 0;\n"
		"\n"
		"			if (dim.s1 > 0 && dim.s1 < equi_height)\n"
		"			{\n"
		"				uint Phase_ID_t = (((dim.s1 - 1) + input_offset) * equi_width) + (dim.s0 + i);\n"
		"				uint Phase_ID_b = (((dim.s1 + 1) + input_offset) * equi_width) + (dim.s0 + i);\n"
		"\n"
		"				phase_img_R = *(__global uchar *)&ip_phase_buf[Phase_ID_b];\n"
		"				magnitude_img_R = *(__global uchar *)&ip_cost_buf[Phase_ID_b];\n"
		"\n"
		"				phase_img_L = *(__global uchar *)&ip_phase_buf[Phase_ID_t];\n"
		"				magnitude_img_L = *(__global uchar *)&ip_cost_buf[Phase_ID_t];\n"
		"			}\n"
		"\n"
		"			int Pixel = select((int)0x7F00FFFF, (int)cost_img, mask_img_1 && mask_img_2);\n"
		"\n"
		"			//Quantize the phase image\n"
		"			phase_img_R = phase_img_R >> 5;\n"
		"			phase_img_L = phase_img_L >> 5;\n"
		"\n"
		"			//Parent at the start of the seam set to control value\n"
		"			if (i == 0)\n"
		"			{\n"
		"				accum.s0 = -1;\n"
		"				accum.s1 = Pixel;\n"
		"				accum.s2 = 0;\n"
		"				if (Pixel != 0x7F00FFFF)\n"
		"					accum.s2 = 1;\n"
		"			}\n"
		"			else\n"
		"			{\n"
		"				int left = 0x7FFFFFFF, right = 0x7FFFFFFF, middle = 0x7FFFFFFF;\n"
		"				int left_prop = 0, right_prop = 0, middle_prop = 0;\n"
		"				uchar mask_1 = 0, mask_2 = 0;\n"
		"\n"
		"				//Finding parent right, left & middle values\n"
		"				if (dim.s1 > 0)\n"
		"				{\n"
		"\n"
		"					uint ID_left = overlap_offset + ((dim.s0 - info.s2 + i - 1) * dim.s2) + (dim.s1 - info.s4 - 1) ;\n"
		"					int4 left_accum = vload4(0, (__global int *)&seam_accum_buf[ID_left * 12]);\n"
		"					mask_1 = *(__global uchar *)&ip_mask_buf[(((dim.s1 - 1) + input_offset) * equi_width) + ((dim.s0 + i) - 1)];\n"
		"					mask_2 = *(__global uchar *)&ip_mask_buf[((dim.s5 - 1) * equi_width) + ((dim.s4 + i) - 1)];\n"
		"					if (mask_1 && mask_2)\n"
		"					{\n"
		"						left = left_accum.s1;\n"
		"						left_prop = left_accum.s2;\n"
		"					}\n"
		"\n"
		"				}\n"
		"\n"
		"				if (dim.s1 < equi_height - 1)\n"
		"				{\n"
		"\n"
		"					uint ID_right = overlap_offset + ((dim.s0 - info.s2 + i - 1) * dim.s2) + (dim.s1 - info.s4 + 1) ;\n"
		"					int4 right_accum = vload4(0, (__global int *)&seam_accum_buf[ID_right * 12]);\n"
		"					mask_1 = *(__global uchar *)&ip_mask_buf[(((dim.s1 + 1) + input_offset) * equi_width) + ((dim.s0 + i) - 1)];\n"
		"					mask_2 = *(__global uchar *)&ip_mask_buf[((dim.s5 + 1) * equi_width) + ((dim.s4 + i) - 1)];\n"
		"					if (mask_1 && mask_2)\n"
		"					{\n"
		"						right = right_accum.s1;\n"
		"						right_prop = right_accum.s2;\n"
		"					}\n"
		"\n"
		"				}\n"
		"\n"
		"\n"
		"					uint ID_middle = overlap_offset + ((dim.s0 - info.s2 + i - 1) * dim.s2) + (dim.s1 - info.s4);\n"
		"					int4 middle_accum = vload4(0, (__global int *)&seam_accum_buf[ID_middle * 12]);\n"
		"					mask_1 = *(__global uchar *)&ip_mask_buf[((dim.s1 + input_offset) * equi_width) + ((dim.s0 + i) - 1)];\n"
		"					mask_2 = *(__global uchar *)&ip_mask_buf[(dim.s5 * equi_width) + ((dim.s4 + i) - 1)];\n"
		"					if (mask_1 && mask_2)\n"
		"					{\n"
		"						middle = middle_accum.s1;\n"
		"						middle_prop = middle_accum.s2;\n"
		"					}\n"
		"\n"
		"				//Adding Bonus to the path next to an Edge\n"
		"				int BONUS = 0, WINNER_R = 0, WINNER_L = 0;\n"
		"\n";
	if (SEAM_QUALITY == 1)
	{
		opencl_kernel_code +=
			"\n"
			"				if(magnitude_img_R > 225) WINNER_R = 50;\n"
			"				if(magnitude_img_L > 225) WINNER_L = 50;\n"
			"\n"
			"				if (magnitude_img_R > 64)\n"
			"					if (phase_img_R == 2 || phase_img_R == 6)\n"
			"						BONUS += WINNER_R + magnitude_img_R;\n"
			"\n"
			"				if (magnitude_img_L > 64)\n"
			"					if (phase_img_L == 2 || phase_img_L == 6)\n"
			"						BONUS += WINNER_L + magnitude_img_L;\n"
			"\n";
	}
	else if (SEAM_QUALITY == 2)
	{
		opencl_kernel_code +=
			"\n"
			"				if(magnitude_img_R > 200) WINNER_R = 50;\n"
			"				if(magnitude_img_L > 200) WINNER_L = 50;\n"
			"\n"
			"				if (magnitude_img_R > 128)\n"
			"					if (phase_img_R == 2 || phase_img_R == 6)\n"
			"						BONUS += WINNER_R + magnitude_img_R;\n"
			"\n"
			"				if (magnitude_img_L > 128)\n"
			"					if (phase_img_L == 2 || phase_img_L == 6)\n"
			"						BONUS += WINNER_L + magnitude_img_L;\n"
			"\n";
	}
	opencl_kernel_code +=
		"\n"
		"			/* Select Right, left or middle parent path */ \n"
		"\n"
		"\n"
		"				if( (mask_img_1 && mask_img_2) && (right_prop || left_prop || middle_prop))\n"
		"				{\n"
		"\n"
		"					int valid_child = 0x7FFFFFFF;\n"
		"\n"
		"					if ((right < valid_child) && right_prop)\n"
		"					{\n"
		"						valid_child = right;\n"
		"						accum.s0 = (((dim.s1 + 1) << 16) & 0xFFFF0000) | (((dim.s0 + i) - 1) & 0x0000FFFF);\n"
		"						accum.s1 = (right + Pixel) + (2 * BONUS);\n"
		"						accum.s2 = 1;\n"
		"					}\n"
		"					if ((left < valid_child) && left_prop)\n"
		"					{\n"
		"						valid_child = left;\n"
		"						accum.s0 = (((dim.s1 - 1) << 16) & 0xFFFF0000) | (((dim.s0 + i) - 1) & 0x0000FFFF);\n"
		"						accum.s1 = (left + Pixel) + (2 * BONUS);\n"
		"						accum.s2 = 1;\n"
		"					}\n"
		"					if ((middle < valid_child) && middle_prop)\n"
		"					{\n"
		"						accum.s0 = ((dim.s1 << 16) & 0xFFFF0000) | (((dim.s0 + i) - 1) & 0x0000FFFF);\n"
		"						accum.s1 = (middle + Pixel) - (2 * BONUS);\n"
		"						accum.s2 = 1;\n"
		"					}\n"
		"				}\n"
		"				else\n"
		"				{\n"
		"					if (right < middle && right < left)\n"
		"					{\n"
		"						accum.s0 = (((dim.s1 + 1) << 16) & 0xFFFF0000) | (((dim.s0 + i) - 1) & 0x0000FFFF);\n"
		"						accum.s1 = (right + Pixel) + (2 * BONUS);\n"
		"						accum.s2 = 0;\n"
		"					}\n"
		"					else if (left < right && left < middle)\n"
		"					{\n"
		"						accum.s0 = (((dim.s1 - 1) << 16) & 0xFFFF0000) | (((dim.s0 + i) - 1) & 0x0000FFFF);\n"
		"						accum.s1 = (left + Pixel) + (2 * BONUS);\n"
		"						accum.s2 = 0;\n"
		"					}\n"
		"					else\n"
		"					{\n"
		"						accum.s0 = ((dim.s1 << 16) & 0xFFFF0000) | (((dim.s0 + i) - 1) & 0x0000FFFF);\n"
		"						accum.s1 = (middle + Pixel) - (2 * BONUS);\n"
		"						accum.s2 = 0;\n"
		"					}\n"
		"				}\n"
		"\n"
		"			}\n"
		"\n"
		"				*(__global int2 *) &seam_accum_buf[output_ID * 12] = accum.s01;\n"
		"				*(__global int *) &seam_accum_buf[output_ID * 12 + 8] = accum.s2;\n"
		"				barrier(CLK_GLOBAL_MEM_FENCE);\n"
		"\n"
		"			}\n"
#endif
		"		}\n"
		"\n";
	opencl_kernel_code +=
		"		}\n"
		"	}\n"
		"}\n";

	return VX_SUCCESS;
}

//! \brief The kernel execution.
static vx_status VX_CALLBACK seamfind_cost_accumulate_kernel(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
	return VX_ERROR_NOT_SUPPORTED;
}

//! \brief The kernel publisher.
vx_status seamfind_cost_accumulate_publish(vx_context context)
{
	// add kernel to the context with callbacks
	vx_kernel kernel = vxAddKernel(context, "com.amd.loomsl.seamfind_cost_accumulate",
		AMDOVX_KERNEL_STITCHING_SEAMFIND_COST_ACCUMULATE,
		seamfind_cost_accumulate_kernel,
		10,
		seamfind_cost_accumulate_input_validator,
		seamfind_cost_accumulate_output_validator,
		nullptr,
		nullptr);
	ERROR_CHECK_OBJECT(kernel);
	amd_kernel_query_target_support_f query_target_support_f = seamfind_cost_accumulate_query_target_support;
	amd_kernel_opencl_codegen_callback_f opencl_codegen_callback_f = seamfind_cost_accumulate_opencl_codegen;
	amd_kernel_opencl_global_work_update_callback_f opencl_global_work_update_callback_f = seamfind_cost_accumulate_opencl_global_work_update;
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_CODEGEN_CALLBACK, &opencl_codegen_callback_f, sizeof(opencl_codegen_callback_f)));
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_GLOBAL_WORK_UPDATE_CALLBACK, &opencl_global_work_update_callback_f, sizeof(opencl_global_work_update_callback_f)));

	// set kernel parameters
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 6, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 7, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 8, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 9, VX_OUTPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));

	// finalize and release kernel object
	ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
	ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

	return VX_SUCCESS;
}

/***********************************************************************************************************************************

Seam Find Kernel: 4 - Path Tracing - Vertical & Horizontal Seam -- GPU/CPU

************************************************************************************************************************************/
//! \brief The input validator callback.
static vx_status VX_CALLBACK seamfind_path_trace_input_validator(vx_node node, vx_uint32 index)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	// get reference for parameter at specified index
	vx_reference ref = avxGetNodeParamRef(node, index);
	ERROR_CHECK_OBJECT(ref);
	// validate each parameter
	if (index == 0)
	{ // object of SCALAR type for 
		vx_enum itemtype = VX_TYPE_INVALID;
		ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)ref, VX_SCALAR_ATTRIBUTE_TYPE, &itemtype, sizeof(itemtype)));

		if (itemtype == VX_TYPE_UINT32) {
			status = VX_SUCCESS;
		}
		else {
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: SeamFind Current Frame should be a UINT32\n");
		}
		ERROR_CHECK_STATUS(vxReleaseScalar((vx_scalar *)&ref));
	}
	else if (index == 1)
	{ // image of format  U008
		// check input image format and dimensions
		vx_uint32 input_width = 0, input_height = 0;
		vx_df_image input_format = VX_DF_IMAGE_VIRT;
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_WIDTH, &input_width, sizeof(input_width)));
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_HEIGHT, &input_height, sizeof(input_height)));
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_FORMAT, &input_format, sizeof(input_format)));

		if (input_format != VX_DF_IMAGE_U8)
		{
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: SeamFind doesn't support weight image format: %4.4s\n", &input_format);
		}
		else
			status = VX_SUCCESS;
		ERROR_CHECK_STATUS(vxReleaseImage((vx_image *)&ref));
	}
	else if (index == 2)
	{ // array object of StitchInfoEntry type
		vx_size itemsize = 0;
		ERROR_CHECK_STATUS(vxQueryArray((vx_array)ref, VX_ARRAY_ATTRIBUTE_ITEMSIZE, &itemsize, sizeof(itemsize)));
		if (itemsize == sizeof(StitchSeamFindInformation)) {
			status = VX_SUCCESS;
		}
		else {
			status = VX_ERROR_INVALID_DIMENSION;
			vxAddLogEntry((vx_reference)node, status, "ERROR: SeamFind array element (StitchSeamFindInformation) size should be 16 bytes\n");
		}
		ERROR_CHECK_STATUS(vxReleaseArray((vx_array *)&ref));
	}
	else if (index == 3)
	{ // array object of StitchSeamFindAccumEntry type
		vx_size itemsize = 0;
		ERROR_CHECK_STATUS(vxQueryArray((vx_array)ref, VX_ARRAY_ATTRIBUTE_ITEMSIZE, &itemsize, sizeof(itemsize)));
		if (itemsize == sizeof(StitchSeamFindAccumEntry)) {
			status = VX_SUCCESS;
		}
		else {
			status = VX_ERROR_INVALID_DIMENSION;
			vxAddLogEntry((vx_reference)node, status, "ERROR: SeamFind array element (StitchSeamFindAccumEntry) size should be 12 bytes\n");
		}
		ERROR_CHECK_STATUS(vxReleaseArray((vx_array *)&ref));
	}
	else if (index == 4)
	{ // array object of StitchSeamFindPreference type
		vx_size itemsize = 0;
		ERROR_CHECK_STATUS(vxQueryArray((vx_array)ref, VX_ARRAY_ATTRIBUTE_ITEMSIZE, &itemsize, sizeof(itemsize)));
		if (itemsize == sizeof(StitchSeamFindPreference)) {
			status = VX_SUCCESS;
		}
		else {
			status = VX_ERROR_INVALID_DIMENSION;
			vxAddLogEntry((vx_reference)node, status, "ERROR: SeamFind array element (StitchSeamFindPreference) size should be 16 bytes\n");
		}
		ERROR_CHECK_STATUS(vxReleaseArray((vx_array *)&ref));
	}
	return status;
}

//! \brief The output validator callback.
static vx_status VX_CALLBACK seamfind_path_trace_output_validator(vx_node node, vx_uint32 index, vx_meta_format meta)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	vx_reference ref = avxGetNodeParamRef(node, index);

	if (index == 5)
	{ // array object of StitchSeamFindPathEntry type
		vx_size itemsize = 0; vx_size arr_capacity = 0;
		vx_enum itemtype;
		ERROR_CHECK_STATUS(vxQueryArray((vx_array)ref, VX_ARRAY_ATTRIBUTE_CAPACITY, &arr_capacity, sizeof(arr_capacity)));
		ERROR_CHECK_STATUS(vxQueryArray((vx_array)ref, VX_ARRAY_ATTRIBUTE_ITEMSIZE, &itemsize, sizeof(itemsize)));
		ERROR_CHECK_STATUS(vxQueryArray((vx_array)ref, VX_ARRAY_ATTRIBUTE_ITEMTYPE, &itemtype, sizeof(itemtype)));
		if (itemsize == sizeof(StitchSeamFindPathEntry)) {
			status = VX_SUCCESS;
		}
		else {
			status = VX_ERROR_INVALID_DIMENSION;
			vxAddLogEntry((vx_reference)node, status, "ERROR: SeamFind array element (StitchSeamFindPathEntry) size should be 4 bytes\n");
		}
		// set output image meta data
		ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_ARRAY_ATTRIBUTE_ITEMTYPE, &itemtype, sizeof(itemtype)));
		ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_ARRAY_ATTRIBUTE_CAPACITY, &arr_capacity, sizeof(arr_capacity)));
		ERROR_CHECK_STATUS(vxReleaseArray((vx_array *)&ref));
	}
	return status;
}

//! \brief The kernel target support callback.
static vx_status VX_CALLBACK seamfind_path_trace_query_target_support(vx_graph graph, vx_node node,
	vx_bool use_opencl_1_2,              // [input]  false: OpenCL driver is 2.0+; true: OpenCL driver is 1.2
	vx_uint32& supported_target_affinity // [output] must be set to AGO_TARGET_AFFINITY_CPU or AGO_TARGET_AFFINITY_GPU or (AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU)
	)
{
	char textBuffer[256];
	int SEAM_FIND_TARGET = 0;
	if (StitchGetEnvironmentVariable("SEAM_FIND_TARGET", textBuffer, sizeof(textBuffer))) { SEAM_FIND_TARGET = atoi(textBuffer); }

	if (!SEAM_FIND_TARGET)
		supported_target_affinity = AGO_TARGET_AFFINITY_GPU;
	else
		supported_target_affinity = AGO_TARGET_AFFINITY_CPU;

	return VX_SUCCESS;
}

//! \brief The OpenCL global work updater callback.
static vx_status VX_CALLBACK seamfind_path_trace_opencl_global_work_update(
	vx_node node,                                  // [input] node
	const vx_reference parameters[],               // [input] parameters
	vx_uint32 num,                                 // [input] number of parameters
	vx_uint32 opencl_work_dim,                     // [input] work_dim for clEnqueueNDRangeKernel()
	vx_size opencl_global_work[],                  // [output] global_work[] for clEnqueueNDRangeKernel()
	const vx_size opencl_local_work[]              // [input] local_work[] for clEnqueueNDRangeKernel()
	)
{
	// Get the number of elements in the array
	vx_size arr_numitems = 0;
	vx_array arr = (vx_array)avxGetNodeParamRef(node, 2);				// input array
	ERROR_CHECK_OBJECT(arr);
	ERROR_CHECK_STATUS(vxQueryArray(arr, VX_ARRAY_ATTRIBUTE_NUMITEMS, &arr_numitems, sizeof(arr_numitems)));
	ERROR_CHECK_STATUS(vxReleaseArray(&arr));
	opencl_global_work[0] = (arr_numitems + opencl_local_work[0] - 1) & ~(opencl_local_work[0] - 1);

	return VX_SUCCESS;
}

//! \brief The OpenCL code generator callback.
static vx_status VX_CALLBACK seamfind_path_trace_opencl_codegen(
	vx_node node,                                  // [input] node
	const vx_reference parameters[],               // [input] parameters
	vx_uint32 num,                                 // [input] number of parameters
	bool opencl_load_function,                     // [input]  false: normal OpenCL kernel; true: reserved
	char opencl_kernel_function_name[64],          // [output] kernel_name for clCreateKernel()
	std::string& opencl_kernel_code,               // [output] string for clCreateProgramWithSource()
	std::string& opencl_build_options,             // [output] options for clBuildProgram()
	vx_uint32& opencl_work_dim,                    // [output] work_dim for clEnqueueNDRangeKernel()
	vx_size opencl_global_work[],                  // [output] global_work[] for clEnqueueNDRangeKernel()
	vx_size opencl_local_work[],                   // [output] local_work[] for clEnqueueNDRangeKernel()
	vx_uint32& opencl_local_buffer_usage_mask,     // [output] reserved: must be ZERO
	vx_uint32& opencl_local_buffer_size_in_bytes   // [output] reserved: must be ZERO
	)
{
	// get input and output image configurations
	vx_size arr_capacity = 0;
	vx_array arr = (vx_array)avxGetNodeParamRef(node, 2);// input array
	ERROR_CHECK_OBJECT(arr);
	ERROR_CHECK_STATUS(vxQueryArray(arr, VX_ARRAY_ATTRIBUTE_CAPACITY, &arr_capacity, sizeof(arr_capacity)));
	ERROR_CHECK_STATUS(vxReleaseArray(&arr));

	// set kernel configuration
	vx_uint32 work_items = (vx_uint32)arr_capacity;
	strcpy(opencl_kernel_function_name, "seamfind_path_trace");
	opencl_work_dim = 1;
	opencl_local_work[0] = 64;
	opencl_global_work[0] = (work_items + opencl_local_work[0] - 1) & ~(opencl_local_work[0] - 1);

	// Setting variables required by the interface
	opencl_local_buffer_usage_mask = 0;
	opencl_local_buffer_size_in_bytes = 0;

	// kernel header and reading
	char item[8192];
	snprintf(item, sizeof(item),
		"#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
		"#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable\n"
		"\n"
		"__kernel __attribute__((reqd_work_group_size(%d, 1, 1)))\n" // opencl_local_work[0]
		"\n"
		"void %s(uint current_frame,\n"								 // opencl_kernel_function_name
		"						uint ip_weight_width, uint ip_weight_height, __global uchar * ip_weight_buf, uint ip_weight_stride, uint ip_weight_offset,\n"
		"						__global char * seam_info_buf, uint seam_info_buf_offset, uint seam_info_num_items,\n"
		"						__global char * seam_accum_buf, uint seam_accum_buf_offset, uint seam_accum_num_items,\n"
		"						__global char * seam_pref_buf, uint seam_pref_buf_offset, uint seam_pref_num_items,\n"
		"						__global char * seam_path_buf, uint seam_path_buf_offset, uint seam_path_num_items)\n"
		, (int)opencl_local_work[0], opencl_kernel_function_name);
	opencl_kernel_code = item;
	opencl_kernel_code +=
		"{\n"
		"\n"
		"int gid = get_global_id(0);\n"
		"\n"
		"	if (gid < seam_info_num_items)\n"
		"	{\n"
		"\n"
		"		seam_info_buf	+= seam_info_buf_offset + (gid * 16);\n"
		"		seam_accum_buf  =  seam_accum_buf + seam_accum_buf_offset;\n"
		"		seam_pref_buf	=  seam_pref_buf + seam_pref_buf_offset;\n"
		"		seam_path_buf	=  seam_path_buf + seam_path_buf_offset;\n"
		"\n"
		"		ip_weight_buf =  ip_weight_buf + ip_weight_offset;\n"
		"		uint equi_height = (ip_weight_width >> 1);\n"
		"\n"
		"		short8 info, pref;\n"
		"		info = vload8(0, (__global short *)seam_info_buf);\n"
		"		pref = vload8(0, (__global short *)&seam_pref_buf[gid * 16]);\n"
		"		uint offset_1 = (info.s0 * equi_height);\n"
		"		uint x_dir = (info.s3 - info.s2);\n"
		"		uint y_dir = (info.s5 - info.s4);\n"
		"		uint overlap_offset = ((info.s7 << 16) & 0xFFFF0000) | (info.s6  & 0x0000FFFF);\n"
		"\n"
		"		int min_x = -1, min_y = -1;\n"
		"		int min_cost = 0X7FFFFFFF;\n"
		"		int4 accum;\n"
		"\n"
		"		if (pref.s5 != -1 && ( (pref.s2 == current_frame) || ((current_frame + 1) % (pref.s3 + pref.s1) == 0)))\n"
		"		{\n"
		"/*			Vertical Seam */\n"
		"			if (y_dir >= x_dir)\n"
		"			{\n"
#if ENABLE_VERTICAL_SEAM
		"\n"
		"				int ye = (int)info.s5 - 1;\n"
		"				min_y = ye;\n"
		"				short p_x = -1, p_y = -1;\n"
		"				for (int xe = (int)info.s3 - 1; xe >= (int)info.s2; xe--)\n"
		"				{\n"
		"					uint pixel_id = overlap_offset + ((ye - info.s4) * x_dir) + (xe - info.s2);\n"
		"					accum = vload4(0, (__global int *)&seam_accum_buf[pixel_id * 12]); \n"
		"\n"
		"					if ((min_cost > accum.s1) && accum.s2)\n"
		"					{\n"
		"						p_x = (short)(accum.s0 & 0x0000FFFF);\n"
		"						p_y = (short)((accum.s0 & 0xFFFF0000) >> 16);\n"
		"						min_cost =  accum.s1;\n"
		"						min_x =  xe;\n"
		"					}\n"
		"\n"
		"				}\n"
		"\n"
		"				uint min_path_start = overlap_offset + ((min_y - info.s4) * x_dir) + (min_x - info.s2);\n"
		"				uint path_offset = (gid * ip_weight_width);\n"
		"\n"
		"				int i_val = 0;\n"
		"				uint weight_pixel_check = (((info.s5 - 1)/2 + offset_1) * ip_weight_width) + info.s3 - 1;\n"
		"				uchar weight_i =	 *(__global uchar *)&ip_weight_buf[weight_pixel_check];\n"
		"				if (weight_i) i_val = 255;\n"
		"\n"
		"				while ((p_x != -1 || p_y != -1) && ( p_x != 0 && p_y != 0))\n"
		"				{\n"
		"\n"
		"					uint path_id = min_y + path_offset;\n"
		"					short2 val;\n"
		"					val.s0 = min_x; val.s1 = i_val;\n"
		"					*(__global short2 *) &seam_path_buf[path_id * 4] = val; \n"
		"\n"
		"					min_y--;\n"
		"					min_x = p_x;\n"
		"\n"
		"					min_path_start = overlap_offset + ((min_y - info.s4) * x_dir) + (min_x - info.s2);\n"
		"					accum = vload4(0, (__global int *)&seam_accum_buf[min_path_start * 12]); \n"
		"					p_x = (accum.s0 & 0x0000FFFF);\n"
		"					p_y = ((accum.s0 & 0xFFFF0000) >> 16);\n"
		"\n"
		"					if ((p_x > min_x + 1) || (p_x < min_x - 1)){ p_x = min_x - 1;}\n"
		"				}\n"
		"\n"
#endif
		"			}\n"
		"\n";
	opencl_kernel_code +=
		"//Horizontal Seam\n"
		"			else if(x_dir > y_dir)\n"
		"			{\n"
#if ENABLE_HORIZONTAL_SEAM
		"\n"
		"				int xe = (int)info.s3 - 1;\n"
		"				min_x = xe;\n"
		"				short p_x = -1, p_y = -1;\n"
		"				for (int ye = (int)info.s5-1; ye >= (int)info.s4; ye--)\n"
		"				{\n"
		"					uint pixel_id = overlap_offset + ((xe - info.s2) * y_dir) + (ye - info.s4);\n"
		"					accum = vload4(0, (__global int *)&seam_accum_buf[pixel_id * 12]); \n"
		"\n"
		"					if (min_cost > accum.s1 && accum.s2)\n"
		"					{\n"
		"						p_x = (accum.s0 & 0x0000FFFF);\n"
		"						p_y = ((accum.s0 & 0xFFFF0000) >> 16);\n"
		"						min_cost =  accum.s1;\n"
		"						min_y =  ye;\n"
		"					}\n"
		"\n"
		"				}\n"
		"\n"
		"				uint min_path_start = overlap_offset + ((min_x - info.s2) * y_dir) + (min_y - info.s4);\n"
		"				uint path_offset = (gid * ip_weight_width);\n"
		"\n"
		"				int i_val = 0;\n"
		"				uint weight_pixel_check = ((info.s5 + offset_1 - 1) * ip_weight_width) + info.s3 - 1;\n"
		"				uchar weight_i =	 *(__global uchar *)&ip_weight_buf[weight_pixel_check];\n"
		"				if (weight_i) i_val = 255;\n"
		"\n"
		"				while (p_x != -1 && ( p_x != 0 && p_y != 0))\n"
		"				{\n"
		"					uint path_id = min_x + path_offset;\n"
		"					short2 val;\n"
		"					val.s0 = min_y; val.s1 = i_val;\n"
		"					*(__global short2 *) &seam_path_buf[path_id * 4] = val; \n"
		"\n"
		"					min_x--;\n"
		"					min_y = p_y;\n"
		"\n"
		"					min_path_start = overlap_offset + ((min_x - info.s2) * y_dir) + (min_y - info.s4);\n"
		"					accum = vload4(0, (__global int *)&seam_accum_buf[min_path_start * 12]); \n"
		"					p_x = (accum.s0 & 0x0000FFFF);\n"
		"					p_y = ((accum.s0 & 0xFFFF0000) >> 16);\n"
		"\n"
		"					if ((p_y > min_y + 1) || (p_y < min_y - 1)){ p_y = min_y - 1; }\n"
		"\n"
		"				}\n"
		"\n"
#endif
		"\n"
		"			}\n";
	opencl_kernel_code +=
		"		}\n"
		"	}\n"
		"}\n";

	return VX_SUCCESS;
}

//! \brief The kernel execution.
static vx_status VX_CALLBACK seamfind_path_trace_kernel(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
	//Number Of Cameras - Variable 0 
	vx_uint32 current_frame = 0, width_eqr = 0, height_eqr = 0;
	ERROR_CHECK_STATUS(vxReadScalarValue((vx_scalar)parameters[0], &current_frame));

	//Input Weight image - Variable 1
	vx_image weight_image = (vx_image)parameters[1];
	vx_uint32 width = 0, height = 0, plane = 0;
	ERROR_CHECK_STATUS(vxQueryImage(weight_image, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
	ERROR_CHECK_STATUS(vxQueryImage(weight_image, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
	void *weight_image_ptr = nullptr; vx_rectangle_t weight_rect;	vx_imagepatch_addressing_t weight_addr;
	weight_rect.start_x = weight_rect.start_y = 0; weight_rect.end_x = width; weight_rect.end_y = height;
	ERROR_CHECK_STATUS(vxAccessImagePatch(weight_image, &weight_rect, plane, &weight_addr, &weight_image_ptr, VX_READ_ONLY));
	vx_uint8 *weight_ptr = (vx_uint8*)weight_image_ptr;

	width_eqr = width;
	height_eqr = (width_eqr >> 1);

	//SeamFindInfo Array - Variable 2
	vx_size arr_numitems = 0;
	vx_array SeamFindInfo = (vx_array)parameters[2];
	ERROR_CHECK_STATUS(vxQueryArray(SeamFindInfo, VX_ARRAY_ATTRIBUTE_NUMITEMS, &arr_numitems, sizeof(arr_numitems)));
	StitchSeamFindInformation *SeamFindInfo_ptr = nullptr;
	vx_size stride = sizeof(StitchSeamFindInformation);
	ERROR_CHECK_STATUS(vxAccessArrayRange(SeamFindInfo, 0, arr_numitems, &stride, (void **)&SeamFindInfo_ptr, VX_READ_ONLY));

	//Seam Find Accum Array - Variable 3
	vx_array Array_SeamFind_ACCUM = (vx_array)parameters[3];
	vx_size SeamFind_ACCUM_max = 0;
	ERROR_CHECK_STATUS(vxQueryArray(Array_SeamFind_ACCUM, VX_ARRAY_ATTRIBUTE_NUMITEMS, &SeamFind_ACCUM_max, sizeof(SeamFind_ACCUM_max)));
	StitchSeamFindAccumEntry *SeamFind_Accum = nullptr;
	vx_size stride_accum = sizeof(StitchSeamFindAccumEntry);
	ERROR_CHECK_STATUS(vxAccessArrayRange(Array_SeamFind_ACCUM, 0, SeamFind_ACCUM_max, &stride_accum, (void **)&SeamFind_Accum, VX_READ_ONLY));

	//Seam Find Accum Array - Variable 4
	vx_array Array_SeamFind_Pref = (vx_array)parameters[4];
	vx_size SeamFind_Pref_max = 0;
	ERROR_CHECK_STATUS(vxQueryArray(Array_SeamFind_Pref, VX_ARRAY_ATTRIBUTE_NUMITEMS, &SeamFind_Pref_max, sizeof(SeamFind_Pref_max)));
	StitchSeamFindPreference *SeamFind_Pref = nullptr;
	vx_size stride_pref = sizeof(StitchSeamFindPreference);
	ERROR_CHECK_STATUS(vxAccessArrayRange(Array_SeamFind_Pref, 0, SeamFind_Pref_max, &stride_pref, (void **)&SeamFind_Pref, VX_READ_ONLY));

	std::vector<StitchSeamFindPathEntry> SeamFind_Path;
	vx_size  path_array_size = (vx_size)(width_eqr * arr_numitems);
	SeamFind_Path.resize(path_array_size);

	//Horizontal Overlap Counter
	vx_uint32 horizontal_overlap = 1;

	//Loop over all the overlaps
	for (vx_uint32 i = 0; i < arr_numitems; i++)
	{
		vx_uint32 offset_1 = SeamFindInfo_ptr[i].cam_id_1 * height_eqr;
		vx_uint32 offset_2 = SeamFindInfo_ptr[i].cam_id_2 * height_eqr;
		int y_dir = SeamFindInfo_ptr[i].end_y - SeamFindInfo_ptr[i].start_y;
		int x_dir = SeamFindInfo_ptr[i].end_x - SeamFindInfo_ptr[i].start_x;

		vx_int32 min_cost = 0X7FFFFFFF;
		vx_int32 min_x = -1, min_y = -1;

		if (SeamFind_Pref[i].priority != -1 && ((SeamFind_Pref[i].start_frame == current_frame) || ((current_frame + 1) % (SeamFind_Pref[i].frequency + SeamFind_Pref[i].seam_type_num) == 0)))
		{
			/***********************************************************************************************************************************
			Vertical SeamCut
			************************************************************************************************************************************/
			if (y_dir >= x_dir)
			{
#if ENABLE_VERTICAL_SEAM

#if GET_TIMING
				int64_t start_path_t = stitchGetClockCounter();
#endif
				//Select the least cost pixel for the start of the seam
				vx_uint32 ye = SeamFindInfo_ptr[i].end_y;
				min_y = ye;

				for (vx_int32 xe = (vx_int32)SeamFindInfo_ptr[i].end_x; xe >= (vx_int32)SeamFindInfo_ptr[i].start_x; xe--)
				{
					vx_uint32 pixel_id = SeamFindInfo_ptr[i].offset + ((ye - SeamFindInfo_ptr[i].start_y) * x_dir) + (xe - SeamFindInfo_ptr[i].start_x);
					if ((min_cost > SeamFind_Accum[pixel_id].value))
					{
						min_cost = SeamFind_Accum[pixel_id].value;
						min_x = xe;
					}
				}
#if GET_TIMING
				int64_t end_path_t = stitchGetClockCounter();
				int64_t freq = stitchGetClockFrequency();
				float factor = 1000.0f / (float)freq; // to convert clock counter to ms
				float Path_find_time = (float)((end_path_t - start_path_t) * factor);
				int64_t start_path_traverse = stitchGetClockCounter();
#endif
				//Selected Min Path 
				vx_uint32 min_path_start = SeamFindInfo_ptr[i].offset + ((min_y - SeamFindInfo_ptr[i].start_y) * x_dir) + (min_x - SeamFindInfo_ptr[i].start_x);
				vx_uint32 path_offset = (i * width_eqr);

				//Selected Weight at End-X for Image i 
				int i_val = 0;
				vx_uint32 weight_pixel_check = ((SeamFindInfo_ptr[i].end_y + offset_1) * width_eqr) + SeamFindInfo_ptr[i].end_x;
				if (weight_ptr[weight_pixel_check] == 255) i_val = 255;

				//Traverse the path to obtain the seam			
				while ((SeamFind_Accum[min_path_start].parent_x != -1 || SeamFind_Accum[min_path_start].parent_y != -1) && (SeamFind_Accum[min_path_start].parent_x != 0 && SeamFind_Accum[min_path_start].parent_y != 0))
				{
					vx_uint32 path_id = min_y + path_offset;
					SeamFind_Path[path_id].min_pixel = min_x;
					SeamFind_Path[path_id].weight_value_i = i_val;

					min_y--;
					min_x = SeamFind_Accum[min_path_start].parent_x;
					min_path_start = SeamFindInfo_ptr[i].offset + ((SeamFind_Accum[min_path_start].parent_y - SeamFindInfo_ptr[i].start_y) * x_dir) + (SeamFind_Accum[min_path_start].parent_x - SeamFindInfo_ptr[i].start_x);
				}
#if GET_TIMING
				int64_t end_path_traverse = stitchGetClockCounter();
				float Path_travese_time = (float)((end_path_traverse - start_path_traverse) * factor);
				printf("Overlap::%d,%d:::Best Path Find Time-->%f (ms) Path Traverse Time--> %f (ms) \n", i, j, Path_find_time, Path_travese_time);
#endif

#endif
			}
			/***********************************************************************************************************************************
			Horizontal SeamCut
			************************************************************************************************************************************/
			else if (x_dir > y_dir)
			{
#if ENABLE_HORIZONTAL_SEAM

#if GET_TIMING
				int64_t start_path_t = stitchGetClockCounter();
#endif
				//Select the least cost pixel for the start of the seam
				vx_uint32 xe = SeamFindInfo_ptr[i].end_x;
				min_x = xe;

				for (vx_int32 ye = (vx_int32)SeamFindInfo_ptr[i].end_y; ye >= (vx_int32)SeamFindInfo_ptr[i].start_y; ye--)
				{
					vx_uint32 pixel_id = SeamFindInfo_ptr[i].offset + ((xe - SeamFindInfo_ptr[i].start_x) * y_dir) + (ye - SeamFindInfo_ptr[i].start_y);
					if ((min_cost > SeamFind_Accum[pixel_id].value))
					{
						min_cost = SeamFind_Accum[pixel_id].value;
						min_y = ye;
					}
				}
#if GET_TIMING
				int64_t end_path_t = stitchGetClockCounter();
				int64_t freq = stitchGetClockFrequency();
				float factor = 1000.0f / (float)freq; // to convert clock counter to ms
				float Path_find_time = (float)((end_path_t - start_path_t) * factor);
				int64_t start_path_traverse = stitchGetClockCounter();
#endif
				//Selected Min Path 
				vx_uint32 min_path_start = SeamFindInfo_ptr[i].offset + ((min_x - SeamFindInfo_ptr[i].start_x) * y_dir) + (min_y - SeamFindInfo_ptr[i].start_y);
				vx_uint32 path_offset = (i * width_eqr);

				//Selected Weight at End-X for Image i 
				int i_val = 0;
				vx_uint32 weight_pixel_check = ((min_y + offset_1) * width_eqr) + SeamFindInfo_ptr[i].end_x;
				if (weight_ptr[weight_pixel_check] == 255) i_val = 255;

				//Traverse the path to obtain the seam			
				while ((SeamFind_Accum[min_path_start].parent_x != -1 || SeamFind_Accum[min_path_start].parent_y != -1) && (SeamFind_Accum[min_path_start].parent_x != 0 && SeamFind_Accum[min_path_start].parent_y != 0))
				{
					vx_uint32 path_id = min_x + path_offset;
					SeamFind_Path[path_id].min_pixel = min_y;
					SeamFind_Path[path_id].weight_value_i = i_val;

					min_x--;
					min_y = SeamFind_Accum[min_path_start].parent_y;

					min_path_start = SeamFindInfo_ptr[i].offset + ((min_x - SeamFindInfo_ptr[i].start_x) * y_dir) + (min_y - SeamFindInfo_ptr[i].start_y);
				}

#if GET_TIMING
				int64_t end_path_traverse = stitchGetClockCounter();
				float Path_travese_time = (float)((end_path_traverse - start_path_traverse) * factor);
				printf("Overlap::%d,%d:::Best Path Find Time-->%f (ms) Path Traverse Time--> %f (ms) \n", i, j, Path_find_time, Path_travese_time);
#endif
				horizontal_overlap++;
#endif
			}
		}
	}
	vx_array accum_seamFindPathEntry = (vx_array)parameters[5];
	vx_size seamcut_path_size = width_eqr * arr_numitems;
	StitchSeamFindPathEntry *StitchSeamCutPath_ptr = &SeamFind_Path[0];
	ERROR_CHECK_STATUS(vxTruncateArray(accum_seamFindPathEntry, 0));
	ERROR_CHECK_STATUS(vxAddArrayItems(accum_seamFindPathEntry, seamcut_path_size, StitchSeamCutPath_ptr, sizeof(StitchSeamFindPathEntry)));

	ERROR_CHECK_STATUS(vxCommitImagePatch(weight_image, &weight_rect, 0, &weight_addr, weight_image_ptr));
	ERROR_CHECK_STATUS(vxCommitArrayRange(SeamFindInfo, 0, arr_numitems, SeamFindInfo_ptr));
	ERROR_CHECK_STATUS(vxCommitArrayRange(Array_SeamFind_ACCUM, 0, SeamFind_ACCUM_max, SeamFind_Accum));
	ERROR_CHECK_STATUS(vxCommitArrayRange(Array_SeamFind_Pref, 0, SeamFind_Pref_max, SeamFind_Pref));

	//Release Memory
	SeamFind_Path.clear();

	return VX_SUCCESS;
}

//! \brief The kernel publisher.
vx_status seamfind_path_trace_publish(vx_context context)
{
	// add kernel to the context with callbacks
	vx_kernel kernel = vxAddKernel(context, "com.amd.loomsl.seamfind_path_trace",
		AMDOVX_KERNEL_STITCHING_SEAMFIND_PATH_TRACE,
		seamfind_path_trace_kernel,
		6,
		seamfind_path_trace_input_validator,
		seamfind_path_trace_output_validator,
		nullptr,
		nullptr);
	ERROR_CHECK_OBJECT(kernel);
	amd_kernel_query_target_support_f query_target_support_f = seamfind_path_trace_query_target_support;
	amd_kernel_opencl_codegen_callback_f opencl_codegen_callback_f = seamfind_path_trace_opencl_codegen;
	amd_kernel_opencl_global_work_update_callback_f opencl_global_work_update_callback_f = seamfind_path_trace_opencl_global_work_update;
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_CODEGEN_CALLBACK, &opencl_codegen_callback_f, sizeof(opencl_codegen_callback_f)));
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_GLOBAL_WORK_UPDATE_CALLBACK, &opencl_global_work_update_callback_f, sizeof(opencl_global_work_update_callback_f)));

	// set kernel parameters
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 5, VX_OUTPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));

	// finalize and release kernel object
	ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
	ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

	return VX_SUCCESS;
}


/***********************************************************************************************************************************

Seam Find Kernel - 5 --- Set Weight - GPU

************************************************************************************************************************************/
//! \brief The input validator callback.
static vx_status VX_CALLBACK seamfind_set_weights_input_validator(vx_node node, vx_uint32 index)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	// get reference for parameter at specified index
	vx_reference ref = avxGetNodeParamRef(node, index);
	ERROR_CHECK_OBJECT(ref);
	// validate each parameter
	if (index == 0 || index == 1 || index == 2 || index == 3 || index == 8)
	{ // object of SCALAR type
		vx_enum itemtype = VX_TYPE_INVALID;
		ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)ref, VX_SCALAR_ATTRIBUTE_TYPE, &itemtype, sizeof(itemtype)));

		if (itemtype == VX_TYPE_UINT32) {
			status = VX_SUCCESS;
		}
		else {
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: SeamFind scalar parameter type should be a UINT32\n");
		}
		ERROR_CHECK_STATUS(vxReleaseScalar((vx_scalar *)&ref));
	}
	else if (index == 4)
	{ // array object of StitchSeamFindWeightEntry type
		vx_size itemsize = 0;
		ERROR_CHECK_STATUS(vxQueryArray((vx_array)ref, VX_ARRAY_ATTRIBUTE_ITEMSIZE, &itemsize, sizeof(itemsize)));
		if (itemsize == sizeof(StitchSeamFindWeightEntry)) {
			status = VX_SUCCESS;
		}
		else {
			status = VX_ERROR_INVALID_DIMENSION;
			vxAddLogEntry((vx_reference)node, status, "ERROR: SeamFind array element (StitchSeamFindWeightEntry) size should be 12 bytes\n");
		}
		ERROR_CHECK_STATUS(vxReleaseArray((vx_array *)&ref));
	}
	else if (index == 5)
	{ // array object of StitchSeamFindPathEntry type
		vx_size itemsize = 0;
		ERROR_CHECK_STATUS(vxQueryArray((vx_array)ref, VX_ARRAY_ATTRIBUTE_ITEMSIZE, &itemsize, sizeof(itemsize)));
		if (itemsize == sizeof(StitchSeamFindPathEntry)) {
			status = VX_SUCCESS;
		}
		else {
			status = VX_ERROR_INVALID_DIMENSION;
			vxAddLogEntry((vx_reference)node, status, "ERROR: SeamFind array element (StitchSeamFindPathEntry) size should be 4 bytes\n");
		}
		ERROR_CHECK_STATUS(vxReleaseArray((vx_array *)&ref));
	}
	else if (index == 6)
	{ // array object of StitchSeamFindPreference type
		vx_size itemsize = 0;
		ERROR_CHECK_STATUS(vxQueryArray((vx_array)ref, VX_ARRAY_ATTRIBUTE_ITEMSIZE, &itemsize, sizeof(itemsize)));
		if (itemsize == sizeof(StitchSeamFindPreference)) {
			status = VX_SUCCESS;
		}
		else {
			status = VX_ERROR_INVALID_DIMENSION;
			vxAddLogEntry((vx_reference)node, status, "ERROR: SeamFind array element (StitchSeamFindPreference) size should be 14 bytes\n");
		}
		ERROR_CHECK_STATUS(vxReleaseArray((vx_array *)&ref));
	}

	return status;
}

//! \brief The output validator callback.
static vx_status VX_CALLBACK seamfind_set_weights_output_validator(vx_node node, vx_uint32 index, vx_meta_format meta)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	if (index == 7)
	{ // Image object	
		//Query Weight Image
		vx_int32 width_img = 0, height_img = 0;
		vx_df_image format = VX_DF_IMAGE_VIRT;
		vx_image image = (vx_image)avxGetNodeParamRef(node, index);
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &width_img, sizeof(width_img)));
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &height_img, sizeof(height_img)));
		if (format != VX_DF_IMAGE_U8)
			status = VX_ERROR_INVALID_FORMAT;

		else if (width_img < 0)
			status = VX_ERROR_INVALID_DIMENSION;

		else if (height_img < 0)
			status = VX_ERROR_INVALID_DIMENSION;
		else
		{
			// set output image data
			ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_WIDTH, &width_img, sizeof(width_img)));
			ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_HEIGHT, &height_img, sizeof(height_img)));
			ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));
			status = VX_SUCCESS;
		}
		ERROR_CHECK_STATUS(vxReleaseImage(&image));
	}
	return status;
}

//! \brief The kernel initialize.
static vx_status VX_CALLBACK seamfind_set_weights_initialize(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
	return VX_SUCCESS;
}
//! \brief The kernel deinitialize.
static vx_status VX_CALLBACK seamfind_set_weights_deinitialize(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
	return VX_SUCCESS;
}

//! \brief The kernel execution.
static vx_status VX_CALLBACK seamfind_set_weights_kernel(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
	return VX_ERROR_NOT_IMPLEMENTED;
}

//! \brief The kernel target support callback.
static vx_status VX_CALLBACK seamfind_set_weights_query_target_support(vx_graph graph, vx_node node,
	vx_bool use_opencl_1_2,              // [input]  false: OpenCL driver is 2.0+; true: OpenCL driver is 1.2
	vx_uint32& supported_target_affinity // [output] must be set to AGO_TARGET_AFFINITY_CPU or AGO_TARGET_AFFINITY_GPU or (AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU)
	)
{
	supported_target_affinity = AGO_TARGET_AFFINITY_GPU;
	return VX_SUCCESS;
}

//! \brief The OpenCL global work updater callback.
static vx_status VX_CALLBACK seamfind_set_weights_opencl_global_work_update(
	vx_node node,                                  // [input] node
	const vx_reference parameters[],               // [input] parameters
	vx_uint32 num,                                 // [input] number of parameters
	vx_uint32 opencl_work_dim,                     // [input] work_dim for clEnqueueNDRangeKernel()
	vx_size opencl_global_work[],                  // [output] global_work[] for clEnqueueNDRangeKernel()
	const vx_size opencl_local_work[]              // [input] local_work[] for clEnqueueNDRangeKernel()
	)
{
	// Get the number of elements in the array
	vx_size arr_numitems = 0;
	vx_array arr = (vx_array)avxGetNodeParamRef(node, 4);				// input array
	ERROR_CHECK_OBJECT(arr);
	ERROR_CHECK_STATUS(vxQueryArray(arr, VX_ARRAY_ATTRIBUTE_NUMITEMS, &arr_numitems, sizeof(arr_numitems)));
	ERROR_CHECK_STATUS(vxReleaseArray(&arr));

	opencl_global_work[0] = (arr_numitems + opencl_local_work[0] - 1) & ~(opencl_local_work[0] - 1);

	return VX_SUCCESS;
}

//! \brief The OpenCL code generator callback.
static vx_status VX_CALLBACK seamfind_set_weights_opencl_codegen(
	vx_node node,                                  // [input] node
	const vx_reference parameters[],               // [input] parameters
	vx_uint32 num,                                 // [input] number of parameters
	bool opencl_load_function,                     // [input]  false: normal OpenCL kernel; true: reserved
	char opencl_kernel_function_name[64],          // [output] kernel_name for clCreateKernel()
	std::string& opencl_kernel_code,               // [output] string for clCreateProgramWithSource()
	std::string& opencl_build_options,             // [output] options for clBuildProgram()
	vx_uint32& opencl_work_dim,                    // [output] work_dim for clEnqueueNDRangeKernel()
	vx_size opencl_global_work[],                  // [output] global_work[] for clEnqueueNDRangeKernel()
	vx_size opencl_local_work[],                   // [output] local_work[] for clEnqueueNDRangeKernel()
	vx_uint32& opencl_local_buffer_usage_mask,     // [output] reserved: must be ZERO
	vx_uint32& opencl_local_buffer_size_in_bytes   // [output] reserved: must be ZERO
	)
{
	// get input and output image configurations
	vx_size arr_capacity = 0;
	vx_array arr = (vx_array)avxGetNodeParamRef(node, 4);	// input array
	ERROR_CHECK_OBJECT(arr);
	ERROR_CHECK_STATUS(vxQueryArray(arr, VX_ARRAY_ATTRIBUTE_CAPACITY, &arr_capacity, sizeof(arr_capacity)));
	ERROR_CHECK_STATUS(vxReleaseArray(&arr));

	// get debug flags
	vx_uint32 debugFlags = 0;
	int DRAW_SEAM = 0, VIEW_SCENE_CHANGE = 0, SHOW_ALL_SEAMS = 0;
	ERROR_CHECK_STATUS(vxReadScalarValue((vx_scalar)parameters[8], &debugFlags));
	DRAW_SEAM = (debugFlags >> 8) & 1;
	VIEW_SCENE_CHANGE = (debugFlags >> 9) & 1;
	SHOW_ALL_SEAMS = (debugFlags >> 10) & 1;
	if (SHOW_ALL_SEAMS)
		DRAW_SEAM = 1;

	// set kernel configuration
	vx_uint32 work_items = (vx_uint32)arr_capacity;
	strcpy(opencl_kernel_function_name, "seamfind_set_weights");
	opencl_work_dim = 1;
	opencl_local_work[0] = 128;
	opencl_global_work[0] = (work_items + opencl_local_work[0] - 1) & ~(opencl_local_work[0] - 1);

	// Setting variables required by the interface
	opencl_local_buffer_usage_mask = 0;
	opencl_local_buffer_size_in_bytes = 0;

	// kernel header and reading
	char item[8192];
	snprintf(item, sizeof(item),
		"#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
		"#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable\n"
		"\n"
		"__kernel __attribute__((reqd_work_group_size(%d, 1, 1)))\n" // opencl_local_work[0]
		"\n"
		"void %s(uint current_frame,\n" // opencl_kernel_function_name
		"		 uint NumCam, uint equi_width, uint equi_height,\n"
		"        __global char * seam_valid_buf, uint seam_valid_buf_offset, uint valid_pix_num_items,\n"
		"        __global char * path_buf, uint path_buf_offset, uint path_num_items,\n"
		"		 __global char * seam_pref_buf, uint seam_pref_buf_offset, uint seam_pref_num_items,\n"
		"        uint weight_width, uint weight_height, __global uchar * weight_buf, uint weight_stride, uint weight_offset, uint flags)\n"
		, (int)opencl_local_work[0], opencl_kernel_function_name);
	opencl_kernel_code = item;
	opencl_kernel_code +=
		"{\n"
		"	int gid = get_global_id(0);\n"
		"\n"
		"		if (gid < valid_pix_num_items)\n"
		"		{\n"
		"\n"
		"			seam_valid_buf += seam_valid_buf_offset + (gid * 12);\n"
		"			path_buf =  path_buf + path_buf_offset;\n"
		"			seam_pref_buf =  seam_pref_buf + seam_pref_buf_offset;\n"
		"			weight_buf =  weight_buf + weight_offset;\n"
		"\n"
		"			short8 dim, pref;\n"
		"			dim = vload8(0, (__global short *)seam_valid_buf);\n"
		"			pref = vload8(0, (__global short *)&seam_pref_buf[dim.s4 * 16]);\n"
		"			short2 path;\n"
		"\n"
		"			if (pref.s5 != -1 && ( (pref.s2 == current_frame) || ((current_frame + 1) % (pref.s3 + pref.s1) == 0)))\n"
		"			{\n"
		"				uint offset_1 = dim.s2 * equi_height;\n"
		"				uint offset_2 = dim.s3 * equi_height;\n"
		"\n"
		"				uint ID1 = ((dim.s1 + offset_1) * equi_width) + dim.s0;\n"
		"				uint ID2 = ((dim.s1 + offset_2) * equi_width) + dim.s0;\n"
		"\n"
		"				if(dim.s5 == 0)\n"// Vertical Seams
		"				{\n"
#if ENABLE_VERTICAL_SEAM
		"					uint overlap_ID = dim.s1 + (dim.s4 * equi_width);\n"
		"					path = vload2(0, (__global short *)&path_buf[overlap_ID << 2]);\n"
		"\n"
		"					uchar i_val_start, j_val_start, i_val_end, j_val_end;\n"
		"\n"
		"					if (path.s1 == 255)\n"
		"					{\n"
		"						i_val_start = 255; j_val_start = 0;\n"
		"						i_val_end = 0; j_val_end = 255;\n"
		"					}\n"
		"					else\n"
		"					{\n"
		"						i_val_start = 0; j_val_start = 255;\n"
		"						i_val_end = 255; j_val_end = 0;\n"
		"					}\n"
		"\n";
	if (SHOW_ALL_SEAMS)
	{
		opencl_kernel_code +=
			"					if (0)\n"
			"					{\n";
	}
	opencl_kernel_code +=
		"\n"
		"					if (dim.s0 >= path.s0)\n"
		"					{\n"
		"						*(__global uchar *) &weight_buf[ID1] = i_val_start;\n"
		"						*(__global uchar *) &weight_buf[ID2] = j_val_start;\n"
		"\n";
	if (VIEW_SCENE_CHANGE)
	{
		opencl_kernel_code +=
			"			             if(pref.s7 == 2)\n"
			"						 {\n"
			"							*(__global uchar *) &weight_buf[ID1] = 50;\n"
			"							*(__global uchar *) &weight_buf[ID2] = 50;\n"
			"						 }\n"
			"			             else if(pref.s7 == 3)\n"
			"						 {\n"
			"							*(__global uchar *) &weight_buf[ID1] = 255;\n"
			"							*(__global uchar *) &weight_buf[ID2] = 255;\n"
			"						 }\n";
	}
	opencl_kernel_code +=
		"\n"
		"							for(int cam = 0; cam < NumCam; cam ++)\n"
		"								if(cam != dim.s2 && cam != dim.s3)\n"
		"								{\n"
		"									uint offset_pixel = cam * equi_height;\n"
		"									uint ID_PIXEL = ((dim.s1 + offset_pixel) * equi_width) + dim.s0;\n"
		"									*(__global uchar *)&weight_buf[ID_PIXEL] = 0;\n"
		"								}\n"
		"\n"
		"					}\n"
		"					else\n"
		"					{\n"
		"						*(__global uchar *) &weight_buf[ID1] = i_val_end;\n"
		"						*(__global uchar *) &weight_buf[ID2] = j_val_end;\n"
		"\n";
	if (VIEW_SCENE_CHANGE)
	{
		opencl_kernel_code +=
			"			             if(pref.s7 == 2)\n"
			"						 {\n"
			"							*(__global uchar *) &weight_buf[ID1] = 50;\n"
			"							*(__global uchar *) &weight_buf[ID2] = 50;\n"
			"						 }\n"
			"			             else if(pref.s7 == 3)\n"
			"						 {\n"
			"							*(__global uchar *) &weight_buf[ID1] = 255;\n"
			"							*(__global uchar *) &weight_buf[ID2] = 255;\n"
			"						 }\n";
	}
	opencl_kernel_code +=
		"\n"
		"						for(int cam = 0; cam < NumCam; cam ++)\n"
		"							if(cam != dim.s2 && cam != dim.s3)\n"
		"							{\n"
		"								uint offset_pixel = cam * equi_height;\n"
		"								uint ID_PIXEL = ((dim.s1 + offset_pixel) * equi_width) + dim.s0;\n"
		"								*(__global uchar *)&weight_buf[ID_PIXEL] = 0;\n"
		"							}\n"
		"\n"
		"					}\n"
		"\n";
	if (SHOW_ALL_SEAMS)
	{
		opencl_kernel_code +=
			"				}\n";
	}
	opencl_kernel_code +=
		"\n"
#endif
		"				}\n"
		"\n";
	opencl_kernel_code +=
		"				else if(dim.s5 == 1)\n"// Horizontal Seams
		"				{\n"
#if ENABLE_HORIZONTAL_SEAM
		"					uint overlap_ID = dim.s0 + (dim.s4 * equi_width);\n"
		"					path = vload2(0, (__global short *)&path_buf[overlap_ID << 2]);\n"
		"\n"
		"					uchar i_val_start = 0, j_val_start = 0, i_val_end = 0, j_val_end = 0;\n"
		"\n"
		"					if (path.s1 == 255)\n"
		"					{\n"
		"						i_val_start = 255; j_val_start = 0;\n"
		"						i_val_end = 0; j_val_end = 255;\n"
		"					}\n"
		"					else\n"
		"					{\n"
		"						i_val_start = 0; j_val_start = 255;\n"
		"						i_val_end = 255; j_val_end = 0;\n"
		"					}\n"
		"\n";
	if (SHOW_ALL_SEAMS)
	{
		opencl_kernel_code +=
			"					if (0)\n"
			"					{\n";
	}
	opencl_kernel_code +=
		"					if (dim.s1 >= path.s0)\n"
		"					{\n"
		"						*(__global uchar *) &weight_buf[ID1] = i_val_start;\n"
		"						*(__global uchar *) &weight_buf[ID2] = j_val_start;\n"
		"\n";
	if (VIEW_SCENE_CHANGE)
	{
		opencl_kernel_code +=
			"			             if(pref.s7 == 2)\n"
			"						 {\n"
			"							*(__global uchar *) &weight_buf[ID1] = 50;\n"
			"							*(__global uchar *) &weight_buf[ID2] = 50;\n"
			"						 }\n"
			"			             else if(pref.s7 == 3)\n"
			"						 {\n"
			"							*(__global uchar *) &weight_buf[ID1] = 255;\n"
			"							*(__global uchar *) &weight_buf[ID2] = 255;\n"
			"						 }\n";
	}
	opencl_kernel_code +=
		"\n"
		"\n"
		"						for(int cam = 0; cam < NumCam; cam ++)\n"
		"							if(cam != dim.s2 && cam != dim.s3)\n"
		"							{\n"
		"								uint offset_pixel = cam * equi_height;\n"
		"								uint ID_PIXEL = ((dim.s1 + offset_pixel) * equi_width) + dim.s0;\n"
		"								*(__global uchar *)&weight_buf[ID_PIXEL] = 0;\n"
		"							}\n"
		"\n"
		"					}\n"
		"					else\n"
		"					{\n"
		"						*(__global uchar *) &weight_buf[ID1] = i_val_end;\n"
		"						*(__global uchar *) &weight_buf[ID2] = j_val_end;\n"
		"\n"
		"\n";
	if (VIEW_SCENE_CHANGE)
	{
		opencl_kernel_code +=
			"			             if(pref.s7 == 2)\n"
			"						 {\n"
			"							*(__global uchar *) &weight_buf[ID1] = 50;\n"
			"							*(__global uchar *) &weight_buf[ID2] = 50;\n"
			"						 }\n"
			"			             else if(pref.s7 == 3)\n"
			"						 {\n"
			"							*(__global uchar *) &weight_buf[ID1] = 255;\n"
			"							*(__global uchar *) &weight_buf[ID2] = 255;\n"
			"						 }\n";
	}
	opencl_kernel_code +=
		"\n"
		"						for(int cam = 0; cam < NumCam; cam ++)\n"
		"							if(cam != dim.s2 && cam != dim.s3)\n"
		"							{\n"
		"								uint offset_pixel = cam * equi_height;\n"
		"								uint ID_PIXEL = ((dim.s1 + offset_pixel) * equi_width) + dim.s0;\n"
		"								*(__global uchar *)&weight_buf[ID_PIXEL] = 0;\n"
		"							}\n"
		"\n"
		"					}\n"
		"\n";
	if (SHOW_ALL_SEAMS)
	{
		opencl_kernel_code +=
			"				}\n";
	}
	opencl_kernel_code +=
#endif
		"				}\n"
		"\n";

	if (DRAW_SEAM == 1)// Black Seam 
	{
		opencl_kernel_code +=
			"			if(dim.s5 == 0)\n"
			"			{\n"
#if ENABLE_VERTICAL_SEAM
			"				if (dim.s0 == path.s0)\n"
			"				{\n"
			"					*(__global uchar *) &weight_buf[ID1] = 0;\n"
			"					*(__global uchar *) &weight_buf[ID2] = 0;\n"
			"				}\n"
#endif
			"			}\n"
			"			else if(dim.s5 == 1)\n"
			"			{\n"
#if ENABLE_HORIZONTAL_SEAM
			"				if (dim.s1 == path.s0)\n"
			"				{\n"
			"					*(__global uchar *) &weight_buf[ID1] = 0;\n"
			"					*(__global uchar *) &weight_buf[ID2] = 0;\n"
			"				}\n"
#endif
			"			}\n"
			"\n";
	}
	if (DRAW_SEAM == 2)// White Seam
	{
		opencl_kernel_code +=
			"			if(dim.s5 == 0)\n"
			"			{\n"
#if ENABLE_VERTICAL_SEAM
			"				if (dim.s0 == path.s0)\n"
			"				{\n"
			"					*(__global uchar *) &weight_buf[ID1] = 255;\n"
			"					*(__global uchar *) &weight_buf[ID2] = 255;\n"
			"				}\n"
#endif
			"			}\n"
			"			else if(dim.s5 == 1)\n"
			"			{\n"
#if ENABLE_HORIZONTAL_SEAM
			"				if (dim.s1 == path.s0)\n"
			"				{\n"
			"					*(__global uchar *) &weight_buf[ID1] = 255;\n"
			"					*(__global uchar *) &weight_buf[ID2] = 255;\n"
			"				}\n"
#endif
			"			}\n"
			"\n";
	}
	opencl_kernel_code +=
		"\n"
		"			}\n"
		"		}\n"
		"\n"
		"}\n";

	return VX_SUCCESS;
}

//! \brief The kernel publisher.
vx_status seamfind_set_weights_publish(vx_context context)
{
	// add kernel to the context with callbacks
	vx_kernel kernel = vxAddKernel(context, "com.amd.loomsl.seamfind_set_weights",
		AMDOVX_KERNEL_STITCHING_SEAMFIND_SET_WEIGHTS,
		seamfind_set_weights_kernel,
		9,
		seamfind_set_weights_input_validator,
		seamfind_set_weights_output_validator,
		seamfind_set_weights_initialize,
		seamfind_set_weights_deinitialize);
	ERROR_CHECK_OBJECT(kernel);

	amd_kernel_query_target_support_f query_target_support_f = seamfind_set_weights_query_target_support;
	amd_kernel_opencl_codegen_callback_f opencl_codegen_callback_f = seamfind_set_weights_opencl_codegen;
	amd_kernel_opencl_global_work_update_callback_f opencl_global_work_update_callback_f = seamfind_set_weights_opencl_global_work_update;

	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_CODEGEN_CALLBACK, &opencl_codegen_callback_f, sizeof(opencl_codegen_callback_f)));
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_GLOBAL_WORK_UPDATE_CALLBACK, &opencl_global_work_update_callback_f, sizeof(opencl_global_work_update_callback_f)));
	// set kernel parameters
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 6, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 7, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 8, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));

	// finalize and release kernel object
	ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
	ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

	return VX_SUCCESS;
}


/***********************************************************************************************************************************

Seam Find Kernel - Analyzer

************************************************************************************************************************************/
//! \brief The input validator callback.
static vx_status VX_CALLBACK seamfind_analyze_input_validator(vx_node node, vx_uint32 index)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	// get reference for parameter at specified index
	vx_reference ref = avxGetNodeParamRef(node, index);
	ERROR_CHECK_OBJECT(ref);

	if (index == 0)
	{ // Current Frame
		vx_enum itemtype = VX_TYPE_INVALID;
		ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)ref, VX_SCALAR_ATTRIBUTE_TYPE, &itemtype, sizeof(itemtype)));

		if (itemtype == VX_TYPE_UINT32) {
			status = VX_SUCCESS;
		}
		else {
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: SeamFind Analyze scalar type should be a UINT32\n");
		}
		ERROR_CHECK_STATUS(vxReleaseScalar((vx_scalar *)&ref));
	}

	if (index == 1)
	{ // array object of StitchSeamFindPreference type
		vx_size itemsize = 0;
		ERROR_CHECK_STATUS(vxQueryArray((vx_array)ref, VX_ARRAY_ATTRIBUTE_ITEMSIZE, &itemsize, sizeof(itemsize)));
		if (itemsize == sizeof(StitchSeamFindPreference)) {
			status = VX_SUCCESS;
		}
		else {
			status = VX_ERROR_INVALID_DIMENSION;
			vxAddLogEntry((vx_reference)node, status, "ERROR: SeamFind array element (StitchSeamFindPreference) size should be 16 bytes\n");
		}
		ERROR_CHECK_STATUS(vxReleaseArray((vx_array *)&ref));
	}

	return status;
}

//! \brief The output validator callback.
static vx_status VX_CALLBACK seamfind_analyze_output_validator(vx_node node, vx_uint32 index, vx_meta_format meta)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	vx_reference ref = avxGetNodeParamRef(node, index);
	if (index == 2)
	{ // array object of StitchSeamScene type
		vx_enum itemtype = VX_TYPE_INVALID;
		ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)ref, VX_SCALAR_ATTRIBUTE_TYPE, &itemtype, sizeof(itemtype)));

		if (itemtype == VX_TYPE_UINT32) {
			status = VX_SUCCESS;
		}
		else {
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: SeamFind Analyze output scalar type should be a UINT32\n");
		}
		ERROR_CHECK_STATUS(vxReleaseScalar((vx_scalar *)&ref));
		ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_SCALAR_ATTRIBUTE_TYPE, &itemtype, sizeof(itemtype)));
	}
	return status;
}

//! \brief The kernel execution.
static vx_status VX_CALLBACK seamfind_analyze_kernel(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
	//Number Of Cameras - Variable 0 
	vx_uint32 current_frame = 0;
	ERROR_CHECK_STATUS(vxReadScalarValue((vx_scalar)parameters[0], &current_frame));

	//Seam Find Pref Array - Variable 1
	vx_array Array_SeamFind_Pref = (vx_array)parameters[1];
	vx_size SeamFind_Pref_max = 0;
	ERROR_CHECK_STATUS(vxQueryArray(Array_SeamFind_Pref, VX_ARRAY_ATTRIBUTE_NUMITEMS, &SeamFind_Pref_max, sizeof(SeamFind_Pref_max)));
	StitchSeamFindPreference *SeamFind_Pref = nullptr;
	vx_size stride_pref = sizeof(StitchSeamFindPreference);
	ERROR_CHECK_STATUS(vxAccessArrayRange(Array_SeamFind_Pref, 0, SeamFind_Pref_max, &stride_pref, (void **)&SeamFind_Pref, VX_READ_ONLY));

	vx_uint32 flag = 0;
	for (int i = 0; i < SeamFind_Pref_max; i++)
	{
		if (SeamFind_Pref[i].priority != -1){
			if (SeamFind_Pref[i].start_frame == current_frame) flag++;
			else if ((current_frame + 1) % (SeamFind_Pref[i].frequency + SeamFind_Pref[i].seam_type_num) == 0) flag++;

			if (flag) break;
		}
	}

	vx_scalar Scalar_flag = (vx_scalar)parameters[2];
	ERROR_CHECK_STATUS(vxWriteScalarValue(Scalar_flag, &flag));
	ERROR_CHECK_STATUS(vxCommitArrayRange(Array_SeamFind_Pref, 0, SeamFind_Pref_max, SeamFind_Pref));

	return VX_SUCCESS;
}

//! \brief The kernel publisher.
vx_status seamfind_analyze_publish(vx_context context)
{
	// add kernel to the context with callbacks
	vx_kernel kernel = vxAddKernel(context, "com.amd.loomsl.seamfind_analyze",
		AMDOVX_KERNEL_STITCHING_SEAMFIND_ANALYZE,
		seamfind_analyze_kernel,
		3,
		seamfind_analyze_input_validator,
		seamfind_analyze_output_validator,
		nullptr,
		nullptr);
	ERROR_CHECK_OBJECT(kernel);

	// set kernel parameters
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_OUTPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));

	// finalize and release kernel object
	ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
	ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

	return VX_SUCCESS;
}

//////////////////////////////////////////////////////////////////////
// Calculate buffer sizes and generate data in buffers for seam find
//   CalculateLargestSeamFindBufferSizes  - useful when reinitialize is enabled
//   CalculateSmallestSeamFindBufferSizes - useful when reinitialize is disabled
//   GenerateSeamFindBuffers              - generate tables

vx_status CalculateLargestSeamFindBufferSizes(
	vx_uint32 numCamera,                    // [in] number of cameras
	vx_uint32 eqrWidth,                     // [in] output equirectangular image width
	vx_uint32 eqrHeight,                    // [in] output equirectangular image height
	vx_size * seamFindValidEntryCount,      // [out] number of entries needed by seamFind valid table
	vx_size * seamFindWeightEntryCount,     // [out] number of entries needed by seamFind weight table
	vx_size * seamFindAccumEntryCount,      // [out] number of entries needed by seamFind accum table
	vx_size * seamFindPrefInfoEntryCount,   // [out] number of entries needed by seamFind pref/info table
	vx_size * seamFindPathEntryCount        // [out] number of entries needed by seamFind path table
	)
{
	*seamFindValidEntryCount = eqrHeight * ((numCamera * (numCamera - 1) )/ 2);
	*seamFindWeightEntryCount = ((eqrWidth * eqrHeight) / 8) * ((numCamera * (numCamera - 1)) / 2);
	*seamFindAccumEntryCount = ((eqrWidth * eqrHeight) / 8) * ((numCamera * (numCamera - 1)) / 2);
	*seamFindPrefInfoEntryCount = ((numCamera * (numCamera - 1)) / 2);
	*seamFindPathEntryCount = eqrWidth * ((numCamera * (numCamera - 1)) / 2);
	return VX_SUCCESS;
}

static bool GenerateSeamFindOverlapsForFishEyeOnEquator(
	vx_uint32 numCamera,                          // [in] number of cameras
	vx_uint32 eqrWidth,                           // [in] output equirectangular image width
	vx_uint32 eqrHeight,                          // [in] output equirectangular image height
	const camera_params * camera_par,             // [in] camera parameters
	const vx_float32 * live_stitch_attr,          // [in] attributes
	const vx_uint32 * validPixelCamMap,           // [in] valid pixel camera index map: size: [eqrWidth * eqrHeight]
	const vx_rectangle_t * const * overlapValid,  // [in] overlap regions: overlapValid[cam_i][cam_j] for overlap of cam_i and cam_j, cam_j <= cam_i
	vx_rectangle_t ** overlapRegion               // [out] overlap regions: overlapRegion[cam_i][cam_j] for overlap of cam_i and cam_j, cam_j <= cam_i
	)
{
	// initialize output overlap region to overlap valid
	for (vx_uint32 i = 0; i < numCamera; i++) {
		for (vx_uint32 j = 0; j <= i; j++) {
			overlapRegion[i][j] = overlapValid[i][j];
		}
	}

	// get configuration parameters
	vx_float32 hfovMin = live_stitch_attr[LIVE_STITCH_ATTR_SEAM_COEQUSH_HFOV_MIN];
	vx_float32 pitchTolerance = live_stitch_attr[LIVE_STITCH_ATTR_SEAM_COEQUSH_PITCH_TOL];
	vx_float32 yawTolerance = live_stitch_attr[LIVE_STITCH_ATTR_SEAM_COEQUSH_YAW_TOL];
	vx_float32 overlapHorizontalRatio = live_stitch_attr[LIVE_STITCH_ATTR_SEAM_COEQUSH_OVERLAP_HR];
	vx_float32 overlapVertialInDegree = live_stitch_attr[LIVE_STITCH_ATTR_SEAM_COEQUSH_OVERLAP_VD];
	vx_float32 topBotTolerance = live_stitch_attr[LIVE_STITCH_ATTR_SEAM_COEQUSH_TOPBOT_TOL];
	vx_float32 topBotVerticalGapInDegree = live_stitch_attr[LIVE_STITCH_ATTR_SEAM_COEQUSH_TOPBOT_VGD];
	vx_int32 SEAM_COEQUSH_ENABLE = ((vx_int32)live_stitch_attr[LIVE_STITCH_ATTR_SEAM_COEQUSH_ENABLE]);

	// find max pitch variance for equator cam
	vx_float32 maxPitchVariance = 0;
	for (vx_uint32 camId_1 = 0; camId_1 < numCamera; camId_1++) {
		for (vx_uint32 camId_2 = camId_1 + 1; camId_2 < numCamera; camId_2++) {
			vx_float32	pitchDiff = fabsf(camera_par[camId_1].focal.pitch - camera_par[camId_2].focal.pitch);
			if (pitchDiff <= pitchTolerance){
				vx_float32 pitchVariance = fabsf(camera_par[camId_1].focal.pitch) - pitchTolerance;
				maxPitchVariance = std::max(maxPitchVariance, pitchVariance);
			}
		}
	}

	// get cameras on equator sorted with yaw from zero and above
	typedef struct { vx_float32 yaw; vx_uint32 camId; } CameraItem;
	CameraItem camList[LIVE_STITCH_MAX_CAMERAS];
	vx_uint32 eqrCamCount = 0, eqrCamIdTop = numCamera, eqrCamIdBot = numCamera;
	for (vx_uint32 camId = 0; camId < numCamera; camId++) {
		if (camera_par[camId].lens.hfov >= hfovMin && fabsf(camera_par[camId].focal.pitch) <= (pitchTolerance + maxPitchVariance)) {
			camList[eqrCamCount].camId = camId;
			camList[eqrCamCount].yaw = camera_par[camId].focal.yaw + (camera_par[camId].focal.yaw < 0 ? 360 : 0);
			eqrCamCount++;
		}
		else if (eqrCamIdTop == numCamera && camera_par[camId].focal.pitch >= (90 - topBotTolerance) && camera_par[camId].focal.pitch <= 90) {
			eqrCamIdTop = camId;
		}
		else if (eqrCamIdBot == numCamera && camera_par[camId].focal.pitch <= (topBotTolerance - 90) && camera_par[camId].focal.pitch >= -90) {
			eqrCamIdBot = camId;
		}
		else {
			return false; // detected that not all cameras are on equator or top or bottom
		}
	}
	for (bool sorted = false; !sorted;) {
		sorted = true;
		for (vx_uint32 i = 1; i < eqrCamCount; i++) {
			if (camList[i - 1].yaw > camList[i].yaw) {
				CameraItem tmp = camList[i - 1];
				camList[i - 1] = camList[i];
				camList[i] = tmp;
				sorted = false;
			}
		}
	}
	for (vx_uint32 cam = 1; cam < eqrCamCount; cam++) {
		if (fabsf(camList[cam].yaw - (camList[0].yaw + 360.0f * cam / eqrCamCount)) > yawTolerance) {
			return false; // detected that not all cameras are equidistant
		}
	}
	if (eqrCamCount < 2) {
		return false; // need atleast two or more cameras on equator
	}

	// disable overlaps between top and bottom cameras (if they both exist)
	if (eqrCamIdTop < numCamera && eqrCamIdBot < numCamera) {
		vx_rectangle_t * overlapRect = &overlapRegion[std::max(eqrCamIdTop, eqrCamIdBot)][std::min(eqrCamIdTop, eqrCamIdBot)];
		overlapRect->start_x = overlapRect->end_x = 0;
		overlapRect->start_y = overlapRect->end_y = 0;
	}

	// calculate overlap region for cameras on equator with their neighbors
	float overlapInDegrees = overlapHorizontalRatio * 360.0f / eqrCamCount;
	for (vx_uint32 index = 0; index < eqrCamCount; index++) {
		vx_uint32 indexToRight = (index + 1) % eqrCamCount;
		vx_uint32 indexToLeft = (index - 1 + eqrCamCount) % eqrCamCount;
		vx_uint32 camId = camList[index].camId;
		vx_uint32 camIdToRight = camList[indexToRight].camId;
		// disable non-neighbor camera overlaps
		for (vx_uint32 indexOfNeighbor = 0; indexOfNeighbor < eqrCamCount; indexOfNeighbor++) {
			if (indexOfNeighbor != index && indexOfNeighbor != indexToLeft && indexOfNeighbor != indexToRight) {
				vx_rectangle_t * overlapRect;
				overlapRect = &overlapRegion[std::max(camId, camList[indexOfNeighbor].camId)][std::min(camId, camList[indexOfNeighbor].camId)];
				overlapRect->start_x = overlapRect->end_x = 0;
				overlapRect->start_y = overlapRect->end_y = 0;
			}
		}
		// update neighbor camera overlaps
		float yawRightBorder = camList[index].yaw < camList[indexToRight].yaw ?
			(camList[index].yaw + camList[indexToRight].yaw) * 0.5f :
			(camList[index].yaw + camList[indexToRight].yaw - 360) * 0.5f;
		vx_int32 start_x = ((vx_int32)((180 + yawRightBorder - overlapInDegrees * 0.5f) * eqrWidth / 360.0f + 1) + (vx_int32)eqrWidth) % (vx_int32)eqrWidth;
		vx_int32 end_x = ((vx_int32)((180 + yawRightBorder + overlapInDegrees * 0.5f) * eqrWidth / 360.0f) + (vx_int32)eqrWidth) % (vx_int32)eqrWidth;
		if (end_x < start_x) {
			// when region wraps from right to left, pick the larger of righside or leftside
			if (end_x > (vx_int32)eqrWidth - start_x) start_x = 0;
			else end_x = (vx_int32)eqrWidth;
		}
		vx_rectangle_t * overlapRect = &overlapRegion[std::max(camId, camIdToRight)][std::min(camId, camIdToRight)];
		overlapRect->start_x = std::max(overlapRect->start_x, (vx_uint32)start_x);
		overlapRect->end_x = std::min(overlapRect->end_x, (vx_uint32)end_x);
		// update overlap with top and bottom cameras (if present)
		float yawLeftBorder = camList[index].yaw > camList[indexToLeft].yaw ?
			(camList[index].yaw + camList[indexToLeft].yaw) * 0.5f :
			(camList[index].yaw + camList[indexToLeft].yaw - 360) * 0.5f;
		if (eqrCamIdTop < numCamera) {
			vx_rectangle_t * overlapRectHoriz = &overlapRegion[std::max(camId, eqrCamIdTop)][std::min(camId, eqrCamIdTop)];
			if (SEAM_COEQUSH_ENABLE > 1) {
				// update overlap between equator and top camera
				float pitchCenterToTop = (camera_par[camId].focal.pitch + camera_par[eqrCamIdTop].focal.pitch) * 0.5f;
				start_x = ((vx_int32)((180 + yawLeftBorder) * eqrWidth / 360.0f + 1) + (vx_int32)eqrWidth) % (vx_int32)eqrWidth;
				end_x = ((vx_int32)((180 + yawRightBorder) * eqrWidth / 360.0f) + (vx_int32)eqrWidth) % (vx_int32)eqrWidth;
				if (end_x < start_x) {
					// when region wraps from right to left, pick the larger of righside or leftside
					if (end_x > (vx_int32)eqrWidth - start_x) start_x = 0;
					else end_x = (vx_int32)eqrWidth;
				}
				overlapRectHoriz->start_x = std::max(overlapRectHoriz->start_x, (vx_uint32)start_x);
				overlapRectHoriz->end_x = std::min(overlapRectHoriz->end_x, (vx_uint32)end_x);
				overlapRectHoriz->start_y = (vx_uint32)std::max((vx_int32)overlapRectHoriz->start_y, (vx_int32)((90 - pitchCenterToTop - overlapVertialInDegree * 0.5f) * eqrHeight / 180.0f + 1));
				overlapRectHoriz->end_y = (vx_uint32)std::min((vx_int32)overlapRectHoriz->end_y, (vx_int32)((90 - pitchCenterToTop + overlapVertialInDegree * 0.5f) * eqrHeight / 180.0f));
				vx_uint32 width = overlapRectHoriz->end_x - overlapRectHoriz->start_x;
				vx_uint32 height = overlapRectHoriz->end_y - overlapRectHoriz->start_y;
				if (width < 2 * height) {
					vx_uint32 mid_y = (overlapRectHoriz->start_y + overlapRectHoriz->end_y) / 2;
					overlapRectHoriz->start_y = mid_y - width / 4;
					overlapRectHoriz->end_y = mid_y + width / 4;
				}
			}
			else {
				// disable overlap between equator and top camera
				overlapRectHoriz->start_x = overlapRectHoriz->end_x = 0;
			}
			// reduce the overlap top to default seam border
			float veriticalGapInDegrees = topBotVerticalGapInDegree;
			if (topBotVerticalGapInDegree < 0) {
				veriticalGapInDegrees = 0;
				if (camera_par[eqrCamIdTop].lens.r_crop > 0) {
					veriticalGapInDegrees = camera_par[eqrCamIdTop].lens.hfov * camera_par[eqrCamIdTop].lens.r_crop / camera_par[eqrCamIdTop].lens.haw;
				}
			}
			overlapRect->start_y = std::max(overlapRect->start_y, (vx_uint32)(veriticalGapInDegrees * eqrHeight / 180.0f));
		}
		if (eqrCamIdBot < numCamera) {
			vx_rectangle_t *overlapRectHoriz = &overlapRegion[std::max(camId, eqrCamIdBot)][std::min(camId, eqrCamIdBot)];
			if (SEAM_COEQUSH_ENABLE > 1) {
				// update overlap between equator and bottom camera
				float pitchCenterToBot = (camera_par[camId].focal.pitch + camera_par[eqrCamIdBot].focal.pitch) * 0.5f;
				start_x = ((vx_int32)((180 + yawLeftBorder) * eqrWidth / 360.0f + 1) + (vx_int32)eqrWidth) % (vx_int32)eqrWidth;
				end_x = ((vx_int32)((180 + yawRightBorder) * eqrWidth / 360.0f) + (vx_int32)eqrWidth) % (vx_int32)eqrWidth;
				if (end_x < start_x) {
					// when region wraps from right to left, pick the larger of righside or leftside
					if (end_x > (vx_int32)eqrWidth - start_x) start_x = 0;
					else end_x = (vx_int32)eqrWidth;
				}
				overlapRectHoriz->start_x = std::max(overlapRectHoriz->start_x, (vx_uint32)start_x);
				overlapRectHoriz->end_x = std::min(overlapRectHoriz->end_x, (vx_uint32)end_x);
				overlapRectHoriz->start_y = (vx_uint32)std::max((vx_int32)overlapRectHoriz->start_y, (vx_int32)((90 - pitchCenterToBot - overlapVertialInDegree * 0.5f) * eqrHeight / 180.0f + 1));
				overlapRectHoriz->end_y = (vx_uint32)std::min((vx_int32)overlapRectHoriz->end_y, (vx_int32)((90 - pitchCenterToBot + overlapVertialInDegree * 0.5f) * eqrHeight / 180.0f));
				vx_uint32 width = overlapRectHoriz->end_x - overlapRectHoriz->start_x;
				vx_uint32 height = overlapRectHoriz->end_y - overlapRectHoriz->start_y;
				if (width < 2 * height) {
					vx_uint32 mid_y = (overlapRectHoriz->start_y + overlapRectHoriz->end_y) / 2;
					overlapRectHoriz->start_y = mid_y - width / 4;
					overlapRectHoriz->end_y = mid_y + width / 4;
				}
			}
			else {
				// disable overlap between equator and bottom camera
				overlapRectHoriz->start_x = overlapRectHoriz->end_x = 0;
			}
			// reduce the overlap bttom to default seam border
			float veriticalGapInDegrees = topBotVerticalGapInDegree;
			if (topBotVerticalGapInDegree < 0) {
				veriticalGapInDegrees = 0;
				if (camera_par[eqrCamIdBot].lens.r_crop > 0) {
					veriticalGapInDegrees = camera_par[eqrCamIdBot].lens.hfov * camera_par[eqrCamIdBot].lens.r_crop / camera_par[eqrCamIdBot].lens.haw;
				}
			}
			overlapRect->end_y = std::min(overlapRect->end_y, (vx_uint32)((180 - veriticalGapInDegrees) * eqrHeight / 180.0f));
		}
	}

	return true;
}

static inline vx_status GenerateSeamFindBuffersModel(
	vx_uint32 numCamera,                            // [in] number of cameras
	vx_uint32 eqrWidth,                             // [in] output equirectangular image width
	vx_uint32 eqrHeight,                            // [in] output equirectangular image height
	const camera_params * camera_par,               // [in] camera parameters
	const vx_uint32 * validPixelCamMap,             // [in] valid pixel camera index map: size: [eqrWidth * eqrHeight]
	const vx_rectangle_t * const * overlapValid,    // [in] overlap regions: overlapValid[cam_i][cam_j] for overlap of cam_i and cam_j, cam_j <= cam_i
	const vx_uint32 * validCamOverlapInfo,          // [in] camera overlap info - use "validCamOverlapInfo[cam_i] & (1 << cam_j)": size: [LIVE_STITCH_MAX_CAMERAS]
	const vx_uint32 * paddedPixelCamMap,            // [in] padded pixel camera index map: size: [eqrWidth * eqrHeight](optional)
	const vx_rectangle_t * const * overlapPadded,   // [in] overlap regions: overlapPadded[cam_i][cam_j] for overlap of cam_i and cam_j, cam_j <= cam_i(optional)
	const vx_uint32 * paddedCamOverlapInfo,         // [in] camera overlap info - use "paddedCamOverlapInfo[cam_i] & (1 << cam_j)": size: [LIVE_STITCH_MAX_CAMERAS](optional)
	const vx_float32 * live_stitch_attr,            // [in] attributes
	vx_size validTableSize,                         // [in] size of seamFind valid table in terms of number of entries
	vx_size weightTableSize,                        // [in] size of seamFind weight table in terms of number of entries
	vx_size accumTableSize,                         // [in] size of seamFind accum table in terms of number of entries
	vx_size prefInfoTableSize,                      // [in] size of seamFind pref/info table in terms of number of entries
	StitchSeamFindValidEntry * validTable,          // [out] valid table
	StitchSeamFindWeightEntry * weightTable,        // [out] weight table
	StitchSeamFindAccumEntry * accumTable,          // [out] accum table
	StitchSeamFindPreference * prefTable,           // [out] preference table
	StitchSeamFindInformation * infoTable,          // [out] info table
	vx_size * seamFindValidEntryCount,              // [out] number of entries needed by seamFind valid table
	vx_size * seamFindWeightEntryCount,             // [out] number of entries needed by seamFind weight table
	vx_size * seamFindAccumEntryCount,              // [out] number of entries needed by seamFind accum table
	vx_size * seamFindPrefInfoEntryCount,           // [out] number of entries needed by seamFind pref/info table
	vx_size * seamFindPathEntryCount                // [out] number of entries needed by seamFind path table
	)
{
	// attributes
	vx_int32 VERTICAL_SEAM_PRIORITY = (vx_int32)live_stitch_attr[LIVE_STITCH_ATTR_SEAM_VERT_PRIORITY];
	vx_int32 HORIZONTAL_SEAM_PRIORITY = (vx_int32)live_stitch_attr[LIVE_STITCH_ATTR_SEAM_HORT_PRIORITY];
	vx_int32 SEAM_FREQUENCY = (vx_int32)live_stitch_attr[LIVE_STITCH_ATTR_SEAM_FREQUENCY];
	vx_int32 SEAM_QUALITY = (vx_int32)live_stitch_attr[LIVE_STITCH_ATTR_SEAM_QUALITY];
	vx_int32 SEAM_STAGGER = (vx_int32)live_stitch_attr[LIVE_STITCH_ATTR_SEAM_STAGGER];
	vx_int32 SEAM_LOCK = (vx_int32)live_stitch_attr[LIVE_STITCH_ATTR_SEAM_LOCK];
	vx_int32 SEAM_FLAG = ((vx_int32)live_stitch_attr[LIVE_STITCH_ATTR_SEAM_FLAGS]) & 3;
	vx_int32 SEAM_COEQUSH_ENABLE = ((vx_int32)live_stitch_attr[LIVE_STITCH_ATTR_SEAM_COEQUSH_ENABLE]);

	// allocate memory for overlap region
	std::vector<vx_rectangle_t> overlapRegionRect(numCamera * numCamera);
	vx_rectangle_t * overlapRegion[LIVE_STITCH_MAX_CAMERAS];
	for (vx_uint32 i = 0; i < numCamera; i++) {
		overlapRegion[i] = overlapRegionRect.data() + i * numCamera;
	}

	if (SEAM_COEQUSH_ENABLE) {
		// check for special handling for circular fisheye lens on equator
		bool doSpecialHandlingOfCircFishEyeOnEquator = GenerateSeamFindOverlapsForFishEyeOnEquator(numCamera, eqrWidth, eqrHeight,
			camera_par, live_stitch_attr, validPixelCamMap, overlapValid, overlapRegion);
		if (doSpecialHandlingOfCircFishEyeOnEquator) {
			overlapValid = overlapRegion;
		}
	}

	vx_uint32 accumulation_entry = 0, valid_entry = 0, weight_entry = 0, overlap_number = 0;
	vx_int16 horizontal_overlap = 0, vertical_overlap = 0;
	// generate tables
	for (vx_uint32 i = 1; i < numCamera; i++) {
		for (vx_uint32 j = 0; j < i; j++) {
			vx_uint32 overlapMaskBits = (1 << i) | (1 << j);
			vx_uint32 start_x = overlapValid[i][j].start_x, end_x = overlapValid[i][j].end_x;
			vx_uint32 start_y = overlapValid[i][j].start_y, end_y = overlapValid[i][j].end_y;
			if (start_x < end_x && (end_x - start_x) <= (end_y - start_y)) {
				// make sure top and bottom has a valid pixels between start_x+1 and end_x-1
				for (; start_y < end_y; start_y++) {
					const vx_uint32 * map = validPixelCamMap + start_y * eqrWidth;
					vx_uint32 x = start_x + 1;
					for (; x < end_x - 1; x++)
						if ((map[x] & overlapMaskBits) == overlapMaskBits)
							break;
					if (x < end_x - 1)
						break;
				}
				for (; start_y < end_y; end_y--) {
					const vx_uint32 * map = validPixelCamMap + (end_y - 1) * eqrWidth;
					vx_uint32 x = start_x + 1;
					for (; x < end_x - 1; x++)
						if ((map[x] & overlapMaskBits) == overlapMaskBits)
							break;
					if (x < end_x - 1)
						break;
				}
			}
			if (start_y < end_y && (end_y - start_y) <= (end_x - start_x)) {
				// make sure left and right has a valid pixels between start_y+1 and end_y-1
				for (; start_x < end_x; start_x++) {
					const vx_uint32 * map = validPixelCamMap + start_x;
					vx_uint32 y = start_y + 1;
					for (; y < end_y - 1; y++)
						if ((map[y * eqrWidth] & overlapMaskBits) == overlapMaskBits)
							break;
					if (y < end_y - 1)
						break;
				}
				for (; start_x < end_x; end_x--) {
					const vx_uint32 * map = validPixelCamMap + (end_x - 1);
					vx_uint32 y = start_y + 1;
					for (; y < end_y - 1; y++)
						if ((map[y * eqrWidth] & overlapMaskBits) == overlapMaskBits)
							break;
					if (y < end_y - 1)
						break;
				}
			}
			if (start_x < end_x && start_y < end_y) {
				vx_uint32 offsetY_i = i * eqrHeight;
				vx_uint32 offsetY_j = j * eqrHeight;
				vx_int16 overlapWidth = end_x - start_x;
				vx_int16 overlapHeight = end_y - start_y;
				vx_int16 SEAM_TYPE = -1;
				if (overlapHeight >= overlapWidth) {
					PRINTF("SEAM-VERTICAL   %2d %2d (%4d,%4d)-(%4d,%4d) (%4dx%4d)\n", i, j, start_x, start_y, end_x, end_y, overlapWidth, overlapHeight);
					// vertical seam
					SEAM_TYPE = VERTICAL_SEAM;
					vertical_overlap++;
					if (validTable) {
						for (vx_uint32 x = start_x; x < end_x; x++) {
							if (valid_entry < validTableSize) {
								StitchSeamFindValidEntry validTableEntry = { 0 };
								validTableEntry.dstX = (vx_int16)x;
								validTableEntry.dstY = (vx_int16)start_y;
								validTableEntry.width = overlapWidth;
								validTableEntry.height = overlapHeight;
								validTableEntry.OverLapX = (vx_int16)x;
								validTableEntry.OverLapY = (vx_int16)(start_y + offsetY_i);
								validTableEntry.CAMERA_ID_1 = (vx_int16)j;
								validTableEntry.ID = (vx_int16)overlap_number;
								validTable[valid_entry] = validTableEntry;
							}
							valid_entry++;
						}
					}
					else {
						valid_entry += end_x - start_x;
					}
					if (prefTable) {
						if (overlap_number < prefInfoTableSize)	{
							StitchSeamFindPreference prefTableEntry = { 0 };
							prefTableEntry.type = SEAM_TYPE;
							prefTableEntry.seam_type_num = vertical_overlap;
							prefTableEntry.start_frame = SEAM_STAGGER * vertical_overlap;
							prefTableEntry.frequency = SEAM_FREQUENCY;
							prefTableEntry.quality = SEAM_QUALITY;
							prefTableEntry.priority = VERTICAL_SEAM_PRIORITY;
							prefTableEntry.seam_lock = SEAM_LOCK;
							prefTableEntry.scene_flag = SEAM_FLAG;
							prefTable[overlap_number] = prefTableEntry;
						}
					}
				}
				else {
					PRINTF("SEAM-HORIZONTAL %2d %2d (%4d,%4d)-(%4d,%4d) (%4dx%4d)\n", i, j, start_x, start_y, end_x, end_y, overlapWidth, overlapHeight);
					// horizontal seam
					SEAM_TYPE = HORIZONTAL_SEAM;
					horizontal_overlap++;
					if (validTable) {
						for (vx_uint32 y = start_y; y < end_y; y++) {
							if (valid_entry < validTableSize) {
								StitchSeamFindValidEntry  validTableEntry = { 0 };
								validTableEntry.dstX = (vx_int16)start_x;
								validTableEntry.dstY = (vx_int16)y;
								validTableEntry.width = overlapWidth;
								validTableEntry.height = overlapHeight;
								validTableEntry.OverLapX = (vx_int16)start_x;
								validTableEntry.OverLapY = (vx_int16)(y + offsetY_i);
								validTableEntry.CAMERA_ID_1 = (vx_int16)j;
								validTableEntry.ID = (vx_int16)overlap_number;
								validTable[valid_entry] = validTableEntry;
							}
							valid_entry++;
						}
					}
					else {
						valid_entry += end_y - start_y;
					}
					if (prefTable) {
						if (overlap_number < prefInfoTableSize) {
							StitchSeamFindPreference prefTableEntry = { 0 };
							prefTableEntry.type = SEAM_TYPE;
							prefTableEntry.seam_type_num = horizontal_overlap;
							prefTableEntry.start_frame = SEAM_STAGGER * horizontal_overlap;
							prefTableEntry.frequency = SEAM_FREQUENCY;
							prefTableEntry.quality = SEAM_QUALITY;
							prefTableEntry.priority = HORIZONTAL_SEAM_PRIORITY;
							prefTableEntry.seam_lock = SEAM_LOCK;
							prefTableEntry.scene_flag = SEAM_FLAG;
							prefTable[overlap_number] = prefTableEntry;
						}
					}
				}
				for (vx_uint32 ye = start_y; ye < end_y; ye++) {
					for (vx_uint32 xe = start_x; xe < end_x; xe++)	{
						if ((validPixelCamMap[ye  * eqrWidth + xe] & overlapMaskBits) == overlapMaskBits) {
							if (weightTable) {
								if (weight_entry < weightTableSize) {
									StitchSeamFindWeightEntry weightTableEntry = { 0 };
									weightTableEntry.x = xe;
									weightTableEntry.y = ye;
									weightTableEntry.cam_id_1 = j;
									weightTableEntry.cam_id_2 = i;
									weightTableEntry.overlap_id = overlap_number;
									weightTableEntry.overlap_type = SEAM_TYPE;
									weightTable[weight_entry] = weightTableEntry;
								}
							}
							weight_entry++;
						}
					}
				}
				if (infoTable) {
					if (overlap_number < prefInfoTableSize)	{
						StitchSeamFindInformation infoTableEntry = { 0 };
						infoTableEntry.cam_id_1 = j;
						infoTableEntry.cam_id_2 = i;
						infoTableEntry.start_x = start_x;
						infoTableEntry.start_y = start_y;
						infoTableEntry.end_x = end_x;
						infoTableEntry.end_y = end_y;
						infoTableEntry.offset = accumulation_entry;
						infoTable[overlap_number] = infoTableEntry;
					}
				}
				accumulation_entry += overlapWidth * overlapHeight;
				overlap_number++;
			}
		}
	}

	// check for buffer overflow error condition
	if ((validTable && valid_entry > validTableSize) ||
		(weightTable && weight_entry > weightTableSize) ||
		(accumTable && accumulation_entry > accumTableSize) ||
		(infoTable && overlap_number > prefInfoTableSize))
	{
		return VX_ERROR_NOT_SUFFICIENT;
	}

	// initialize accumTable
	if (accumTable) {
		if (accumulation_entry <= accumTableSize) {
			StitchSeamFindAccumEntry entry = { 0 };
			entry.parent_x = -1;
			entry.parent_y = -1;
			entry.value = -1;
			entry.propagate = -1;
			for (vx_uint32 i = 0; i < accumulation_entry; i++) {
				accumTable[i] = entry;
			}
		}
	}

	*seamFindValidEntryCount = valid_entry;
	*seamFindWeightEntryCount = weight_entry;
	*seamFindAccumEntryCount = accumulation_entry;
	*seamFindPrefInfoEntryCount = overlap_number;
	*seamFindPathEntryCount = overlap_number * eqrWidth;

	return VX_SUCCESS;
}

vx_status CalculateSmallestSeamFindBufferSizes(
	vx_uint32 numCamera,                           // [in] number of cameras
	vx_uint32 eqrWidth,                            // [in] output equirectangular image width
	vx_uint32 eqrHeight,                           // [in] output equirectangular image height
	const camera_params * camera_par,              // [in] camera parameters
	const vx_uint32 * validPixelCamMap,            // [in] valid pixel camera index map: size: [eqrWidth * eqrHeight]
	const vx_rectangle_t * const * overlapValid,   // [in] overlap regions: overlapValid[cam_i][cam_j] for overlap of cam_i and cam_j, cam_j <= cam_i
	const vx_uint32 * validCamOverlapInfo,         // [in] camera overlap info - use "validCamOverlapInfo[cam_i] & (1 << cam_j)": size: [LIVE_STITCH_MAX_CAMERAS]
	const vx_uint32 * paddedPixelCamMap,           // [in] padded pixel camera index map: size: [eqrWidth * eqrHeight](optional)
	const vx_rectangle_t * const * overlapPadded,  // [in] overlap regions: overlapPadded[cam_i][cam_j] for overlap of cam_i and cam_j, cam_j <= cam_i(optional)
	const vx_uint32 * paddedCamOverlapInfo,        // [in] camera overlap info - use "paddedCamOverlapInfo[cam_i] & (1 << cam_j)": size: [LIVE_STITCH_MAX_CAMERAS](optional)
	const vx_float32 * live_stitch_attr,           // [in] attributes
	vx_size * seamFindValidEntryCount,             // [out] number of entries needed by seamFind valid table
	vx_size * seamFindWeightEntryCount,            // [out] number of entries needed by seamFind weight table
	vx_size * seamFindAccumEntryCount,             // [out] number of entries needed by seamFind accum table
	vx_size * seamFindPrefInfoEntryCount,          // [out] number of entries needed by seamFind pref/info table
	vx_size * seamFindPathEntryCount               // [out] number of entries needed by seamFind path table
	)
{
	return GenerateSeamFindBuffersModel(numCamera, eqrWidth, eqrHeight, camera_par,
		validPixelCamMap, overlapValid, validCamOverlapInfo,
		paddedPixelCamMap, overlapPadded, paddedCamOverlapInfo,
		live_stitch_attr, 0, 0, 0, 0, nullptr, nullptr, nullptr, nullptr, nullptr,
		seamFindValidEntryCount, seamFindWeightEntryCount, seamFindAccumEntryCount, seamFindPrefInfoEntryCount, seamFindPathEntryCount);
}

vx_status GenerateSeamFindBuffers(
	vx_uint32 numCamera,                            // [in] number of cameras
	vx_uint32 eqrWidth,                             // [in] output equirectangular image width
	vx_uint32 eqrHeight,                            // [in] output equirectangular image height
	const camera_params * camera_par,               // [in] camera parameters
	const vx_uint32 * validPixelCamMap,             // [in] valid pixel camera index map: size: [eqrWidth * eqrHeight]
	const vx_rectangle_t * const * overlapValid,    // [in] overlap regions: overlapValid[cam_i][cam_j] for overlap of cam_i and cam_j, cam_j <= cam_i
	const vx_uint32 * validCamOverlapInfo,          // [in] camera overlap info - use "validCamOverlapInfo[cam_i] & (1 << cam_j)": size: [LIVE_STITCH_MAX_CAMERAS]
	const vx_uint32 * paddedPixelCamMap,            // [in] padded pixel camera index map: size: [eqrWidth * eqrHeight](optional)
	const vx_rectangle_t * const * overlapPadded,   // [in] overlap regions: overlapPadded[cam_i][cam_j] for overlap of cam_i and cam_j, cam_j <= cam_i(optional)
	const vx_uint32 * paddedCamOverlapInfo,         // [in] camera overlap info - use "paddedCamOverlapInfo[cam_i] & (1 << cam_j)": size: [LIVE_STITCH_MAX_CAMERAS](optional)
	const vx_float32 * live_stitch_attr,            // [in] attributes
	vx_size validTableSize,                         // [in] size of seamFind valid table in terms of number of entries
	vx_size weightTableSize,                        // [in] size of seamFind weight table in terms of number of entries
	vx_size accumTableSize,                         // [in] size of seamFind accum table in terms of number of entries
	vx_size prefInfoTableSize,                      // [in] size of seamFind pref/info table in terms of number of entries
	StitchSeamFindValidEntry * validTable,          // [out] valid table
	StitchSeamFindWeightEntry * weightTable,        // [out] weight table
	StitchSeamFindAccumEntry * accumTable,          // [out] accum table
	StitchSeamFindPreference * prefTable,           // [out] preference table
	StitchSeamFindInformation * infoTable,          // [out] info table
	vx_size * seamFindValidEntryCount,              // [out] number of entries needed by seamFind valid table
	vx_size * seamFindWeightEntryCount,             // [out] number of entries needed by seamFind weight table
	vx_size * seamFindAccumEntryCount,              // [out] number of entries needed by seamFind accum table
	vx_size * seamFindPrefInfoEntryCount,           // [out] number of entries needed by seamFind pref/info table
	vx_size * seamFindPathEntryCount                // [out] number of entries needed by seamFind path table
	)
{
	return GenerateSeamFindBuffersModel(numCamera, eqrWidth, eqrHeight, camera_par,
		validPixelCamMap, overlapValid, validCamOverlapInfo,
		paddedPixelCamMap, overlapPadded, paddedCamOverlapInfo,
		live_stitch_attr, validTableSize, weightTableSize, accumTableSize, prefInfoTableSize, validTable, weightTable, accumTable, prefTable, infoTable,
		seamFindValidEntryCount, seamFindWeightEntryCount, seamFindAccumEntryCount, seamFindPrefInfoEntryCount, seamFindPathEntryCount);
}
