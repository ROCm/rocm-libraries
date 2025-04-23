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

#include "internal_publishKernels.h"
#include <string>

int poppedDonuts = 0;
int globalDonutFlag = 0;
Mat globalDonutRefImage;
int globalDonutChange = 0;

/************************************************************************************************************
Bubbles
*************************************************************************************************************/
class AMD_donut_pop
{
private:
	int donutX, donutY, donutWidth, donutHeight;

public:
	AMD_donut_pop(int bX, int bY, int bW, int bH)
	{
		donutX = bX;
		donutY = bY;
		donutWidth = bW;
		donutHeight = bH;
	}

	~AMD_donut_pop()
	{
		donutX = 0;
		donutY = 0;
		donutWidth = 0;
		donutHeight = 0;
	}

	int update(int width, int height, Mat *Image)
	{
		int movementAmount = 0;
		if (globalDonutFlag > 10)
		{
			Mat diff_image;
			absdiff(*Image, globalDonutRefImage, diff_image);
			blur(diff_image, diff_image, Size(3, 3));

			cv::erode(diff_image, diff_image, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
			cv::erode(diff_image, diff_image, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
			cv::dilate(diff_image, diff_image, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
			cv::dilate(diff_image, diff_image, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));

			cv::threshold(diff_image, diff_image, 160, 255, 0);

			unsigned char *input = (unsigned char *)(diff_image.data);
			int b;

			for (int x = donutX; x <= (donutX + (donutWidth - 1)); x++)
				for (int y = donutY; y <= (donutY + (donutWidth - 1)); y++)
				{
					if ((x < diff_image.cols && x > 0) && (y < diff_image.rows && y > 0))
					{
						b = input[diff_image.cols * y + x];
						if (b == 255)
							movementAmount++;
					}
				}
		}

		if (movementAmount > 100)
		{
			poppedDonuts++;
			return 1;
		}
		else
		{
			donutY += 5;

			if (donutY > height)
				return 1;

			Mat test_image;
			test_image = *Image;

			// draw Donuts
			if (globalDonutChange == 0)
			{
				Point2f cen(donutX, donutY);
				cv::circle(*Image, cen, 8, Scalar(255, 255, 55), 5);
			}
			else
			{
				Point2f cen(donutX, donutY);
				cv::circle(*Image, cen, 5, Scalar(255, 255, 55), 5);
			}

			return 0;
		}
	}
};

struct Linked_list_pop
{
	AMD_donut_pop donut;
	int data;
	struct Linked_list_pop *next;
};
typedef struct Linked_list_pop donutNode;

// Function Prototyping
donutNode *donut_insert(donutNode *head, donutNode *x);
donutNode *donut_position_delete(donutNode *head, int p);
donutNode *donut_clean_node(donutNode *head);
int draw_pop_donuts(int, int, Mat *);
donutNode *PopDonuts = NULL;

/************************************************************************************************************
Draw Bubbles
*************************************************************************************************************/
int draw_pop_donuts(int width, int height, Mat *Image)
{
	static int count = 0;

	int randx = rand() % (width + 1);
	AMD_donut_pop new_element = AMD_donut_pop(randx, 0, 20, 20);

	donutNode *temp = (donutNode *)malloc(sizeof(donutNode));
	temp->donut = new_element;
	temp->next = NULL;
	PopDonuts = donut_insert(PopDonuts, temp);
	count++;

	donutNode *_donuts;
	_donuts = PopDonuts;
	int K = 0;
	int flag = 0;

	while (_donuts != NULL)
	{
		K++;
		flag = 0;

		if (_donuts->donut.update(width, height, Image) == 1)
		{
			_donuts = _donuts->next;
			PopDonuts = donut_position_delete(PopDonuts, K);
			count--;
			K--;
			flag = 1;
		}

		if (flag == 0)
			_donuts = _donuts->next;
	}

	return 0;
}

/************************************************************************************************************
input parameter validator.
param [in] node The handle to the node.
param [in] index The index of the parameter to validate.
*************************************************************************************************************/
vx_status VX_CALLBACK VX_bubbles_InputValidator(vx_node node, vx_uint32 index)
{
	vx_status status = VX_SUCCESS;
	vx_parameter param = vxGetParameterByIndex(node, index);

	if (index == 0)
	{
		vx_image image;
		vx_df_image df_image = VX_DF_IMAGE_VIRT;
		STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &image, sizeof(vx_image)));
		STATUS_ERROR_CHECK(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_FORMAT, &df_image, sizeof(df_image)));
		if (df_image != VX_DF_IMAGE_U8)
			status = VX_ERROR_INVALID_VALUE;
		vxReleaseImage(&image);
	}

	else if (index == 1)
	{
		vx_image image;
		vx_df_image df_image = VX_DF_IMAGE_VIRT;
		STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &image, sizeof(vx_image)));
		STATUS_ERROR_CHECK(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_FORMAT, &df_image, sizeof(df_image)));
		if (df_image != VX_DF_IMAGE_U8 && df_image != VX_DF_IMAGE_RGB)
			status = VX_ERROR_INVALID_VALUE;
		vxReleaseImage(&image);
	}

	vxReleaseParameter(&param);
	return status;
}

/************************************************************************************************************
output parameter validator.
*************************************************************************************************************/
vx_status VX_CALLBACK VX_bubbles_OutputValidator(vx_node node, vx_uint32 index, vx_meta_format meta)
{
	vx_status status = VX_SUCCESS;
	if (index == 1)
	{
		vx_parameter output_param = vxGetParameterByIndex(node, 1);
		vx_image output;
		vx_uint32 width = 0, height = 0;
		vx_df_image format = VX_DF_IMAGE_VIRT;

		STATUS_ERROR_CHECK(vxQueryParameter(output_param, VX_PARAMETER_ATTRIBUTE_REF, &output, sizeof(vx_image)));
		STATUS_ERROR_CHECK(vxQueryImage(output, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));
		STATUS_ERROR_CHECK(vxQueryImage(output, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
		STATUS_ERROR_CHECK(vxQueryImage(output, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));

		if (format != VX_DF_IMAGE_U8 && format != VX_DF_IMAGE_RGB)
			status = VX_ERROR_INVALID_VALUE;

		STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
		STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
		STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));

		vxReleaseImage(&output);
		vxReleaseParameter(&output_param);
	}
	return status;
}

/************************************************************************************************************
Execution Kernel
*************************************************************************************************************/
vx_status VX_CALLBACK VX_bubbles_Kernel(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	vx_status status = VX_SUCCESS;

	vx_image image_in = (vx_image)parameters[0];
	vx_image image_out = (vx_image)parameters[1];
	Mat *mat, bl;

	// wait to restart - press any key
	if (poppedDonuts >= 1015)
	{
		poppedDonuts = 0;
		waitKey(0);
	}

	// Converting VX Image to OpenCV Mat
	STATUS_ERROR_CHECK(VX_to_CV_Image(&mat, image_in));
	Mat Image = *mat, clean_img;
	flip(Image, Image, 1);

	if (globalDonutFlag == 0)
	{
		globalDonutRefImage = Image;
	}
	else
	{
		clean_img = Image;
	}

	// change donut size - press "d"
	if (waitKey(2) == 100)
	{
		if (globalDonutChange == 0)
			globalDonutChange = 1;
		else
			globalDonutChange = 0;
	}
	if (draw_pop_donuts(Image.cols, Image.rows, &Image))
		return VX_FAILURE;

	std::ostringstream statusStr;
	if (poppedDonuts >= 1000)
	{
		statusStr << "Congratulations! Click any Key to Contiue Popping!";
		putText(Image, statusStr.str(), cvPoint(5, int(Image.rows / 2)), FONT_HERSHEY_COMPLEX_SMALL, 1, cvScalar(200, 200, 250), 1, CV_AA);
	}
	else
	{
		statusStr << "Bubbles Popped: " << poppedDonuts;
		putText(Image, statusStr.str(), cvPoint(30, 30), FONT_HERSHEY_COMPLEX_SMALL, 1.2, cvScalar(200, 200, 250), 1, CV_AA);
	}

	// Converting OpenCV Mat into VX Image
	STATUS_ERROR_CHECK(CV_to_VX_Image(image_out, &Image));

	if (globalDonutFlag == 0)
		globalDonutFlag++;
	else
	{
		globalDonutRefImage = clean_img;
		globalDonutFlag++;
	}

	return status;
}

/************************************************************************************************************
Function to Register the Kernel for Publish
*************************************************************************************************************/
vx_status VX_donut_pop_Register(vx_context context)
{
	vx_status status = VX_SUCCESS;
	vx_kernel kernel = vxAddKernel(context,
								   "org.pop.donut_pop",
								   VX_KERNEL_EXT_POP_DONUT_POP,
								   VX_bubbles_Kernel,
								   2,
								   VX_bubbles_InputValidator,
								   VX_bubbles_OutputValidator,
								   nullptr,
								   nullptr);

	if (kernel)
	{
		PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
		PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 1, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
		PARAM_ERROR_CHECK(vxFinalizeKernel(kernel));
	}

	if (status != VX_SUCCESS)
	{
	exit:
		vxRemoveKernel(kernel);
		return VX_FAILURE;
	}

	return status;
}

/*
 * linked_list.c
 * Author: Kiriti Nagesh Gowda
 */

// Insert a variable function
donutNode *donut_insert(donutNode *head, donutNode *x)
{
	donutNode *temp;
	donutNode *temp1 = x;

	if (head == NULL)
		head = temp1;

	else
	{
		temp = head;
		while (temp->next != NULL)
		{
			temp = temp->next;
		}
		temp->next = temp1;
	}
	return (head);
}

// Delete a node from the list
donutNode *donut_position_delete(donutNode *head, int p)
{
	donutNode *temp;
	donutNode *temp1;
	int count = 2;
	temp = head;

	if (temp == NULL || p <= 0)
	{
		printf("The List is empty or the position is invalid\n");
		return (head);
	}

	if (p == 1)
	{
		head = temp->next;
		free(temp);
		return (head);
	}
	while (temp != NULL)
	{
		if (count == (p))
		{
			temp1 = temp->next;
			temp->next = temp1->next;
			free(temp1);
			return (head);
		}
		temp = temp->next;

		if (temp == NULL)
			break;
		++count;
	}
	return head;
}

// clean node
donutNode *donut_clean_node(donutNode *head)
{

	donutNode *temp1;
	while (head != NULL)
	{
		temp1 = head->next;
		free(head);
		head = temp1;
	}
	return (head);
}