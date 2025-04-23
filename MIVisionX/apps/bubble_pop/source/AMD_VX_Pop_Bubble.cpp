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

int poppedBubbles = 0;
int globalBubblesflag = 0;
Mat globalBubblesRefImage;
int globalBubblesChange = 0;
Mat bubble_PNG = imread("image/b20.png", -1);

/************************************************************************************************************
Overlay Image Function
*************************************************************************************************************/
void overlayImage(const Mat &background, const Mat &foreground, Mat &output, Point2i location)
{
	background.copyTo(output);

	for (int y = std::max(location.y, 0); y < background.rows; ++y)
	{
		int fY = y - location.y;
		if (fY >= foreground.rows)
			break;

		for (int x = std::max(location.x, 0); x < background.cols; ++x)
		{
			int fX = x - location.x;

			if (fX >= foreground.cols)
				break;

			double opacity =
				((double)foreground.data[fY * foreground.step + fX * foreground.channels() + 3])

				/ 255.;

			for (int c = 0; opacity > 0 && c < output.channels(); ++c)
			{
				unsigned char foregroundPx =
					foreground.data[fY * foreground.step + fX * foreground.channels() + c];
				unsigned char backgroundPx =
					background.data[y * background.step + x * background.channels() + c];
				output.data[y * output.step + output.channels() * x + c] =
					backgroundPx * (1. - opacity) + foregroundPx * opacity;
			}
		}
	}
}

/************************************************************************************************************
Bubbles
*************************************************************************************************************/
class AMD_bubble_pop
{
private:
	int bubbleX, bubbleY, bubbleWidth, bubbleHeight;

public:
	AMD_bubble_pop(int bX, int bY, int bW, int bH)
	{
		bubbleX = bX;
		bubbleY = bY;
		bubbleWidth = bW;
		bubbleHeight = bH;
	}

	~AMD_bubble_pop()
	{
		bubbleX = 0;
		bubbleY = 0;
		bubbleWidth = 0;
		bubbleHeight = 0;
	}

	int update(int width, int height, Mat *Image)
	{
		int movementAmount = 0;
		if (globalBubblesflag > 10)
		{
			Mat diff_image;
			absdiff(*Image, globalBubblesRefImage, diff_image);
			blur(diff_image, diff_image, Size(3, 3));

			cv::erode(diff_image, diff_image, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
			cv::erode(diff_image, diff_image, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
			cv::dilate(diff_image, diff_image, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
			cv::dilate(diff_image, diff_image, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));

			cv::threshold(diff_image, diff_image, 160, 255, 0);

			unsigned char *input = (unsigned char *)(diff_image.data);
			int b;

			for (int x = bubbleX; x <= (bubbleX + (bubbleWidth - 1)); x++)
				for (int y = bubbleY; y <= (bubbleY + (bubbleWidth - 1)); y++)
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
			poppedBubbles++;
			return 1;
		}
		else
		{
			bubbleY += 5;

			if (bubbleY > height)
				return 1;

			Mat test_image;
			test_image = *Image;

			// draw bubbles
			if (bubble_PNG.empty())
			{
				printf("--image/b20.png-- Image not found\n");
				return -1;
			};
			overlayImage(test_image, bubble_PNG, *Image, cv::Point(bubbleX, bubbleY));

			return 0;
		}
	}
};

struct Linked_list_pop
{
	AMD_bubble_pop bubble;
	int data;
	struct Linked_list_pop *next;
};
typedef struct Linked_list_pop BubbleNode;

// Function Prototyping
BubbleNode *insert_pop(BubbleNode *head, BubbleNode *x);
BubbleNode *pop_position_delete(BubbleNode *head, int p);
BubbleNode *pop_clean_node(BubbleNode *head);
void draw_pop_bubbles(int, int, Mat *);
BubbleNode *POPbubble = NULL;

/************************************************************************************************************
Draw Bubbles
*************************************************************************************************************/
void draw_pop_bubbles(int width, int height, Mat *Image)
{
	static int count = 0;

	int randx = rand() % (width + 1);
	AMD_bubble_pop new_element = AMD_bubble_pop(randx, 0, 20, 20);

	BubbleNode *temp = (BubbleNode *)malloc(sizeof(BubbleNode));
	temp->bubble = new_element;
	temp->next = NULL;
	POPbubble = insert_pop(POPbubble, temp);
	count++;

	BubbleNode *_bubbles;
	_bubbles = POPbubble;
	int K = 0;
	int flag = 0;

	while (_bubbles != NULL)
	{
		K++;
		flag = 0;

		if (_bubbles->bubble.update(width, height, Image) == 1)
		{
			_bubbles = _bubbles->next;
			POPbubble = pop_position_delete(POPbubble, K);
			count--;
			K--;
			flag = 1;
		}

		if (flag == 0)
			_bubbles = _bubbles->next;
	}

	return;
}

/************************************************************************************************************
input parameter validator.
param [in] node The handle to the node.
param [in] index The index of the parameter to validate.
*************************************************************************************************************/
static vx_status VX_CALLBACK VX_bubbles_InputValidator(vx_node node, vx_uint32 index)
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
static vx_status VX_CALLBACK VX_bubbles_OutputValidator(vx_node node, vx_uint32 index, vx_meta_format meta)
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
static vx_status VX_CALLBACK VX_bubbles_Kernel(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	vx_status status = VX_SUCCESS;

	vx_image image_in = (vx_image)parameters[0];
	vx_image image_out = (vx_image)parameters[1];
	Mat *mat, bl;

	// wait to restart - press any key
	if (poppedBubbles >= 1015)
	{
		poppedBubbles = 0;
		waitKey(0);
	}

	// Converting VX Image to OpenCV Mat
	STATUS_ERROR_CHECK(VX_to_CV_Image(&mat, image_in));
	Mat Image = *mat, clean_img;
	flip(Image, Image, 1);

	if (globalBubblesflag == 0)
	{
		globalBubblesRefImage = Image;
	}
	else
	{
		clean_img = Image;
	}

	draw_pop_bubbles(Image.cols, Image.rows, &Image);

	std::ostringstream statusStr;
	if (poppedBubbles >= 1000)
	{
		statusStr << "Congratulations! Click any Key to Contiue Popping!";
		putText(Image, statusStr.str(), cvPoint(5, int(Image.rows / 2)), FONT_HERSHEY_COMPLEX_SMALL, 1, cvScalar(200, 200, 250), 1, CV_AA);
	}
	else
	{
		statusStr << "Bubbles Popped: " << poppedBubbles;
		putText(Image, statusStr.str(), cvPoint(30, 30), FONT_HERSHEY_COMPLEX_SMALL, 1.2, cvScalar(200, 200, 250), 1, CV_AA);
	}

	// Converting OpenCV Mat into VX Image
	STATUS_ERROR_CHECK(CV_to_VX_Image(image_out, &Image));

	if (globalBubblesflag == 0)
		globalBubblesflag++;
	else
	{
		globalBubblesRefImage = clean_img;
		globalBubblesflag++;
	}

	return status;
}

/************************************************************************************************************
Function to Register the Kernel for Publish
*************************************************************************************************************/
vx_status VX_bubbles_pop_Register(vx_context context)
{
	vx_status status = VX_SUCCESS;
	vx_kernel kernel = vxAddKernel(context,
								   "org.pop.bubble_pop",
								   VX_KERNEL_EXT_POP_BUBBLE_POP,
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
BubbleNode *insert_pop(BubbleNode *head, BubbleNode *x)
{
	BubbleNode *temp;
	BubbleNode *temp1 = x;

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
BubbleNode *pop_position_delete(BubbleNode *head, int p)
{
	BubbleNode *temp;
	BubbleNode *temp1;
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
BubbleNode *pop_clean_node(BubbleNode *head)
{

	BubbleNode *temp1;
	while (head != NULL)
	{
		temp1 = head->next;
		free(head);
		head = temp1;
	}
	return (head);
}
