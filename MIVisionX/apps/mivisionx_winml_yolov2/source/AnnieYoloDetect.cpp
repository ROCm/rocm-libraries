/*
Copyright (c) 2017 - 2024 Advanced Micro Devices, Inc. All rights reserved.

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

#include "AnnieYoloDetect.h"
#include <fstream>

using namespace std; 
using namespace cv;

#define ERROR_CHECK_STATUS( status ) { \
        vx_status status_ = (status); \
        if(status_ != VX_SUCCESS) { \
            printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
            exit(1); \
        } \
}

#define ERROR_CHECK_OBJECT( obj ) { \
        vx_status status_ = vxGetStatus((vx_reference)(obj)); \
        if(status_ != VX_SUCCESS) { \
            printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
            exit(1); \
        } \
}

AnnieYoloDetect::AnnieYoloDetect(string input, string modelLoc, int confidence, int mode) : mInput(input), mModelLoc(modelLoc), mConfidence(confidence), mMode(mode) {
	mRegion = make_unique<Region>();
	fstream file(modelLoc);
	if (!file.is_open()) {
		cout << "Unable to open the model " << mModelLoc << ". Please check the model directory." << endl;
		file.close();
		exit(1);
	}
	file.close();
};

AnnieYoloDetect::~AnnieYoloDetect() {};

void AnnieYoloDetect::detect() {

	vx_context context = vxCreateContext();
	ERROR_CHECK_OBJECT(context);
	
	vx_size out_dim[4] = { 13, 13, 125, 1 };
	vx_tensor output = vxCreateTensor(context, 4, out_dim, VX_TYPE_FLOAT32, 0);
	ERROR_CHECK_OBJECT(output);

	vx_graph graph = vxCreateGraph(context);
	ERROR_CHECK_OBJECT(graph);

	vx_image input_image = vxCreateImage(context, mWidth, mHeight, VX_DF_IMAGE_RGB);
	ERROR_CHECK_OBJECT(input_image);

	vx_float32 num_a = 1.0, num_b = 0.0;
	vx_bool reverse = 0;
	vx_scalar a = vxCreateScalar(context, VX_TYPE_FLOAT32, &num_a);
	vx_scalar b = vxCreateScalar(context, VX_TYPE_FLOAT32, &num_b);
	vx_scalar rev = vxCreateScalar(context, VX_TYPE_BOOL, &reverse);
	ERROR_CHECK_OBJECT(a);
	ERROR_CHECK_OBJECT(b);
	ERROR_CHECK_OBJECT(rev);

	char modelLocBuf[2048];
	int n = sprintf(modelLocBuf, "%s", mModelLoc.c_str());

	string modelIn = "image";
	char modelInBuf[2048];
	n = sprintf(modelInBuf, "%s", modelIn.c_str());

	string modelOut = "grid";
	char modelOutBuf[2048];
	n = sprintf(modelOutBuf, "%s", modelOut.c_str());

	vx_scalar modelLocation = vxCreateScalar(context, VX_TYPE_STRING_AMD, &mModelLoc);
	ERROR_CHECK_STATUS(vxWriteScalarValue(modelLocation, modelLocBuf));
	vx_scalar modelInputName = vxCreateScalar(context, VX_TYPE_STRING_AMD, &modelIn);
	ERROR_CHECK_STATUS(vxWriteScalarValue(modelInputName, modelInBuf));
	vx_scalar modelOutputName = vxCreateScalar(context, VX_TYPE_STRING_AMD, &modelOut);
	ERROR_CHECK_STATUS(vxWriteScalarValue(modelOutputName, modelOutBuf));

	vx_size dimension[4] = { 416, 416, 3, 1 };
	vx_tensor input_tensor = vxCreateTensor(context, 4, dimension, VX_TYPE_FLOAT32, 0);

	vx_int32 device = 0;
	vx_scalar deviceKind = vxCreateScalar(context, VX_TYPE_INT32, &device);
	
	vx_array setup_array_yolo = vxCreateArray(context, VX_TYPE_SIZE, sizeof(VX_TYPE_SIZE));
	ERROR_CHECK_OBJECT(setup_array_yolo);

	vxLoadKernels(context, "vx_winml");

	vx_node nodes[] =
	{
		vxExtWinMLNode_convertImageToTensor(graph, input_image, input_tensor, a, b, rev),
		vxExtWinMLNode_OnnxToMivisionX(graph, modelLocation, modelInputName, modelOutputName, input_tensor, setup_array_yolo, output, deviceKind),
	};

	for (vx_size i = 0; i < sizeof(nodes) / sizeof(nodes[0]); i++)
	{
		ERROR_CHECK_OBJECT(nodes[i]);
		ERROR_CHECK_STATUS(vxReleaseNode(&nodes[i]));
	}

	ERROR_CHECK_STATUS(vxVerifyGraph(graph));

	Mat input, img_cp;
	vector<DetectedObject> results;

	vx_size num_of_dims;
	vx_map_id map_id;
	vx_size stride[4];
	int classes = 20;
	float threshold = static_cast<float>(0.18);
	float nms = static_cast<float>(0.4);
	int targetBlockwd = 13;

	float * ptr = nullptr;

	if (mMode == 0) {
		input = imread(mInput);
		if (input.empty()) {
			cout << "Unable to open the image: " << mInput << endl;
			exit(1);
		}
		img_cp = input.clone();
		resize(input, input, Size(mWidth, mHeight));
		vx_rectangle_t cv_image_region;
		cv_image_region.start_x = 0;
		cv_image_region.start_y = 0;
		cv_image_region.end_x = mWidth;
		cv_image_region.end_y = mHeight;
		vx_imagepatch_addressing_t cv_image_layout;
		cv_image_layout.stride_x = 3;
		cv_image_layout.stride_y = (vx_int32)input.step;
		vx_uint8 * cv_image_buffer = input.data;

		ERROR_CHECK_STATUS(vxCopyImagePatch(input_image, &cv_image_region, 0,
			&cv_image_layout, cv_image_buffer,
			VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));

		ERROR_CHECK_STATUS(vxProcessGraph(graph));

		ERROR_CHECK_STATUS(vxQueryTensor(output, VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)));
		ERROR_CHECK_STATUS(vxMapTensorPatch(output, num_of_dims, NULL, NULL, &map_id, stride, (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

		mRegion->GetDetections(ptr, (int)out_dim[2], (int)out_dim[1], (int)out_dim[0], classes, input.cols, input.rows, threshold, nms, targetBlockwd, results);
		ERROR_CHECK_STATUS(vxUnmapTensorPatch(output, map_id));
		mVisualize = make_unique<Visualize>(img_cp, mConfidence, results);
		mVisualize->show();
		waitKey(0);
	}
	else {
		VideoCapture cap;
		if (mMode == 1) {
			cap.open(atoi(mInput.c_str()));
			if (!cap.isOpened()) {
				cout << "Unable to open the camera" << endl;
				exit(1);
			}
		}
		else if (mMode == 2) {
			cap.open(mInput);
			if (!cap.isOpened()) {
				cout << "Unable to open the video: " << mInput << endl;
				exit(1);
			}
		}
		for (;;) {
			cap >> input;
			img_cp = input.clone();
			resize(input, input, Size(mWidth, mHeight));
			vx_rectangle_t cv_image_region;
			cv_image_region.start_x = 0;
			cv_image_region.start_y = 0;
			cv_image_region.end_x = mWidth;
			cv_image_region.end_y = mHeight;
			vx_imagepatch_addressing_t cv_image_layout;
			cv_image_layout.stride_x = 3;
			cv_image_layout.stride_y = (vx_int32)input.step;
			vx_uint8 * cv_image_buffer = input.data;

			ERROR_CHECK_STATUS(vxCopyImagePatch(input_image, &cv_image_region, 0,
				&cv_image_layout, cv_image_buffer,
				VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));

			ERROR_CHECK_STATUS(vxProcessGraph(graph));
			ERROR_CHECK_STATUS(vxQueryTensor(output, VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)));
			ERROR_CHECK_STATUS(vxMapTensorPatch(output, num_of_dims, NULL, NULL, &map_id, stride, (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

			mRegion->GetDetections(ptr, (int)out_dim[2], (int)out_dim[1], (int)out_dim[0], classes, input.cols, input.rows, (float)threshold, nms, targetBlockwd, results);
			ERROR_CHECK_STATUS(vxUnmapTensorPatch(output, map_id));
			mVisualize = make_unique<Visualize>(img_cp, mConfidence, results);
			mVisualize->show();
			mVisualize->LegendImage();
			if (waitKey(30) >= 0) break;
		}
	}
	
	ERROR_CHECK_STATUS(vxReleaseGraph(&graph));
	ERROR_CHECK_STATUS(vxReleaseImage(&input_image));
	ERROR_CHECK_STATUS(vxReleaseContext(&context));
}