/*
Copyright (c) 2019 - 2024 Advanced Micro Devices, Inc. All rights reserved.

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


#include "internal_winmlTunnel.h"

TensorFloat VX_to_ML_tensor(vx_tensor inputTensor)
{
	vx_status status = VX_SUCCESS;

	vx_enum usage = VX_READ_ONLY;
	vx_size num_of_dims, inputDims[4] = { 1, 1, 1, 1 }, stride[4];
	vx_map_id map_id;
	float *ptr = nullptr;
	vector<float> inputPtr;
	status = (vxQueryTensor(inputTensor, VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)));
	if (status) { std::cerr << "ERROR: vxQueryTensor(VX_TENSOR_NUMBER_OF_DIMS) failed for inputTensor" << std::endl; return nullptr; }
	status = (vxQueryTensor(inputTensor, VX_TENSOR_DIMS, &inputDims, sizeof(inputDims[0])*num_of_dims));
	if (status) { std::cerr << "ERROR: vxQueryTensor(VX_TENSOR_DIMS) failed for inputTensor" << std::endl; return nullptr; }

	status = vxMapTensorPatch(inputTensor, num_of_dims, nullptr, nullptr, &map_id, stride, (void **)&ptr, usage, VX_MEMORY_TYPE_HOST);
	if (status) { std::cerr << "ERROR: vxMapTensorPatch() failed for inputTensor" << std::endl; return nullptr; }

	vx_size inputTensorSize = inputDims[0] * inputDims[1] * inputDims[2] * inputDims[3];
	inputPtr.resize(inputTensorSize);
	memcpy(&inputPtr.front(), ptr, inputTensorSize * sizeof(float));

	status = vxUnmapTensorPatch(inputTensor, map_id);
	if (status) { std::cerr << "ERROR: vxUnmapTensorPatch() failed for inputTensor" << std::endl; return nullptr; }

	vector<int64_t> inputShape({ (int64_t)inputDims[3],  (int64_t)inputDims[2],  (int64_t)inputDims[1],  (int64_t)inputDims[0] });

	return TensorFloat::CreateFromIterable(inputShape, inputPtr);;
}

vx_status ML_to_VX_tensor(IVectorView<float> inputTensor, vx_tensor outputTensor)
{
	vx_status status = VX_SUCCESS;

	vx_enum usage = VX_WRITE_ONLY;
	vx_size num_of_dims, stride[4];
	vx_size outputDims[4] = { 1, 1, 1, 1 };
	vx_map_id map_id;
	float * outputPtr;
	STATUS_ERROR_CHECK(vxQueryTensor(outputTensor, VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)));
	STATUS_ERROR_CHECK(vxQueryTensor(outputTensor, VX_TENSOR_DIMS, &outputDims, sizeof(outputDims[0])*num_of_dims));
	vx_size outputTensorSize = outputDims[0] * outputDims[1] * outputDims[2] * outputDims[3];
	status = vxMapTensorPatch(outputTensor, num_of_dims, nullptr, nullptr, &map_id, stride, (void **)&outputPtr, usage, VX_MEMORY_TYPE_HOST);
	if (status) { std::cerr << "ERROR: vxMapTensorPatch() failed for outputTensor" << std::endl; return status; }

	for (int i = 0; i < outputTensorSize; i++) {
		outputPtr[i] = inputTensor.GetAt(i);
	}

	status = vxUnmapTensorPatch(outputTensor, map_id);
	if (status) { std::cerr << "ERROR: vxUnmapTensorPatch() failed for outputTensor" << std::endl; return status; }

	return status;
}