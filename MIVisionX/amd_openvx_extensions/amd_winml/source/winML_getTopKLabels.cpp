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

#include "internal_publishKernels.h"

vector<string> labels;

// load label.txt
static void LoadLabels(string labelsFilePath)
{
	// Parse labels from labels file.  We know the file's entries are already sorted in order.
	ifstream labelFile{ labelsFilePath, ifstream::in };
	if (labelFile.fail())
	{
		printf("ERROR:failed to load the %s file.  Make sure it exists in the same folder as the app\r\n", labelsFilePath.c_str());
		exit(EXIT_FAILURE);
	}

	std::string s;
	while (std::getline(labelFile, s, ','))
	{
		int labelValue = atoi(s.c_str());
		if (labelValue >= labels.size())
		{
			labels.resize(labelValue + 1);
		}
		std::getline(labelFile, s);
		labels[labelValue] = s;
	}
}

/************************************************************************************************************
input parameter validator.
param [in] node The handle to the node.
param [in] index The index of the parameter to validate.
*************************************************************************************************************/
static vx_status VX_CALLBACK WINML_getTopKLabels_InputValidator(vx_node node, vx_uint32 index)
{
        vx_status status = VX_SUCCESS;
        vx_parameter param = vxGetParameterByIndex(node, index);

		if (index == 0)
		{
			vx_size num_dims;
			vx_enum type;
			vx_tensor inputTensor;
			STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &inputTensor, sizeof(vx_tensor)));
			STATUS_ERROR_CHECK(vxQueryTensor(inputTensor, VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
			STATUS_ERROR_CHECK(vxQueryTensor(inputTensor, VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
			if (num_dims != 4 && num_dims != 2)
			{
				printf("validate: getTopKLabels: #0 num_dims=%zd (must be 4||2)\n", num_dims);
				return VX_ERROR_INVALID_DIMENSION;
			}
			if ((type != VX_TYPE_FLOAT32) && (type != VX_TYPE_FLOAT16))
			{
				printf("validate: getTopKLabels: #0 tensor type=%d (not float/float16)\n", type);
				return VX_ERROR_INVALID_TYPE;
			}
			STATUS_ERROR_CHECK(vxReleaseTensor(&inputTensor));
		}
		else if (index == 1)
		{
			vx_enum type;
			vx_scalar inputScalar;
			STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &inputScalar, sizeof(vx_scalar)));
			STATUS_ERROR_CHECK(vxQueryScalar(inputScalar, VX_SCALAR_TYPE, &type, sizeof(type)));
			if (type != VX_TYPE_STRING_AMD)
			{
				printf("validate: getTopKLabels: #1 scalar type=%d (not VX_TYPE_STRING_AMD)\n", type);
				return VX_ERROR_INVALID_TYPE;
			}
			STATUS_ERROR_CHECK(vxReleaseScalar(&inputScalar));
		}

		STATUS_ERROR_CHECK(vxReleaseParameter(&param));
        return status;
}

/************************************************************************************************************
output parameter validator.
param [in] node The handle to the node.
param [in] index The index of the parameter to validate.
param [out] meta
*************************************************************************************************************/
static vx_status VX_CALLBACK WINML_getTopKLabels_OutputValidator(vx_node node, vx_uint32 index, vx_meta_format meta)
{
        vx_status status = VX_SUCCESS;
        vx_parameter param = vxGetParameterByIndex(node, index);

		if (index == 2)
		{
			vx_enum type;
			vx_scalar inputScalar;
			STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &inputScalar, sizeof(vx_scalar)));
			STATUS_ERROR_CHECK(vxQueryScalar(inputScalar, VX_SCALAR_TYPE, &type, sizeof(type)));
			if (type != VX_TYPE_STRING_AMD)
			{
				printf("validate: getTopKLabels: #2 scalar type=%d (not VX_TYPE_STRING_AMD)\n", type);
				return VX_ERROR_INVALID_TYPE;
			}
			STATUS_ERROR_CHECK(vxReleaseScalar(&inputScalar));

			STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(meta, VX_SCALAR_TYPE, &type, sizeof(type)));
		}
		if (index == 3 || index == 4 || index == 5 || index == 6)
		{
			if (param) {
				vx_enum type;
				vx_scalar inputScalar;
				STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &inputScalar, sizeof(vx_scalar)));
				STATUS_ERROR_CHECK(vxQueryScalar(inputScalar, VX_SCALAR_TYPE, &type, sizeof(type)));
				if (type != VX_TYPE_STRING_AMD)
				{
					printf("validate: getTopKLabels: #3/#4/#5/#6 scalar type=%d (not VX_TYPE_STRING_AMD)\n", type);
					return VX_ERROR_INVALID_TYPE;
				}
				STATUS_ERROR_CHECK(vxReleaseScalar(&inputScalar));

				STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(meta, VX_SCALAR_TYPE, &type, sizeof(type)));
			}
		}
        return status;
}

/************************************************************************************************************
Initialize Kernel
param [in] node The handle to the node.
param [in] paramete.
param [in] num.
*************************************************************************************************************/
static vx_status VX_CALLBACK WINML_getTopKLabels_Initialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
        vx_status status = VX_SUCCESS;
		vx_scalar inputLabelScalar = (vx_scalar)parameters[1];

		// get model input tensor name
		char labelName[1024];
		STATUS_ERROR_CHECK(vxReadScalarValue(inputLabelScalar, labelName));
		std::string labelLocation(labelName);
		// load label
		LoadLabels(labelLocation);

        return status;
}

/************************************************************************************************************
Uninitialize Kernel
param [in] node The handle to the node.
param [in] paramete.
param [in] num.
*************************************************************************************************************/
static vx_status VX_CALLBACK WINML_getTopKLabels_Uninitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
        vx_status status = VX_SUCCESS;
		
        return status;
}

/************************************************************************************************************
Execution Kernel
param [in] node The handle to the node.
param [in] parameter.
param [in] num.
*************************************************************************************************************/
static vx_status VX_CALLBACK WINML_getTopKLabels_Kernel(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
        vx_status status = VX_SUCCESS;

		// input tensor and scalar
		vx_tensor inputTensor = (vx_tensor)parameters[0];

		// get tensor values
		vx_enum usage = VX_READ_ONLY;
		vx_size num_of_dims, inputDims[4] = { 1, 1, 1, 1 }, stride[4];
		vx_map_id map_id;
		float *ptr = nullptr;

		status = (vxQueryTensor(inputTensor, VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)));
		if (status) { std::cerr << "ERROR: vxQueryTensor(VX_TENSOR_NUMBER_OF_DIMS) failed for inputTensor" << std::endl; return status; }
		status = (vxQueryTensor(inputTensor, VX_TENSOR_DIMS, &inputDims, sizeof(inputDims[0])*num_of_dims));
		if (status) { std::cerr << "ERROR: vxQueryTensor(VX_TENSOR_DIMS) failed for inputTensor" << std::endl; return status; }

		status = vxMapTensorPatch(inputTensor, num_of_dims, nullptr, nullptr, &map_id, stride, (void **)&ptr, usage, VX_MEMORY_TYPE_HOST);
		if (status) { std::cerr << "ERROR: vxMapTensorPatch() failed for inputTensor" << std::endl; return status; }

		vx_size inputTensorSize = inputDims[0] * inputDims[1] * inputDims[2] * inputDims[3];
		vector<float> inputPtr;
		inputPtr.resize(inputTensorSize);
		memcpy(&inputPtr[0], ptr, (inputTensorSize * sizeof(float)));
		status = vxUnmapTensorPatch(inputTensor, map_id);
		if (status) { std::cerr << "ERROR: vxUnmapTensorPatch() failed for inputTensor" << std::endl; return status; }

		// Find the top K probabilities
		vector<float> topProbabilities(5);
		vector<int> topProbabilityLabelIndexes(5);
		// list of inputTensorSize options, with probabilities for each, loop through all
		for (uint32_t i = 0; i < inputTensorSize; i++)
		{
			// is it one of the top 5
			for (int j = 0; j < 5; j++)
			{
				if (inputPtr[i] > topProbabilities[j])
				{
					topProbabilityLabelIndexes[j] = i;
					topProbabilities[j] = inputPtr[i];
					break;
				}
			}
		}

		// print final values
		char outputBuffer[2048];
		int n = snprintf(outputBuffer, sizeof(outputBuffer),"%s - %.3f", labels[topProbabilityLabelIndexes[0]].c_str(), topProbabilities[0]);

		vx_scalar outputLabelScalar = (vx_scalar)parameters[2];
		STATUS_ERROR_CHECK(vxWriteScalarValue(outputLabelScalar, outputBuffer));

		vx_scalar outputLabelScalar_1 = (vx_scalar)parameters[3];
		vx_scalar outputLabelScalar_2 = (vx_scalar)parameters[4];
		vx_scalar outputLabelScalar_3 = (vx_scalar)parameters[5];
		vx_scalar outputLabelScalar_4 = (vx_scalar)parameters[6];

		if (outputLabelScalar_1) 
		{
			char outputBuffer_optional[2048];
			int n = snprintf(outputBuffer_optional, sizeof(outputBuffer_optional), "%s - %.3f", labels[topProbabilityLabelIndexes[1]].c_str(), topProbabilities[1]);
			STATUS_ERROR_CHECK(vxWriteScalarValue(outputLabelScalar_1, outputBuffer_optional));
			STATUS_ERROR_CHECK(vxReleaseScalar(&outputLabelScalar_1));
		}
		if (outputLabelScalar_2)
		{
			char outputBuffer_optional[2048];
			int n = snprintf(outputBuffer_optional, sizeof(outputBuffer_optional), "%s - %.3f", labels[topProbabilityLabelIndexes[2]].c_str(), topProbabilities[2]);
			STATUS_ERROR_CHECK(vxWriteScalarValue(outputLabelScalar_2, outputBuffer_optional));
			STATUS_ERROR_CHECK(vxReleaseScalar(&outputLabelScalar_2));
		}
		if (outputLabelScalar_3)
		{
			char outputBuffer_optional[2048];
			int n = snprintf(outputBuffer_optional, sizeof(outputBuffer_optional), "%s - %.3f", labels[topProbabilityLabelIndexes[3]].c_str(), topProbabilities[3]);
			STATUS_ERROR_CHECK(vxWriteScalarValue(outputLabelScalar_3, outputBuffer_optional));
			STATUS_ERROR_CHECK(vxReleaseScalar(&outputLabelScalar_3));
		}
		if (outputLabelScalar_4)
		{
			char outputBuffer_optional[2048];
			int n = snprintf(outputBuffer_optional, sizeof(outputBuffer_optional), "%s - %.3f", labels[topProbabilityLabelIndexes[4]].c_str(), topProbabilities[4]);
			STATUS_ERROR_CHECK(vxWriteScalarValue(outputLabelScalar_4, outputBuffer_optional));
			STATUS_ERROR_CHECK(vxReleaseScalar(&outputLabelScalar_4));
		}

		// release scalar
		STATUS_ERROR_CHECK(vxReleaseScalar(&outputLabelScalar));
		// release tensors
		STATUS_ERROR_CHECK(vxReleaseTensor(&inputTensor));

        return status;
}

/************************************************************************************************************
Function to Register the Kernel for Publish
*************************************************************************************************************/
vx_status  WINML_getTopKLabels_Register(vx_context context)
{
        vx_status status = VX_SUCCESS;
        vx_kernel kernel = vxAddKernel(context,
                "com.winml.get_top_k_labels",
                VX_KERNEL_WINML_GET_TOP_K_LABELS,
                WINML_getTopKLabels_Kernel,
                7,
                WINML_getTopKLabels_InputValidator,
                WINML_getTopKLabels_OutputValidator,
                WINML_getTopKLabels_Initialize,
                WINML_getTopKLabels_Uninitialize);

        if (kernel)
        {
                PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
                PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
                PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 2, VX_OUTPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
				PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 3, VX_OUTPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL));
				PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 4, VX_OUTPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL));
				PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 5, VX_OUTPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL));
				PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 6, VX_OUTPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL));
                PARAM_ERROR_CHECK(vxFinalizeKernel(kernel));
				PARAM_ERROR_CHECK(vxReleaseKernel(&kernel));
        }

        if (status != VX_SUCCESS)
        {
        exit:   vxRemoveKernel(kernel);  std::cerr << "ERROR: vxAddParameterToKernel() failed for get_top_k_labels" << std::endl; return VX_FAILURE;
        }

        return status;
}