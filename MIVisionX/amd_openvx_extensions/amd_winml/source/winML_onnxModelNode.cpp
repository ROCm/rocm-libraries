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
#include "vx_winml.h"

// deploy deive kinds
const LearningModelDeviceKind deviceKindArray[5] =	{	LearningModelDeviceKind::Default,
								LearningModelDeviceKind::Cpu,
								LearningModelDeviceKind::DirectX,
								LearningModelDeviceKind::DirectXHighPerformance,
								LearningModelDeviceKind::DirectXMinPower
							};
// Node Struct variables
struct learning_model {
	LearningModel model = nullptr;
	LearningModelSession session = nullptr;
	LearningModelBinding binding = nullptr;
};


// load ONNX model to WinML
static void LoadModelFromPath(hstring modelLocation, learning_model *models)
{
	models->model = LearningModel::LoadFromFilePath(modelLocation);
}

// bind the ONNX model
static void BindModel(hstring inputTensorName, hstring outputTensorName, int64_t * inputDim, int64_t *outputDim, int deviceIndex, learning_model *models)
{
	// create a session and binding
	models->session = LearningModelSession{ models->model, LearningModelDevice(deviceKindArray[deviceIndex]) };
	models->binding = LearningModelBinding{ models->session };

	// bind the intput image (bind input in kernel)
	vector<int64_t> inputShape({ inputDim[3],  inputDim[2],  inputDim[1],  inputDim[0] });
	//inputTensorElement = TensorFloat::Create(inputShape);
	//binding.Bind(inputTensorName, inputTensorElement);

	// bind the output
	vector<int64_t> outputShape({ outputDim[3],  outputDim[2],  outputDim[1],  outputDim[0] });
	models->binding.Bind(outputTensorName, TensorFloat::Create(outputShape));
}

// close and clear session
static void closeWinmlNode(learning_model *models)
{
	models->binding.Clear();
	models->session.Close();
	models->model.Close();
}

/************************************************************************************************************
input parameter validator.
param [in] node The handle to the node.
param [in] index The index of the parameter to validate.
*************************************************************************************************************/
static vx_status VX_CALLBACK WINML_OnnxToMivisionX_InputValidator(vx_node node, vx_uint32 index)
{
        vx_status status = VX_SUCCESS;
        vx_parameter param = vxGetParameterByIndex(node, index);

        if (index == 0 || index == 1 || index == 2)
        {
                vx_enum type;
                vx_scalar inputScalar;
                STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &inputScalar, sizeof(vx_scalar)));
                STATUS_ERROR_CHECK(vxQueryScalar(inputScalar, VX_SCALAR_TYPE, &type, sizeof(type)));
                if (type != VX_TYPE_STRING_AMD)
                {
                        printf("validate: OnnxToMivisionX: #0 or #1 or #2 scalar type=%d (not VX_TYPE_STRING_AMD)\n", type);
                        return VX_ERROR_INVALID_TYPE;
                }
		STATUS_ERROR_CHECK(vxReleaseScalar(&inputScalar));
        }
        else if (index == 3)
        {
                vx_size num_dims;
                vx_enum type;
                vx_tensor inputTensor;
                STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &inputTensor, sizeof(vx_tensor)));
                STATUS_ERROR_CHECK(vxQueryTensor(inputTensor, VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
                STATUS_ERROR_CHECK(vxQueryTensor(inputTensor, VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
                if (num_dims != 4)
                {
                        printf("validate: OnnxToMivisionX: #3 num_dims=%zd (must be 4)\n", num_dims);
                        return VX_ERROR_INVALID_DIMENSION;
                }
                if ((type != VX_TYPE_FLOAT32) && (type != VX_TYPE_FLOAT16))
                {
                        printf("validate: OnnxToMivisionX: #3 tensor type=%d (not float/float16)\n", type);
                        return VX_ERROR_INVALID_TYPE;
                }
		STATUS_ERROR_CHECK(vxReleaseTensor(&inputTensor));
        }
	else if (index == 4)
	{
		vx_size size;
		vx_enum type;
		vx_array setupArray;
		STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &setupArray, sizeof(vx_array)));
		STATUS_ERROR_CHECK(vxQueryArray(setupArray, VX_ARRAY_ATTRIBUTE_ITEMTYPE, &type, sizeof(type)));
		if (type != VX_TYPE_SIZE)  return VX_ERROR_INVALID_TYPE;
		STATUS_ERROR_CHECK(vxReleaseArray(&setupArray));
	}
	else if (index == 6)
	{
		if (param)
		{
			vx_enum type;
			vx_scalar inputScalar;
			STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &inputScalar, sizeof(vx_scalar)));
			STATUS_ERROR_CHECK(vxQueryScalar(inputScalar, VX_SCALAR_TYPE, &type, sizeof(type)));
			if (type != VX_TYPE_INT32)
			{
				printf("validate: OnnxToMivisionX: #5 scalar type=%d (not VX_TYPE_INT32)\n", type);
				return VX_ERROR_INVALID_TYPE;
			}
			STATUS_ERROR_CHECK(vxReleaseScalar(&inputScalar));
		}
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
static vx_status VX_CALLBACK WINML_OnnxToMivisionX_OutputValidator(vx_node node, vx_uint32 index, vx_meta_format meta)
{
        vx_status status = VX_SUCCESS;
        vx_parameter param = vxGetParameterByIndex(node, index);

        if (index == 5)
        {
		vx_size num_dims;
		vx_enum type;
		vx_size output_dims_2[2], output_dims_4[4];
		vx_tensor outputTensor;
		STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &outputTensor, sizeof(vx_tensor)));
		STATUS_ERROR_CHECK(vxQueryTensor(outputTensor, VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
		STATUS_ERROR_CHECK(vxQueryTensor(outputTensor, VX_TENSOR_DATA_TYPE, &type, sizeof(type)));

		if (num_dims == 2)
		{
				STATUS_ERROR_CHECK(vxQueryTensor(outputTensor, VX_TENSOR_DIMS, output_dims_2, sizeof(output_dims_2)));
				STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(meta, VX_TENSOR_DIMS, output_dims_2, sizeof(output_dims_2)));
		}
		else if (num_dims == 4)
		{
				STATUS_ERROR_CHECK(vxQueryTensor(outputTensor, VX_TENSOR_DIMS, output_dims_4, sizeof(output_dims_4)))
				STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(meta, VX_TENSOR_DIMS, output_dims_4, sizeof(output_dims_4)))
		}
		else
		{
				printf("Output validate: OnnxToMivisionX: #4 num_dims=%zd (must be 2/4)\n", num_dims);
				return VX_ERROR_INVALID_DIMENSION;
		}

		if ((type != VX_TYPE_FLOAT32) && (type != VX_TYPE_FLOAT16))
		{
				printf("Output validate: OnnxToMivisionX: #4 tensor type=%d (not float/float16)\n", type);
				return VX_ERROR_INVALID_TYPE;
		}

		STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(meta, VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
		STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(meta, VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));

		STATUS_ERROR_CHECK(vxReleaseTensor(&outputTensor));
        }
		
        return status;
}

/************************************************************************************************************
Initialize Kernel
param [in] node The handle to the node.
param [in] paramete.
param [in] num.
*************************************************************************************************************/
static vx_status VX_CALLBACK WINML_OnnxToMivisionX_Initialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
        vx_status status = VX_SUCCESS;

	// Read scalar strings
	char modelFileLocation[1024], modelInputName[1024], modelOutputName[1024];
	vx_scalar modelLocationScalar = (vx_scalar)parameters[0];
	vx_scalar inputNameScalar = (vx_scalar)parameters[1];
	vx_scalar outputNameScalar = (vx_scalar)parameters[2];
	STATUS_ERROR_CHECK(vxReadScalarValue(modelLocationScalar, modelFileLocation));
	STATUS_ERROR_CHECK(vxReadScalarValue(inputNameScalar, modelInputName));
	STATUS_ERROR_CHECK(vxReadScalarValue(outputNameScalar, modelOutputName));

	// read optional device kind index
	vx_int32 deviceKindIndex = 0;
	vx_scalar deviceKindScalar = (vx_scalar)parameters[6];
	if(deviceKindScalar)
		STATUS_ERROR_CHECK(vxReadScalarValue(deviceKindScalar, &deviceKindIndex));

	// get model location
	std::string Model(modelFileLocation);
	wstring wModel(Model.begin(), Model.end());
	hstring ModelLocation = wModel.c_str();
	// get model input tensor name
	std::string inputName(modelInputName);
	wstring wInputName(inputName.begin(), inputName.end());
	hstring ModelInputTensorName = wInputName.c_str();
	// get model output tensor name
	std::string outputName(modelOutputName);
	wstring wOutputName(outputName.begin(), outputName.end());
	hstring ModelOutputTensorName = wOutputName.c_str();
	// get model input tensor dims
	int64_t inputDims[4] = { 1 };
	vx_size input_dims[4] = { 1, 1, 1, 1 };
	vx_tensor inputTensor = (vx_tensor)parameters[3];
	STATUS_ERROR_CHECK(vxQueryTensor(inputTensor, VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));
	inputDims[0] = (int64_t)input_dims[0];
	inputDims[1] = (int64_t)input_dims[1];
	inputDims[2] = (int64_t)input_dims[2];
	inputDims[3] = (int64_t)input_dims[3];
	// get model output tensor dim
	int64_t outputDims[4] = { 1 };
	vx_size num_dims;
	vx_size output_dims_2[2] = { 1, 1}, output_dims_4[4] = { 1, 1, 1, 1 };
	vx_tensor outputTensor = (vx_tensor)parameters[5];
	STATUS_ERROR_CHECK(vxQueryTensor(outputTensor, VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
	if (num_dims == 2)
	{
		STATUS_ERROR_CHECK(vxQueryTensor(outputTensor, VX_TENSOR_DIMS, output_dims_2, sizeof(output_dims_2)));
		outputDims[0] = output_dims_2[0];
		outputDims[1] = output_dims_2[1];
	}
	else if (num_dims == 4)
	{
		STATUS_ERROR_CHECK(vxQueryTensor(outputTensor, VX_TENSOR_DIMS, output_dims_4, sizeof(output_dims_4)))
		outputDims[0] = output_dims_4[0];
		outputDims[1] = output_dims_4[1];
		outputDims[2] = output_dims_4[2];
		outputDims[3] = output_dims_4[3];
	}

	vx_array model_array = (vx_array)parameters[4];

	vx_size size = 0;

	learning_model *models = new learning_model;
	memset(models, 0, sizeof(*models));

	// load model location
	LoadModelFromPath(ModelLocation, models);

	// bind the model
	BindModel(ModelInputTensorName, ModelOutputTensorName, inputDims, outputDims, deviceKindIndex, models);

	void *model_ptr = &models;

	//vx_size size = 0;
	vx_size size_p = 0, num_items = 0;
	vxAddArrayItems(model_array, 1, model_ptr, sizeof(VX_TYPE_SIZE));
	vxQueryArray(model_array, VX_ARRAY_ATTRIBUTE_ITEMSIZE, &size, sizeof(size));

	vxSetParameterByIndex(node, 4, (vx_reference)model_array);
	vxCopyArrayRange(model_array, 0, size, 0, parameters[4], VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

	// release scalars
	STATUS_ERROR_CHECK(vxReleaseScalar(&modelLocationScalar));
	STATUS_ERROR_CHECK(vxReleaseScalar(&inputNameScalar));
	STATUS_ERROR_CHECK(vxReleaseScalar(&outputNameScalar));
	if (deviceKindScalar)
		STATUS_ERROR_CHECK(vxReleaseScalar(&deviceKindScalar));
	// release tensors
	STATUS_ERROR_CHECK(vxReleaseTensor(&inputTensor));
	STATUS_ERROR_CHECK(vxReleaseTensor(&outputTensor));
	//release array
	STATUS_ERROR_CHECK(vxReleaseArray(&model_array));

        return status;
}

/************************************************************************************************************
Uninitialize Kernel
param [in] node The handle to the node.
param [in] paramete.
param [in] num.
*************************************************************************************************************/
static vx_status VX_CALLBACK WINML_OnnxToMivisionX_Uninitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{	
	void **model_ptr = NULL;
	vx_array model_array = (vx_array)parameters[4];
	vx_size stride = 0ul;
	STATUS_ERROR_CHECK(vxAccessArrayRange((vx_array)model_array, 0, 1, &stride, (void**)&model_ptr, VX_READ_ONLY));
	STATUS_ERROR_CHECK(vxCommitArrayRange((vx_array)model_array, 0, 1, model_ptr));
	learning_model *model_struct = static_cast<learning_model *>(*model_ptr);
        vx_status status = VX_SUCCESS;
	// close and delete resources
	closeWinmlNode(model_struct);
	delete model_struct;
        return status;
}

/************************************************************************************************************
Execution Kernel
param [in] node The handle to the node.
param [in] parameter.
param [in] num.
*************************************************************************************************************/
static vx_status VX_CALLBACK WINML_OnnxToMivisionX_Kernel(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
        vx_status status = VX_SUCCESS;
		
	vx_array model_array = (vx_array)parameters[4];
	vx_size size = 0, num_items = 0;
	vx_enum type;

	STATUS_ERROR_CHECK(vxQueryArray((vx_array)model_array, VX_ARRAY_ATTRIBUTE_ITEMSIZE, &size, sizeof(size)));
	STATUS_ERROR_CHECK(vxQueryArray((vx_array)model_array, VX_ARRAY_ATTRIBUTE_ITEMTYPE, &type, sizeof(type)));
	if (type != VX_TYPE_SIZE)  return VX_ERROR_INVALID_TYPE;

	void **model_ptr = NULL;
	vx_size stride = 0ul;
	STATUS_ERROR_CHECK(vxAccessArrayRange((vx_array)model_array, 0, 1, &stride, (void**)&model_ptr, VX_READ_ONLY));
	STATUS_ERROR_CHECK(vxCommitArrayRange((vx_array)model_array, 0, 1, model_ptr));

	learning_model *model_struct = new learning_model;
	model_struct = static_cast<learning_model *>(*model_ptr);

	// load input tensor into WinML TensorFloat
	vx_tensor inputTensor = (vx_tensor)parameters[3];
	TensorFloat inputTensorElement = VX_to_ML_tensor(inputTensor);

	// get model input tensor name
	char modelInputName[1024];
	vx_scalar inputNameScalar = (vx_scalar)parameters[1];
	STATUS_ERROR_CHECK(vxReadScalarValue(inputNameScalar, modelInputName));
	std::string inputName(modelInputName);
	wstring wInputtName(inputName.begin(), inputName.end());
	hstring ModelInputTensorName = wInputtName.c_str();

	// bind the intput image
	model_struct->binding.Bind(ModelInputTensorName, inputTensorElement);
	// run inference
	auto results = model_struct->session.Evaluate(model_struct->binding, L"RunId");

	// load ouput tensor from WinML TensorFloat
	vx_tensor outputTensor = (vx_tensor)parameters[5];
	// get model output tensor name
	char modelOutputName[1024];
	vx_scalar outputNameScalar = (vx_scalar)parameters[2];
	STATUS_ERROR_CHECK(vxReadScalarValue(outputNameScalar, modelOutputName));
	std::string outputName(modelOutputName);
	wstring wOutputName(outputName.begin(), outputName.end());
	hstring ModelOutputTensorName = wOutputName.c_str();
	auto resultTensor = results.Outputs().Lookup(ModelOutputTensorName).as<TensorFloat>();
	auto resultVector = resultTensor.GetAsVectorView();

	// copy results to output
	STATUS_ERROR_CHECK(ML_to_VX_tensor(resultVector, outputTensor));

	// release scalar
	STATUS_ERROR_CHECK(vxReleaseScalar(&outputNameScalar));
	// release tensors
	STATUS_ERROR_CHECK(vxReleaseTensor(&inputTensor));
	STATUS_ERROR_CHECK(vxReleaseTensor(&outputTensor));

	//release array
	STATUS_ERROR_CHECK(vxReleaseArray(&model_array));
        return status;
}

/************************************************************************************************************
Function to Register the Kernel for Publish
*************************************************************************************************************/
vx_status  WINML_OnnxToMivisionX_Register(vx_context context)
{
        vx_status status = VX_SUCCESS;
        vx_kernel kernel = vxAddKernel(context,
                "com.winml.onnx_to_mivisionx",
                VX_KERNEL_WINML_ONNX_TO_MIVISIONX,
                WINML_OnnxToMivisionX_Kernel,
                7,
                WINML_OnnxToMivisionX_InputValidator,
                WINML_OnnxToMivisionX_OutputValidator,
                WINML_OnnxToMivisionX_Initialize,
                WINML_OnnxToMivisionX_Uninitialize);

        if (kernel)
        {
                PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
                PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
                PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
                PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
		PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
                PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 5, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
		PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 6, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL));
		PARAM_ERROR_CHECK(vxFinalizeKernel(kernel));
		PARAM_ERROR_CHECK(vxReleaseKernel(&kernel));
        }

        if (status != VX_SUCCESS)
        {
        	exit:   vxRemoveKernel(kernel);  std::cerr << "ERROR: vxAddParameterToKernel() failed for onnx_to_mivisionx" << std::endl; return VX_FAILURE;
        }

        return status;
}
