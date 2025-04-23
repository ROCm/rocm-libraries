/*
MIT License

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

/* MiVisionX Inference deployment API (MiViDa) */
#ifndef mvdeploy_api_h
#define mvdeploy_api_h

#ifdef __cplusplus
extern "C" {
#endif

#ifndef MIVID_API_ENTRY
#define MIVID_API_ENTRY
#endif
#ifndef MIVID_API_CALL
#if defined(_WIN32)
#define MIVID_API_CALL __stdcall
#define MIVID_CALLBACK __stdcall
#else
#define MIVID_API_CALL
#define MIVID_CALLBACK
#endif
#endif

//! \brief: error return codes (has one to one correspondance with VX error codes)
enum mv_status
{
    /* add new codes here */
    MV_STATUS_MIN                       = -25,/*!< \brief Indicates the lower bound of status codes in VX. Used for bounds checks only. */
    /* add new codes here */
    MV_ERROR_REFERENCE_NONZERO          = -24,/*!< \brief Indicates that an operation did not complete due to a reference count being non-zero. */
    MV_ERROR_MULTIPLE_WRITERS           = -23,/*!< \brief Indicates that the graph has more than one node outputting to the same data object. This is an invalid graph structure. */
    MV_ERROR_GRAPH_ABANDONED            = -22,/*!< \brief Indicates that the graph is stopped due to an error or a callback that abandoned execution. */
    MV_ERROR_GRAPH_SCHEDULED            = -21,/*!< \brief Indicates that the supplied graph already has been scheduled and may be currently executing. */
    MV_ERROR_INVALID_SCOPE              = -20,/*!< \brief Indicates that the supplied parameter is from another scope and cannot be used in the current scope. */
    MV_ERROR_INVALID_NODE               = -19,/*!< \brief Indicates that the supplied node could not be created.*/
    MV_ERROR_INVALID_GRAPH              = -18,/*!< \brief Indicates that the supplied graph has invalid connections (cycles). */
    MV_ERROR_INVALID_TYPE               = -17,/*!< \brief Indicates that the supplied type parameter is incorrect. */
    MV_ERROR_INVALID_VALUE              = -16,/*!< \brief Indicates that the supplied parameter has an incorrect value. */
    MV_ERROR_INVALID_DIMENSION          = -15,/*!< \brief Indicates that the supplied parameter is too big or too small in dimension. */
    MV_ERROR_INVALID_FORMAT             = -14,/*!< \brief Indicates that the supplied parameter is in an invalid format. */
    MV_ERROR_INVALID_LINK               = -13,/*!< \brief Indicates that the link is not possible as specified. The parameters are incompatible. */
    MV_ERROR_INVALID_REFERENCE          = -12,/*!< \brief Indicates that the reference provided is not valid. */
    MV_ERROR_INVALID_MODULE             = -11,/*!< \brief This is returned from <tt>\ref vxLoadKernels</tt> when the module does not contain the entry point. */
    MV_ERROR_INVALID_PARAMETERS         = -10,/*!< \brief Indicates that the supplied parameter information does not match the kernel contract. */    
    MV_ERROR_OPTIMIZED_AWAY             = -9,/*!< \brief Indicates that the object refered to has been optimized out of existence. */
    MV_ERROR_NO_MEMORY                  = -8,/*!< \brief Indicates that an internal or implicit allocation failed. Typically catastrophic. After detection, deconstruct the context. \see vxVerifyGraph. */
    MV_ERROR_NO_RESOURCES               = -7,/*!< \brief Indicates that an internal or implicit resource can not be acquired (not memory). This is typically catastrophic. After detection, deconstruct the context. \see vxVerifyGraph. */
    MV_ERROR_NOT_COMPATIBLE             = -6,/*!< \brief Indicates that the attempt to link two parameters together failed due to type incompatibilty. */
    MV_ERROR_NOT_ALLOCATED              = -5,/*!< \brief Indicates to the system that the parameter must be allocated by the system.  */
    MV_ERROR_NOT_SUFFICIENT             = -4,/*!< \brief Indicates that the given graph has failed verification due to an insufficient number of required parameters, which cannot be automatically created. Typically this indicates required atomic parameters. \see vxVerifyGraph. */
    MV_ERROR_NOT_SUPPORTED              = -3,/*!< \brief Indicates that the requested set of parameters produce a configuration that cannot be supported. Refer to the supplied documentation on the configured kernels. \see vx_kernel_e. This is also returned if a function to set an attribute is called on a Read-only attribute.*/
    MV_ERROR_NOT_IMPLEMENTED            = -2,/*!< \brief Indicates that the requested kernel is missing. \see vx_kernel_e vxGetKernelByName. */
    MV_FAILURE                          = -1,/*!< \brief Indicates a generic error code, used when no other describes the error. */
    MV_SUCCESS                          =  0,/*!< \brief No error. */	
};

//! \brief: backend for inference runtime
enum mivid_backend
{
	OpenVX_Rocm_GPU 	= 0,
	OpenVX_WinML 		= 1,
	OpenVX_CPU   		= 2,
};


//! \brief: datatype for inference engine (tensor data type)
enum mivid_datatype
{
	datatype_float = 0,
	datatype_float16 = 1,
	datatype_int8	= 2
};

//! \brief: quantization_mode for modelcompiler
enum mivid_quantization_mode
{
	quant_fp32 = 0,
	quant_fp16 = 1,
	quant_int8 = 2,
	quant_int4	= 3
};

enum mivid_memory_type
{
	mv_mem_type_host = 0,
	mv_mem_type_opencl = 1,
	mv_mem_type_hcc	= 2				// not supported in version <= 1
};

//! \brief: needed for update of original model without retraining.
// some might loose accuracy like quantization to fp16 or int8
// the operation will only be supported if backend allows inference using updated model
typedef struct mivid_update_model_params_t
{
	int batch_size;
	bool fused_convolution_bias_activation;
	bool quantize_model;
	int quantization_mode;
	int run_precision_calibration;
}mivid_update_model_params;

#if ENABLE_MVDEPLOY

typedef void *mv_deploy;
typedef int mivid_status;
typedef void * mivid_session;

//! \brief Parameters required to be passed from preprocess callback function
typedef struct mv_preprocess_callback_args_t
{
    const char *inp_string_decoder;         // input for amd_video_decoder node
    int loop_decode;                        // indicate to loop at eof
    float preproc_a, preproc_b;       // proprocess multiply and add factor for image to tensor node
}mv_preprocess_callback_args;
//! \brief The log callback function
typedef void(*mivid_log_callback_f)(const char * message);
typedef vx_status(*mivid_add_preprocess_callback_f)(mivid_session inf_session, vx_tensor outp_tensor, mv_preprocess_callback_args *preproc_args);
typedef vx_status(*mivid_add_postprocess_callback_f)(mivid_session inf_session, vx_tensor inp_tensor);


//! \brief Query the version of the MivisionX inference engine.
MIVID_API_ENTRY const char * MIVID_API_CALL mvGetVersion();


//! \brief Creates deployment instance (loads the deployment library for the specific compiled backend and intializes all function pointers)
MIVID_API_ENTRY mv_status MIVID_API_CALL mvInitializeDeployment(const char* install_folder);

//! \brief: returns input outut configuration strings (name and dimensions)
MIVID_API_ENTRY mv_status MIVID_API_CALL QueryInference(int *num_inputs, int *num_outputs, const char **inp_out_config);

//! \brief Set callback for log messages.
//  - by default, log messages from library will be printed to stdout
//  - the log messages can be redirected to application using a callback
MIVID_API_ENTRY void MIVID_API_CALL SetLogCallback(mivid_log_callback_f callback);

//! \brief: load and add preprocessing module/nodes to graph if needed.
// need to call this before calling CreateInferenceSession
// output of the preprocessing node should be same as input tensor NN module
MIVID_API_ENTRY void MIVID_API_CALL SetPreProcessCallback(mivid_add_preprocess_callback_f preproc_f, mv_preprocess_callback_args *preproc_args);

//! \brief: load and add postprocessing modules/nodes to graph if needed.
// need to call this before calling CreateInferenceSession
// input to the preprocessing node should be same as output tensor of NN module
MIVID_API_ENTRY void MIVID_API_CALL SetPostProcessCallback(mivid_add_postprocess_callback_f postproc_f);

//! \brief: Creates an active inference deployment session with the set model and parameters
MIVID_API_ENTRY mv_status MIVID_API_CALL mvCreateInferenceSession(mivid_session *inf_session, const char *install_folder, mivid_memory_type in_type);

//! \brief: Releases inference session and all the resources associated
MIVID_API_ENTRY mv_status MIVID_API_CALL mvReleaseInferenceSession(mivid_session inf_session);

//! \brief: the input data memory pointer of the input tensor:  
MIVID_API_ENTRY mv_status MIVID_API_CALL mvSetInputDataFromMemory(mivid_session inf_session, int input_num, void *input_data, size_t size, mivid_memory_type type);
//! \brief: input data file name with full path: (if not file has to be present in the same folder as deployment)
MIVID_API_ENTRY mv_status MIVID_API_CALL mvSetInputDataFromFile(mivid_session inf_session, int input_num, char *input_name, bool reverseOrder, float preprocess_mulfac, float preprocess_addfac);

//! \brief: run an instance of the inference engine: can be run multiple iterations for performance timing
MIVID_API_ENTRY mv_status MIVID_API_CALL mvRunInference(mivid_session inf_session, float *p_time_in_millisec, int num_iterations=1);

//! \brief: run an instance of the inference engine: can be run multiple iterations for performance timing
MIVID_API_ENTRY mv_status MIVID_API_CALL mvGetOutputData(mivid_session inf_session, int out_num, void *out_buf, size_t size);

//! \brief: run inference engine asynchronous: every ScheduleInference() call has to be followed by WaitForCompletion() call
MIVID_API_ENTRY mv_status MIVID_API_CALL mvScheduleInferenceSession(mivid_session inf_session);
MIVID_API_ENTRY mv_status MIVID_API_CALL mvWaitForSessionCompletion(mivid_session inf_session);

//! \brief: shutdown deploy: shutdown deployment for model and backend
MIVID_API_ENTRY void MIVID_API_CALL mvShutdown();

#endif 

#ifdef __cplusplus
}
#endif

#endif
