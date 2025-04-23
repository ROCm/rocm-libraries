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

#ifndef _CUSTOM_API_H_
#define _CUSTOM_API_H_
#include <hip/hip_runtime.h>

#ifdef __cplusplus
extern  "C" {
#endif

/*! @enum customStatus_t
 * Error codes that are returned by all custom API calls.
 */
typedef enum
{
    customStatusSuccess              = 0, /*!< No errors */
    customStatusNotInitialized       = 1, /*!< Data not initialized. */
    customStatusInvalidValue         = 2, /*!< Incorrect variable value. */
    customStatusBadParm              = 3, /*!< Incorrect parameter detected. */
    customStatusAllocFailed          = 4, /*!< Memory allocation error. */
    customStatusInternalError        = 5, /*!< custom failure. */
    customStatusNotImplemented       = 6, /*!< Use of unimplemented feature. */
    customStatusUnknownError         = 7, /*!< Unknown error occurred. */
    customStatusUnsupportedOp        = 8, /*!< Unsupported operator for fusion. */
    customStatusGpuOperationsSkipped = 9, /*!< This is not an error. */
} customStatus_t;

enum CustomFunctionType
{
    Copy,
};

enum customBackend
{
    CPU,
    GPU
};

typedef enum
{
    FP32              = 0, /*!< float */
    FP16              = 1, /*!< float16. */
    UCHAR             = 2, /*!< char */
    UCHAR3            = 3, /*!< char3 */
    INT32             = 4, /*!< int32. */

}customDataType;

typedef void* customStream;
typedef void * customHandle;

typedef struct customTensorDesc_t
{
    customDataType data_type;
    unsigned         dims[4];
    unsigned         strides[4];
}customTensorDesc;

#if 0
struct customContext
{
    explicit customContext(CustomFunctionType function): _custom_fn_type(function)
    {
        customimpl = CustomImpl(function);
    };
    ~customContext() {};
    std::shared_ptr<custom_base> customimpl;
    CustomFunctionType custom_function_type() { return _custom_fn_type; }

private:
    CustomFunctionType _custom_fn_type;
};
#endif

customHandle CustomCreate(CustomFunctionType function);
customStatus_t CustomSetup(customHandle input_handle, customTensorDesc &inputdesc, customTensorDesc &outputdesc, customBackend backend, customStream stream, int num_cpu_threads=0);
customStatus_t CustomExecute(customHandle custom_handle, void *input_handle, customTensorDesc &inputdesc, void *output_handle, customTensorDesc &outputdesc);
customStatus_t CustomShutdown(customHandle custom_handle);


#ifdef __cplusplus
}
#endif

#endif