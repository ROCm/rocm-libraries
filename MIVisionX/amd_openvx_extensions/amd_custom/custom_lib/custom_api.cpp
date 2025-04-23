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

#include "custom_api.h"
#include "custom_template.h"
#include "custom_copy_impl.h"
#include <memory>

// add new function types for supporting new custom api
custom_base * CreateCustomClass(CustomFunctionType function) {

    switch(function)
    {
        case Copy:
            return new customCopy();
            break;
        // todo:: add new custom function types here with corresponding implemetation files
        default:
            throw std::runtime_error ("Custom function type is unsupported");
            return nullptr;
    }
}


customHandle CustomCreate(CustomFunctionType function) {
    customHandle handle = (customHandle)CreateCustomClass(function);
    return handle;
}

customStatus_t customRelease(customHandle custom_handle)
{
    // Deleting context is required to call the destructor of all the member objects
    //std::shared_ptr<custom_base> *custom_ptr = static_cast<std::shared_ptr<custom_base>*>(custom_handle);
    custom_base *custom_ptr = static_cast<custom_base*>(custom_handle);
    delete custom_ptr;
    return customStatusSuccess;
}


customStatus_t CustomSetup(customHandle custom_handle, customTensorDesc &inputdesc, customTensorDesc &outputdesc, customBackend backend, customStream stream, int num_cpu_threads)
{
    if (!custom_handle)
      return customStatusInvalidValue;
    customStatus_t status = customStatusInternalError;
    custom_base *custom_ptr = static_cast<custom_base*>(custom_handle);
    //std::shared_ptr<custom_base> *custom_ptr = static_cast<std::shared_ptr<custom_base>*>(custom_handle);
    // Create custom base class factory
    status = custom_ptr->Setup(inputdesc, outputdesc, backend, stream, num_cpu_threads);
    return status;
}

customStatus_t CustomExecute(customHandle custom_handle, void *input_handle, customTensorDesc &inputdesc, void *output_handle, customTensorDesc &outputdesc)
{
    if (!custom_handle)
      return customStatusInvalidValue;
    customStatus_t status = customStatusInternalError;
    //std::shared_ptr<custom_base> *custom_ptr = static_cast<std::shared_ptr<custom_base>*>(custom_handle);
    custom_base *custom_ptr = static_cast<custom_base*>(custom_handle);
    status = custom_ptr->Execute(input_handle, inputdesc, output_handle, outputdesc);
    return status;
}

customStatus_t CustomShutdown(customHandle custom_handle)
{
    if (!custom_handle)
      return customStatusInvalidValue;
    customStatus_t status = customStatusInternalError;
    //std::shared_ptr<custom_base> *custom_ptr = static_cast<std::shared_ptr<custom_base>*>(custom_handle);
    custom_base *custom_ptr = static_cast<custom_base*>(custom_handle);
    status = custom_ptr->Shutdown();
    return status;
}
