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
#include <stdio.h>
#include <stdlib.h>
#include "custom_copy_impl.h"
#if ENABLE_HIP
#include "hip/hip_runtime_api.h"
#include "hip/hip_runtime.h"
#endif

customStatus_t customCopy::Setup(customTensorDesc &inputdesc, customTensorDesc &outputdesc, customBackend backend, customStream stream, int num_cpu_threads) {
    _input_desc = inputdesc, _output_desc = outputdesc;
    _backend = backend, _stream = stream;
    // find the number of cpu threads available in the system
    if (num_cpu_threads == 0) {
        unsigned sys_cpu_thread_count;
        sys_cpu_thread_count = std::thread::hardware_concurrency();
        if (sys_cpu_thread_count < 2) sys_cpu_thread_count = 2;
       _cpu_num_threads = sys_cpu_thread_count>>1; // don't count hyperthreads
    } else
        _cpu_num_threads = num_cpu_threads;

    if ((_input_desc.data_type != _output_desc.data_type) || (_input_desc.dims[0] != _output_desc.dims[0]) || 
        (_input_desc.dims[1] != _output_desc.dims[1]) || (_input_desc.dims[2] != _output_desc.dims[2]) || (_input_desc.dims[3] != _output_desc.dims[3]))
        return  customStatus_t::customStatusInvalidValue;
    return customStatusSuccess;
}

customStatus_t customCopy::Execute(void *input_handle, customTensorDesc &inputdesc, void *output_handle, customTensorDesc &outputdesc) {
    unsigned size = outputdesc.dims[0] * outputdesc.dims[1] * outputdesc.dims[3] * sizeof(_output_desc.data_type);
    unsigned batch_size = outputdesc.dims[3];
    if (_backend == customBackend::CPU) {
        int omp_threads =  (_cpu_num_threads < batch_size)?  _cpu_num_threads: batch_size;
        #pragma omp parallel for num_threads(omp_threads)
        for (size_t i = 0; i < batch_size; i++) {
            unsigned char *src, *dst;
            src = (unsigned char *)input_handle + size*i;
            dst = (unsigned char *)output_handle + size*i;
            memcpy(dst, src, size);
        }
    } else {
#if ENABLE_HIP
        for (size_t i = 0; i < batch_size; i++) {
            unsigned char *src, *dst;
            src = (unsigned char *)input_handle + size*i;
            dst = (unsigned char *)output_handle + size*i;
            hipMemcpy(dst, src, size, hipMemcpyDeviceToDevice);
        }
#endif
    }
    return customStatusSuccess;
}

customStatus_t customCopy::Shutdown()
{
    // nothing to do since we don't have any local resources to release.
    return customStatusSuccess;
}