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

#pragma once
#include "custom_api.h"
#include "custom_template.h"

class customCopy: public custom_base {
public:
    customCopy() {};
    ~customCopy() {};
    /*!
     \param input_handle  => memory handle of input tensor
     \param inputdesc => Input tensor desc
     \param output_handle  => memory handle of output tensor
     \param outputdesc => output tensor desc
     \param backend  => backend for the impl
     \param stream  => Output command queue
    */
    customStatus_t Setup(customTensorDesc &inputdesc, customTensorDesc &outputdesc, customBackend backend, customStream stream, int num_cpu_threads=1);
    /*!
     \param input_handle  => memory handle of input tensor
     \param inputdesc => Input tensor desc
     \param output_handle  => memory handle of output tensor
    */
    customStatus_t Execute(void *input_handle, customTensorDesc &inputdesc, void *output_handle, customTensorDesc &outputdesc);
     
    //* Shutdown and release resources */
    customStatus_t Shutdown();
// workspace data    
private:
    customTensorDesc _input_desc, _output_desc;
    customBackend _backend;
    customStream  _stream;
    int _cpu_num_threads;
};