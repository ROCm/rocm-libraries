/******************************************************************************
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights
 * reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *******************************************************************************/

#ifndef HIPLIBXT_H_
#define HIPLIBXT_H_

#define MAX_HIP_DESCRIPTOR_GPUS 64

/*! @struct hipXtDesc_t
 * @brief Struct for single-process multi-GPU transform
 *
 * This struct holds pointers to device memory, including the device
 * the memory resides on and the size of each block of memory.
 *
 * @warning Experimental
 */
typedef struct hipXtDesc_t
{
    int version;
    // Count of GPUs
    int nGPUs;
    // Device IDs
    int GPUs[MAX_HIP_DESCRIPTOR_GPUS];
    // Data pointers for each GPU
    void* data[MAX_HIP_DESCRIPTOR_GPUS];
    // Size of data pointed to, for each GPU
    size_t size[MAX_HIP_DESCRIPTOR_GPUS];
    // Internal state
    void* hipXtState;
} hipXtDesc;

typedef enum hiplibFormat_t
{
    HIPLIB_FORMAT_HIPFFT    = 0x0,
    HIPLIB_FORMAT_UNDEFINED = 0x1
} hiplibFormat;

/*! @struct hipLibXtDesc_t
 * @brief Struct for single-process multi-GPU transform
 *
 * This struct holds \ref hipXtDesc_t structures that define blocks
 * of memory for use in a transform.
 *
 * @warning Experimental
 */
typedef struct hipLibXtDesc_t
{
    int version;
    // Descriptor of memory layout
    hipXtDesc* descriptor;
    // Which library is using this format
    hiplibFormat library;
    // Additional format information specific to the library
    int subFormat;
    // Other information specific to the library
    void* libDescriptor;
} hipLibXtDesc;

typedef enum hipfftXtCopyType_t
{
    HIPFFT_COPY_HOST_TO_DEVICE   = 0x00,
    HIPFFT_COPY_DEVICE_TO_HOST   = 0x01,
    HIPFFT_COPY_DEVICE_TO_DEVICE = 0x02,
    HIPFFT_COPY_UNDEFINED        = 0x03
} hipfftXtCopyType;

#endif
