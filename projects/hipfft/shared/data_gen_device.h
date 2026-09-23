// Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef DATA_GEN_DEVICE_H
#define DATA_GEN_DEVICE_H

#include "../shared/rocfft_complex.h"
#include <hip/hip_runtime_api.h>
#include <vector>
static const unsigned int DATA_GEN_THREADS = 8;

#ifdef USE_HIPRAND
template <typename Tint, typename Treal>
void generate_random_interleaved_data(const Tint&            whole_length,
                                      const size_t           idist,
                                      const size_t           isize,
                                      const Tint&            whole_stride,
                                      rocfft_complex<Treal>* input_data,
                                      const hipDeviceProp_t& deviceProp,
                                      const Tint&            field_lower,
                                      const size_t           field_lower_batch,
                                      const Tint&            field_contig_stride,
                                      const size_t           field_contig_dist);

template <typename Tint, typename Treal>
void generate_random_planar_data(const Tint&            whole_length,
                                 const size_t           idist,
                                 const size_t           isize,
                                 const Tint&            whole_stride,
                                 Treal*                 real_data,
                                 Treal*                 imag_data,
                                 const hipDeviceProp_t& deviceProp,
                                 const Tint&            field_lower,
                                 const size_t           field_lower_batch,
                                 const Tint&            field_contig_stride,
                                 const size_t           field_contig_dist);

template <typename Tint, typename Treal>
void generate_random_real_data(const Tint&            whole_length,
                               const size_t           idist,
                               const size_t           isize,
                               const Tint&            whole_stride,
                               Treal*                 input_data,
                               const hipDeviceProp_t& deviceProp,
                               const Tint             field_lower,
                               const size_t           field_lower_batch,
                               const Tint             field_contig_stride,
                               const size_t           field_contig_dist);
#endif // USE_HIPRAND

template <typename Tint, typename Treal>
void generate_interleaved_data(const Tint&            whole_length,
                               const size_t           idist,
                               const size_t           isize,
                               const Tint&            whole_stride,
                               const size_t           nbatch,
                               rocfft_complex<Treal>* input_data,
                               const hipDeviceProp_t& deviceProp);

template <typename Tint, typename Treal>
void generate_planar_data(const Tint&            whole_length,
                          const size_t           idist,
                          const size_t           isize,
                          const Tint&            whole_stride,
                          const size_t           nbatch,
                          Treal*                 real_data,
                          Treal*                 imag_data,
                          const hipDeviceProp_t& deviceProp);

template <typename Tint, typename Treal>
void generate_real_data(const Tint&            whole_length,
                        const size_t           idist,
                        const size_t           isize,
                        const Tint&            whole_stride,
                        const size_t           nbatch,
                        Treal*                 input_data,
                        const hipDeviceProp_t& deviceProp);

template <typename Tcomplex>
void impose_hermitian_symmetry_interleaved(const std::vector<size_t>& length,
                                           const std::vector<size_t>& ilength,
                                           const std::vector<size_t>& stride,
                                           const size_t               dist,
                                           const size_t               batch,
                                           Tcomplex*                  input_data,
                                           const hipDeviceProp_t&     deviceProp);

template <typename Tfloat>
void impose_hermitian_symmetry_planar(const std::vector<size_t>& length,
                                      const std::vector<size_t>& ilength,
                                      const std::vector<size_t>& stride,
                                      const size_t               dist,
                                      const size_t               batch,
                                      Tfloat*                    input_data_real,
                                      Tfloat*                    input_data_imag,
                                      const hipDeviceProp_t&     deviceProp);
#endif // DATA_GEN_DEVICE_H
