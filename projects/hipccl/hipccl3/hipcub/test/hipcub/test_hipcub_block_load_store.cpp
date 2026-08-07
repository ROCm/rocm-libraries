/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2017-2020, Advanced Micro Devices, Inc.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include "common_test_header.hpp"

// required test headers
#include "test_utils_types.hpp"

// kernel definitions
#include "test_hipcub_block_load_store.kernels.hpp"
#include <hipcub/iterator/discard_output_iterator.hpp>

// Start stamping out tests
struct HipcubBlockLoadStoreTests;

struct Direct;
#define suite_name HipcubBlockLoadStoreTests
#define load_store_params LoadStoreParamsDirect
#define name_suffix Direct

#include "test_hipcub_block_load_store.hpp"

#undef suite_name
#undef load_store_params
#undef name_suffix

struct Vectorize;
#define suite_name HipcubBlockLoadStoreTests
#define load_store_params LoadStoreParamsVectorize
#define name_suffix Vectorize

#include "test_hipcub_block_load_store.hpp"

#undef suite_name
#undef load_store_params
#undef name_suffix

struct Transpose;
#define suite_name HipcubBlockLoadStoreTests
#define load_store_params LoadStoreParamsTranspose
#define name_suffix Transpose

#include "test_hipcub_block_load_store.hpp"

#undef suite_name
#undef load_store_params
#undef name_suffix

struct Striped;
#define suite_name HipcubBlockLoadStoreTests
#define load_store_params LoadStoreParamsStriped
#define name_suffix Striped

#include "test_hipcub_block_load_store.hpp"

#undef suite_name
#undef load_store_params
#undef name_suffix
