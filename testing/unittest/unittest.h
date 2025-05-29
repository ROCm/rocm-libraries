/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

// this is the only header included by unittests
// it pulls in all the others used for unittesting

#include <unittest/assertions.h>
#include <unittest/meta.h>
#include <unittest/random.h>
#include <unittest/special_types.h>
#include <unittest/testframework.h>

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HIP
#  define THRUST_DEVICE_BACKEND                 hip
#  define THRUST_DEVICE_BACKEND_DETAIL          hip_rocprim
#  define SPECIALIZE_DEVICE_RESOURCE_NAME(name) hip##name
#elif THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC
#  define THRUST_DEVICE_BACKEND                 cuda
#  define THRUST_DEVICE_BACKEND_DETAIL          cuda_cub
#  define SPECIALIZE_DEVICE_RESOURCE_NAME(name) cuda##name
#endif
