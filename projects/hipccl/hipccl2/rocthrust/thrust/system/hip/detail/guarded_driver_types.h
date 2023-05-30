/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2019 Advanced Micro Devices, Inc. All rights reserved.
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

#include <thrust/detail/config.h>

// the purpose of this header is to #include <driver_types.h> without causing
// warnings from redefinitions of __host__ and __device__.
// carefully save their definitions and restore them


#ifdef __host__
#  pragma push_macro("__host__")
#  undef __host__
#  define THRUST_HOST_NEEDS_RESTORATION
#endif
#ifdef __device__
#  pragma push_macro("__device__")
#  undef __device__
#  define THRUST_DEVICE_NEEDS_RESTORATION
#endif

#include <hip/amd_detail/host_defines.h>

#ifdef THRUST_HOST_NEEDS_RESTORATION
#  pragma pop_macro("__host__")
#  undef THRUST_HOST_NEEDS_RESTORATION
#endif
#ifdef THRUST_DEVICE_NEEDS_RESTORATION
#  pragma pop_macro("__device__")
#  undef THRUST_DEVICE_NEEDS_RESTORATION
#endif
