//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023-24 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef LIBCXX_WRAPPER__CCCL_CONFIG_H
#define LIBCXX_WRAPPER__CCCL_CONFIG_H

// TODO(libhipcxx): remove this file once libhipcxx gets ready

#include <thrust/detail/config/libcxx.h>

#if _THRUST_HAS_DEVICE_SYSTEM_STD
#  include _THRUST_LIBCXX_INCLUDE(__cccl_config) // IWYU pragma: export
#endif

#include <thrust/detail/libcxx_wrapper/std/__cccl/attributes.h> // IWYU pragma: export
#include <thrust/detail/libcxx_wrapper/std/__cccl/builtin.h> // IWYU pragma: export
#include <thrust/detail/libcxx_wrapper/std/__cccl/compiler.h> // IWYU pragma: export
#include <thrust/detail/libcxx_wrapper/std/__cccl/cuda_toolkit.h> // IWYU pragma: export
#include <thrust/detail/libcxx_wrapper/std/__cccl/deprecated.h> // IWYU pragma: export
#include <thrust/detail/libcxx_wrapper/std/__cccl/diagnostic.h> // IWYU pragma: export
#include <thrust/detail/libcxx_wrapper/std/__cccl/dialect.h> // IWYU pragma: export
#include <thrust/detail/libcxx_wrapper/std/__cccl/exceptions.h> // IWYU pragma: export
#include <thrust/detail/libcxx_wrapper/std/__cccl/execution_space.h> // IWYU pragma: export
#include <thrust/detail/libcxx_wrapper/std/__cccl/preprocessor.h> // IWYU pragma: export
#include <thrust/detail/libcxx_wrapper/std/__cccl/rtti.h> // IWYU pragma: export
#include <thrust/detail/libcxx_wrapper/std/__cccl/sequence_access.h> // IWYU pragma: export
#include <thrust/detail/libcxx_wrapper/std/__cccl/unreachable.h> // IWYU pragma: export
#include <thrust/detail/libcxx_wrapper/std/__cccl/visibility.h> // IWYU pragma: export

#endif // LIBCXX_WRAPPER__CCCL_CONFIG_H
