// SPDX-FileCopyrightText: Copyright (c) 2008-2009, NVIDIA Corporation. All rights reserved.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if !THRUST_CUDA_COMPILATION() || !THRUST_HAS_HIP_COMPILER()

#  ifndef __host__
#    define __host__
#  endif

#  ifndef __device__
#    define __device__
#  endif

#endif
