// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Include the architecture-specific WMMA implementations and traits
#include "ck_tile/core/arch/mma/wmma/wmma_gfx11.hpp"
#include "ck_tile/core/arch/mma/wmma/wmma_gfx12.hpp"
#include "ck_tile/core/arch/mma/wmma/wmma_selector.hpp"
#include "ck_tile/core/arch/mma/wmma/wmma_traits.hpp"
#include "ck_tile/core/arch/mma/wmma/wmma_transforms.hpp"
