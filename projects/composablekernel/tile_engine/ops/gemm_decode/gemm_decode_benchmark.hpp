// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// P0 placeholder. The full benchmark harness (matching the role of
// tile_engine/ops/gemm/gemm_universal/gemm_universal_benchmark.hpp) lands
// in P1+ together with the Python instance builder. P0 benchmark execs
// pull the helpers they need directly from `gemm_decode_common.hpp`.
#include "gemm_decode_common.hpp"
