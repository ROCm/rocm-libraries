// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Compatibility include for the former MMA-I/O helper surface.
 * New authoring uses helper_rocke.helpers.tiling.h for load/store, ir_storage.h
 * for storage typing, and helper_rocke.helpers.packing.h for encoded patterns.
 * Existing symbols remain available; no independent I/O policy lives here.
 */
#ifndef ROCKE_HELPER_MMA_IO_H
#define ROCKE_HELPER_MMA_IO_H

#include "rocke/helper_rocke.helpers.packing.h"
#include "rocke/ir_storage.h"
#include "rocke/tiling_compat.h"

#endif /* ROCKE_HELPER_MMA_IO_H */
