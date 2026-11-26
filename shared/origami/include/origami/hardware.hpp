// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Public header of the hardware layer.  Each piece can also be included on its own:
//   architecture.hpp         architectures, their names and families
//   variant.hpp              chip variants and how raw device identifiers select one
//   traits.hpp               structural facts of each architecture
//   constants.hpp            measured constants of each (architecture, variant)
//   matrix_instructions.hpp  matrix-instruction latencies of each family
//   hardware.hpp             hardware_t, one device
//   family.hpp               compile-time family tags and dispatch

#include "origami/hardware/architecture.hpp"
#include "origami/hardware/constants.hpp"
#include "origami/hardware/family.hpp"
#include "origami/hardware/hardware.hpp"
#include "origami/hardware/matrix_instructions.hpp"
#include "origami/hardware/traits.hpp"
#include "origami/hardware/variant.hpp"
