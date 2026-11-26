// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <unordered_map>

#include "origami/hardware/architecture.hpp"
#include "origami/origami_export.h"
#include "origami/types.hpp"

namespace origami {

/**
 * @brief Latency in clocks of each matrix instruction of one instruction set.
 */
using matrix_instruction_latencies_t = std::unordered_map<matrix_instruction, std::size_t>;

/**
 * @brief Matrix-instruction latencies of an architecture family.
 *
 * The architectures of a family share one instruction set, so there is one table per family.
 * The tables live in the Origami library behind this single exported accessor, so every
 * shared library that includes this header reads the same definition.
 *
 * @param family Architecture family
 * @return const matrix_instruction_latencies_t& The family's latency table
 * @throws std::out_of_range if @p family is family_t::unknown
 */
ORIGAMI_EXPORT const matrix_instruction_latencies_t& matrix_instruction_latencies(family_t family);

}  // namespace origami
