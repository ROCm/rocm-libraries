// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>

#include <origami/hardware.hpp>
#include <origami/streamk.hpp>
#include <origami/types.hpp>

namespace TensileLite
{
    /**
     * Plain-value inputs to the rocRoller Stream-K grid solve.
     *
     * This is a port of chooseStreamKGridSize() in
     * library/src/amd_detail/rocblaslt/src/rocroller/runtime_args_selection.cpp,
     * which is the grid the rocRoller JIT path launches. The rocRoller-off path
     * must pick the same grid, so the two must agree field for field -- the
     * fields left unset below are as load-bearing as the ones that are set,
     * because chooseStreamKGridSize leaves them at origami's defaults too.
     *
     * Deliberately takes only scalars: no ContractionSolution, no Problem, no
     * GPU, so RocRollerStreamKGrid_test.cpp can hold it against a verbatim
     * transcription of chooseStreamKGridSize.
     */
    struct RocRollerStreamKGridInputs
    {
        size_t m     = 0;
        size_t n     = 0;
        size_t k     = 0;
        size_t batch = 1;

        /// rocRoller's sm_count_target. 0 means "use all CUs".
        size_t numCUs = 0;

        /// Kernel macrotile (M, N, K). rocRoller passes its workgroupTile here,
        /// NOT depthU: the K entry is the kernel's tile K.
        size_t tileM = 0;
        size_t tileN = 0;
        size_t tileK = 0;

        /// Workgroups resident per CU. origami rejects <= 0 (config_t::is_valid),
        /// so callers that do not know it must not reach this function.
        int occupancy = 1;

        size_t workspaceSize = 0;

        /// Bytes of one accumulate-type element (4 for Float).
        size_t workspaceSizePerElemC = 0;

        origami::data_type_t aDtype = origami::data_type_t::None;
        origami::data_type_t bDtype = origami::data_type_t::None;

        /// Element widths of A and B in bits; only their order matters, and only
        /// to pick mi_dtype (see rocRollerStreamKGridProblem).
        size_t aElementBits = 0;
        size_t bElementBits = 0;
    };

    /// The grid-selection mode chooseStreamKGridSize uses for both origami calls.
    constexpr origami::grid_selection_t RocRollerStreamKGridSelection
        = origami::grid_selection_t::k_split_aware;

    /**
     * The origami::problem_t chooseStreamKGridSize builds.
     *
     * Sets exactly size, batch, num_cus, a_dtype, b_dtype and mi_dtype. In
     * particular a_transpose, b_transpose, c_dtype and d_dtype stay at origami's
     * defaults -- rocRoller does not set them, so neither may this.
     *
     * mi_dtype is whichever of A/B has the larger element width, matching
     * rocRoller's `elementBitsA < elementBitsB ? B : A`.
     */
    inline origami::problem_t
        rocRollerStreamKGridProblem(RocRollerStreamKGridInputs const& in)
    {
        return origami::problem_t{
            .size    = {in.m, in.n, in.k},
            .batch   = in.batch,
            .num_cus = in.numCUs,
            .a_dtype = in.aDtype,
            .b_dtype = in.bDtype,
            .mi_dtype = in.aElementBits < in.bElementBits ? in.bDtype : in.aDtype,
        };
    }

    /**
     * The origami::config_t chooseStreamKGridSize builds, before
     * reduction_strategy is filled in by select_reduction.
     *
     * Sets exactly mt, occupancy, workspace_size and workspace_size_per_elem_c.
     * mi and workgroup_mapping stay at origami's defaults because rocRoller does
     * not set them. (The Tensile-side adapter getPersistentGridImpl() does set
     * both and uses depthU for mt.k; copying it here would silently pick a
     * different grid.)
     */
    inline origami::config_t rocRollerStreamKGridConfig(RocRollerStreamKGridInputs const& in)
    {
        return origami::config_t{
            .mt                        = {in.tileM, in.tileN, in.tileK},
            .occupancy                 = in.occupancy,
            .workspace_size            = in.workspaceSize,
            .workspace_size_per_elem_c = in.workspaceSizePerElemC,
        };
    }

    /**
     * The Stream-K grid the rocRoller JIT path would launch for this problem on
     * this hardware: select_reduction, then select_grid_size with the selected
     * strategy, both in k_split_aware mode.
     *
     * Returns origami's answer unmodified. Range clamps belong to the caller.
     */
    inline uint32_t rocRollerStreamKGrid(RocRollerStreamKGridInputs const& in,
                                         origami::hardware_t const&        hw)
    {
        origami::problem_t problem = rocRollerStreamKGridProblem(in);
        origami::config_t  config  = rocRollerStreamKGridConfig(in);

        config.reduction_strategy = origami::streamk::select_reduction(
            problem, hw, config, RocRollerStreamKGridSelection);

        // problem.num_cus carries the caller's budget (sm_count_target);
        // select_grid_size derives the CU budget from it internally.
        return static_cast<uint32_t>(origami::streamk::select_grid_size(
            problem, hw, config, RocRollerStreamKGridSelection));
    }
} // namespace TensileLite
