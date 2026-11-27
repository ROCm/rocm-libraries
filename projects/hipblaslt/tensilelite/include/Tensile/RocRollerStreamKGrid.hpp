// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

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

    /// Product of free/bound/batch extents. An empty list is 1, matching a
    /// loop that starts at 1 and multiplies each index.
    inline size_t rocRollerStreamKExtentProduct(std::vector<size_t> const& extents)
    {
        size_t product = 1;
        for(size_t extent : extents)
            product *= extent;
        return product;
    }

    /// CU count capped by a positive smCountTarget, a positive persistentMaxCUs
    /// budget, and the flag-region length. At least one workgroup.
    inline uint32_t rocRollerStreamKClampedCuCount(int      computeUnitCount,
                                                   int      smCountTarget,
                                                   int      persistentMaxCUs,
                                                   uint32_t flagElements)
    {
        int grid = computeUnitCount;
        if(smCountTarget > 0)
            grid = std::min(grid, smCountTarget);
        if(persistentMaxCUs > 0)
            grid = std::min(grid, persistentMaxCUs);
        grid = std::min(grid, static_cast<int>(flagElements));
        return static_cast<uint32_t>(std::max(1, grid));
    }

    /**
     * Scalar inputs to assembleRocRollerStreamKGrid.
     *
     * freeA/freeB/bound/batch are the problem extents those index groups
     * contribute; the assembly multiplies each group. numCUs on the origami
     * path is smCountTarget alone (0 means all CUs). persistentMaxCUs applies
     * only to the CU-count fallback.
     */
    struct RocRollerStreamKGridRequest
    {
        std::vector<size_t> freeA;
        std::vector<size_t> freeB;
        std::vector<size_t> bound;
        std::vector<size_t> batch;

        int smCountTarget     = 0;
        int computeUnitCount  = 0;
        int persistentMaxCUs  = 0;

        size_t tileM = 0;
        size_t tileN = 0;
        size_t tileK = 0;

        int occupancy = 0;

        size_t workspaceSize         = 0;
        size_t workspaceSizePerElemC = 0;

        origami::data_type_t aDtype = origami::data_type_t::None;
        origami::data_type_t bDtype = origami::data_type_t::None;
        size_t               aElementBits = 0;
        size_t               bElementBits = 0;

        /// False when the device is not a HipAMDGPU or has no analytical hardware.
        bool hasAnalyticalHardware = false;
    };

    /// useOrigami false means the launch grid is fallbackGrid (the clamped CU
    /// count). inputs is filled either way so a test can read the products.
    struct RocRollerStreamKGridPlan
    {
        bool                       useOrigami   = false;
        uint32_t                   fallbackGrid = 1;
        RocRollerStreamKGridInputs inputs{};
    };

    /**
     * The Stream-K grid inputs ContractionSolution::rocRollerStreamKGrid builds,
     * plus the cases that never call origami.
     *
     * origami is not called when occupancy is 0 (the logic file did not record
     * it; config_t::is_valid rejects <= 0), when analytical hardware is missing,
     * or when M, N or batch is 0 (origami divides by the tile count). Those
     * launches use the clamped CU count. Otherwise inputs is what
     * rocRollerStreamKGrid consumes. That answer is not clamped:
     * chooseStreamKGridSize returns it unchanged.
     */
    inline RocRollerStreamKGridPlan assembleRocRollerStreamKGrid(RocRollerStreamKGridRequest const& req,
                                                                uint32_t flagElements)
    {
        RocRollerStreamKGridPlan plan;
        plan.fallbackGrid = rocRollerStreamKClampedCuCount(
            req.computeUnitCount, req.smCountTarget, req.persistentMaxCUs, flagElements);

        plan.inputs.m     = rocRollerStreamKExtentProduct(req.freeA);
        plan.inputs.n     = rocRollerStreamKExtentProduct(req.freeB);
        plan.inputs.k     = rocRollerStreamKExtentProduct(req.bound);
        plan.inputs.batch = rocRollerStreamKExtentProduct(req.batch);
        // Only sm_count_target. persistentMaxCUs is a fallback cap, not an
        // origami input.
        plan.inputs.numCUs = static_cast<size_t>(std::max(0, req.smCountTarget));
        plan.inputs.tileM  = req.tileM;
        plan.inputs.tileN  = req.tileN;
        plan.inputs.tileK  = req.tileK;
        plan.inputs.occupancy             = req.occupancy;
        plan.inputs.workspaceSize         = req.workspaceSize;
        plan.inputs.workspaceSizePerElemC = req.workspaceSizePerElemC;
        plan.inputs.aDtype                = req.aDtype;
        plan.inputs.bDtype                = req.bDtype;
        plan.inputs.aElementBits          = req.aElementBits;
        plan.inputs.bElementBits          = req.bElementBits;

        plan.useOrigami = req.occupancy != 0 && req.hasAnalyticalHardware && plan.inputs.m != 0
                          && plan.inputs.n != 0 && plan.inputs.batch != 0;
        return plan;
    }
} // namespace TensileLite
