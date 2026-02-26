// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Example 02: All Convolution Directions
 *
 * Demonstrates forward, backward-data, and backward-weight convolution
 * declarations in both 2D and 3D, all in one example.
 *
 * Build: cd dispatcher/build && cmake .. && make grouped_conv_02_all_dirs
 */

#include <iostream>
#include <iomanip>
#include <vector>

#include "ck_tile/dispatcher/grouped_conv_utils.hpp"
#include "ck_tile/dispatcher/example_args.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::grouped_conv_utils;
using GroupedConvSig  = grouped_conv_decl::GroupedConvSignature;
using GroupedConvAlgo = grouped_conv_decl::GroupedConvAlgorithm;

// =============================================================================
// 2D FORWARD
// =============================================================================
DECL_GROUPED_CONV_KERNEL_SET(
    conv2d_fwd,
    .add(GroupedConvSig().dtype("fp16").layout("nhwc").conv_type("forward").dims(2),
         GroupedConvAlgo().tile(1, 128, 128).pipeline("compv4").vector_sizes(4, 8, 8),
         "gfx950"));

// =============================================================================
// 3D FORWARD
// =============================================================================
DECL_GROUPED_CONV_KERNEL_SET(
    conv3d_fwd,
    .add(GroupedConvSig().dtype("fp16").layout("ndhwc").conv_type("forward").dims(3),
         GroupedConvAlgo().tile(1, 64, 64).pipeline("compv3").vector_sizes(4, 8, 8),
         "gfx950"));

// =============================================================================
// 2D BACKWARD DATA
// =============================================================================
DECL_GROUPED_CONV_KERNEL_SET(
    conv2d_bwdd,
    .add(GroupedConvSig().dtype("fp16").layout("nhwc").conv_type("bwd_data").dims(2),
         GroupedConvAlgo().tile(1, 128, 128).pipeline("compv3").vector_sizes(4, 8, 8),
         "gfx950"));

// =============================================================================
// 2D BACKWARD WEIGHT
// =============================================================================
DECL_GROUPED_CONV_KERNEL_SET(
    conv2d_bwdw,
    .add(GroupedConvSig().dtype("fp16").layout("nhwc").conv_type("bwd_weight").dims(2),
         GroupedConvAlgo()
             .tile(1, 128, 128)
             .pipeline("compv3")
             .memory_op("atomic_add")
             .vector_sizes(4, 8, 8),
         "gfx950"));

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char* argv[])
{
    utils::ExampleArgs args("Example 02: All Convolution Directions",
                            "Forward/BwdData/BwdWeight in 2D and 3D");

    if(!args.parse(argc, argv))
        return 0;

    utils::print_header("Example 02: All Convolution Directions");

    // =========================================================================
    // Show all registered kernel sets
    // =========================================================================
    std::cout << "\nRegistered Kernel Sets:\n";
    GroupedConvKernelSetRegistry::instance().print();

    auto& reg = GroupedConvKernelSetRegistry::instance();

    // =========================================================================
    // 2D Forward
    // =========================================================================
    std::cout << "\n--- 2D Forward ---\n";
    {
        auto problem = create_grouped_conv2d_problem(1, 64, 128, 28, 28, 3, 3, 1, 1);
        print_grouped_conv_problem(problem);

        GroupedConvRegistry registry;
        registry.set_name("fwd_2d");
        registry.register_set(reg.get("conv2d_fwd"));
        std::cout << "  Registered " << registry.size() << " kernel(s)\n";

        GroupedConvDispatcher dispatcher(&registry);
        const auto* sel = dispatcher.select(problem);
        std::cout << "  Selected: " << (sel ? sel->name() : "none") << "\n";
    }

    // =========================================================================
    // 3D Forward
    // =========================================================================
    std::cout << "\n--- 3D Forward ---\n";
    {
        auto problem = create_grouped_conv3d_problem(1, 32, 64, 8, 16, 16, 3, 3, 3, 1, 1);
        print_grouped_conv_problem(problem);

        GroupedConvRegistry registry;
        registry.set_name("fwd_3d");
        registry.register_set(reg.get("conv3d_fwd"));
        std::cout << "  Registered " << registry.size() << " kernel(s)\n";
    }

    // =========================================================================
    // 2D Backward Data
    // =========================================================================
    std::cout << "\n--- 2D Backward Data ---\n";
    {
        auto problem = create_grouped_conv2d_problem(
            1, 128, 64, 28, 28, 3, 3, 1, 1, GroupedConvOp::BackwardData);
        print_grouped_conv_problem(problem);

        GroupedConvRegistry registry;
        registry.set_name("bwdd_2d");
        registry.register_set(reg.get("conv2d_bwdd"));
        std::cout << "  Registered " << registry.size() << " kernel(s)\n";

        GroupedConvDispatcher dispatcher(&registry);
        const auto* sel = dispatcher.select(problem);
        std::cout << "  Selected: " << (sel ? sel->name() : "none") << "\n";
    }

    // =========================================================================
    // 2D Backward Weight
    // =========================================================================
    std::cout << "\n--- 2D Backward Weight ---\n";
    {
        auto problem = create_grouped_conv2d_problem(
            1, 64, 128, 28, 28, 3, 3, 1, 1, GroupedConvOp::BackwardWeight);
        print_grouped_conv_problem(problem);

        GroupedConvRegistry registry;
        registry.set_name("bwdw_2d");
        registry.register_set(reg.get("conv2d_bwdw"));
        std::cout << "  Registered " << registry.size() << " kernel(s)\n";

        GroupedConvDispatcher dispatcher(&registry);
        const auto* sel = dispatcher.select(problem);
        std::cout << "  Selected: " << (sel ? sel->name() : "none") << "\n";
    }

    // =========================================================================
    // Summary
    // =========================================================================
    utils::print_separator();
    std::cout << "ALL DIRECTIONS DEMONSTRATED:\n";
    std::cout << "  conv2d_fwd:  forward 2D    (Y = Conv(X, W))\n";
    std::cout << "  conv3d_fwd:  forward 3D    (Y = Conv3D(X, W))\n";
    std::cout << "  conv2d_bwdd: backward data (dX = ConvBwdData(dY, W))\n";
    std::cout << "  conv2d_bwdw: backward wt   (dW = ConvBwdWeight(X, dY))\n";
    utils::print_separator();

    std::cout << "\n  Status: PASS\n";
    return 0;
}
