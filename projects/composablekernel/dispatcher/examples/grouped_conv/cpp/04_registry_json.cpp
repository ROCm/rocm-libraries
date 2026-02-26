// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Example 04: Multi-Registry and JSON Export
 *
 * Demonstrates:
 * - Multiple registries for different workloads (throughput vs latency)
 * - GroupedConvDispatcher for kernel selection
 * - JSON export with statistics
 * - filter_by_arch for architecture-specific deployment
 *
 * Build: cd dispatcher/build && cmake .. && make grouped_conv_04_registry_json
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

// Throughput-optimized kernels (large tiles)
DECL_GROUPED_CONV_KERNEL_SET(
    throughput_kernels,
    .add(GroupedConvSig().dtype("fp16").layout("nhwc").conv_type("forward").dims(2),
         GroupedConvAlgo().tile(1, 256, 256).pipeline("compv4").vector_sizes(4, 8, 8),
         "gfx950")
    .add(GroupedConvSig().dtype("fp16").layout("nhwc").conv_type("forward").dims(2),
         GroupedConvAlgo().tile(1, 128, 256).pipeline("compv4").vector_sizes(4, 8, 8),
         "gfx950"));

// Latency-optimized kernels (small tiles)
DECL_GROUPED_CONV_KERNEL_SET(
    latency_kernels,
    .add(GroupedConvSig().dtype("fp16").layout("nhwc").conv_type("forward").dims(2),
         GroupedConvAlgo().tile(1, 64, 64).pipeline("compv3").vector_sizes(4, 8, 8),
         "gfx950")
    .add(GroupedConvSig().dtype("fp16").layout("nhwc").conv_type("forward").dims(2),
         GroupedConvAlgo().tile(1, 32, 32).pipeline("compv3").vector_sizes(4, 4, 4),
         "gfx950"));

// Multi-arch kernels
DECL_GROUPED_CONV_KERNEL_SET(
    multi_arch_kernels,
    .add(GroupedConvSig().dtype("fp16").layout("nhwc").conv_type("forward").dims(2),
         GroupedConvAlgo().tile(1, 128, 128).pipeline("compv4"),
         "gfx950")
    .add(GroupedConvSig().dtype("fp16").layout("nhwc").conv_type("forward").dims(2),
         GroupedConvAlgo().tile(1, 128, 128).pipeline("compv3"),
         "gfx942")
    .add(GroupedConvSig().dtype("bf16").layout("nhwc").conv_type("forward").dims(2),
         GroupedConvAlgo().tile(1, 128, 128).pipeline("compv4"),
         "gfx950"));

int main(int argc, char* argv[])
{
    utils::ExampleArgs args("Example 04: Multi-Registry & JSON Export",
                            "Separate registries and JSON export with statistics");
    args.add_option("--output", "", "JSON output file (optional)");

    if(!args.parse(argc, argv))
        return 0;

    utils::print_header("Example 04: Multi-Registry & JSON Export");

    auto& kset_reg = GroupedConvKernelSetRegistry::instance();

    // =========================================================================
    // Throughput registry
    // =========================================================================
    std::cout << "\n--- Throughput Registry ---\n";
    GroupedConvRegistry throughput_reg;
    throughput_reg.set_name("throughput");
    throughput_reg.register_set(kset_reg.get("throughput_kernels"), GroupedConvRegistry::Priority::High);
    std::cout << "  Kernels: " << throughput_reg.size() << "\n";

    // =========================================================================
    // Latency registry
    // =========================================================================
    std::cout << "\n--- Latency Registry ---\n";
    GroupedConvRegistry latency_reg;
    latency_reg.set_name("latency");
    latency_reg.register_set(kset_reg.get("latency_kernels"), GroupedConvRegistry::Priority::High);
    std::cout << "  Kernels: " << latency_reg.size() << "\n";

    // =========================================================================
    // Dispatcher selection
    // =========================================================================
    std::cout << "\n--- Dispatcher Selection ---\n";

    auto large_problem = create_grouped_conv2d_problem(8, 128, 256, 56, 56, 3, 3, 1, 1);
    auto small_problem = create_grouped_conv2d_problem(1, 32, 64, 14, 14, 1, 1, 1, 0);

    GroupedConvDispatcher throughput_disp(&throughput_reg);
    GroupedConvDispatcher latency_disp(&latency_reg);

    auto* tp_sel = throughput_disp.select(large_problem);
    auto* lt_sel = latency_disp.select(small_problem);

    std::cout << "  Large problem -> throughput: " << (tp_sel ? tp_sel->name() : "none") << "\n";
    std::cout << "  Small problem -> latency:    " << (lt_sel ? lt_sel->name() : "none") << "\n";

    // =========================================================================
    // Multi-arch with filter_by_arch
    // =========================================================================
    std::cout << "\n--- Multi-Arch Filter ---\n";
    GroupedConvRegistry multi_arch_reg;
    multi_arch_reg.set_name("multi_arch");
    multi_arch_reg.register_set(kset_reg.get("multi_arch_kernels"));
    std::cout << "  Before filter: " << multi_arch_reg.size() << " kernels\n";

    auto removed = multi_arch_reg.filter_by_arch("gfx950");
    std::cout << "  Removed " << removed << " non-gfx950 kernels\n";
    std::cout << "  After filter:  " << multi_arch_reg.size() << " kernels\n";

    // =========================================================================
    // JSON export with statistics
    // =========================================================================
    std::cout << "\n--- JSON Export ---\n";

    // Merge all into one registry for comprehensive export
    GroupedConvRegistry combined;
    combined.set_name("all_conv_kernels");
    combined.register_set(kset_reg.get("throughput_kernels"));
    combined.register_set(kset_reg.get("latency_kernels"));
    combined.register_set(kset_reg.get("multi_arch_kernels"));

    std::string json = combined.export_json(true);
    std::cout << "  Total kernels in combined registry: " << combined.size() << "\n";
    std::cout << "  JSON size: " << json.size() << " bytes\n";

    // Print first portion
    std::cout << "\n  Preview:\n";
    auto preview = json.substr(0, std::min(json.size(), size_t(500)));
    std::cout << preview << "\n  ...\n";

    // Optionally write to file
    std::string output_file = args.get("--output", "");
    if(!output_file.empty())
    {
        combined.export_json_to_file(output_file, true);
        std::cout << "\n  Written to: " << output_file << "\n";
    }

    // =========================================================================
    // Summary
    // =========================================================================
    utils::print_separator();
    std::cout << "MULTI-REGISTRY & JSON FEATURES:\n";
    std::cout << "  - Separate registries: throughput vs latency\n";
    std::cout << "  - Priority-based kernel registration\n";
    std::cout << "  - GroupedConvDispatcher selects best kernel per problem\n";
    std::cout << "  - filter_by_arch() for deployment-time arch filtering\n";
    std::cout << "  - export_json(include_statistics=true) for analysis\n";
    utils::print_separator();

    std::cout << "\n  Status: PASS\n";
    return 0;
}
