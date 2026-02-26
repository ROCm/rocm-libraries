// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Example 01: Basic Grouped Convolution
 *
 * Demonstrates THREE declaration patterns (mirrors GEMM 01):
 *
 * 1. AUTOFILL: Minimal declaration - missing params filled with defaults
 * 2. AUTOCORRECT: Invalid params corrected to valid values
 * 3. FULL: All parameters explicitly specified
 *
 * Shows the declarative workflow: declare -> register -> dispatch -> JSON.
 * For actual GPU execution + validation, see 03_benchmark_validation.cpp.
 *
 * Build: cd dispatcher/build && cmake .. && make grouped_conv_01_basic
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

#include "ck_tile/dispatcher/grouped_conv_utils.hpp"
#include "ck_tile/dispatcher/example_args.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::grouped_conv_utils;
using GroupedConvSig  = grouped_conv_decl::GroupedConvSignature;
using GroupedConvAlgo = grouped_conv_decl::GroupedConvAlgorithm;

// =============================================================================
// THREE DECLARATION PATTERNS
// =============================================================================

DECL_GROUPED_CONV_KERNEL_SET(
    basic_conv_kernels,

    // Pattern 1: AUTOFILL - only required params, defaults filled
    .add(GroupedConvSig().dtype("fp16").layout("nhwc").conv_type("forward").dims(2),
         GroupedConvAlgo()
             .tile(1, 128, 128)
             .pipeline("compv4")
             .scheduler("intrawave"),
         "gfx950")

    // Pattern 2: AUTOCORRECT - invalid wave(1,1,1) fixed to (2,2,1)
    .add(GroupedConvSig().dtype("fp16").layout("nhwc").conv_type("forward").dims(2),
         GroupedConvAlgo()
             .tile(1, 64, 64)
             .wave(1, 1, 1)
             .warp(16, 16, 32)
             .pipeline("compv3")
             .scheduler("intrawave")
             .epilogue("cshuffle")
             .vector_sizes(4, 8, 8),
         "gfx950")

    // Pattern 3: FULL - all params explicit
    .add(GroupedConvSig()
             .dtype("fp16", "fp16", "fp16", "fp32")
             .layout("nhwc")
             .conv_type("forward")
             .dims(2),
         GroupedConvAlgo()
             .tile(1, 128, 128)
             .wave(2, 2, 1)
             .warp(32, 32, 16)
             .pipeline("compv3")
             .scheduler("intrawave")
             .epilogue("cshuffle")
             .vector_sizes(4, 8, 8)
             .block_per_cu(1),
         "gfx950"));

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char* argv[])
{
    utils::ExampleArgs args("Example 01: Basic Grouped Convolution",
                            "Autofill, autocorrect, and full declaration patterns");
    args.add_option("-n", "1", "Batch size");
    args.add_option("-c", "64", "Input channels C");
    args.add_option("-k", "128", "Output channels K");
    args.add_option("--size", "28", "Input spatial size (HxW)");

    if(!args.parse(argc, argv))
        return 0;

    const int N  = args.get_int("-n", 1);
    const int C  = args.get_int("-c", 64);
    const int K  = args.get_int("-k", 128);
    const int HW = args.get_int("--size", 28);

    utils::print_header("Example 01: Basic Grouped Convolution");

    // =========================================================================
    // Step 1: Show declared kernels
    // =========================================================================
    std::cout << "\nStep 1: Declared Kernel Sets\n";
    std::cout << "  THREE PATTERNS:\n";
    std::cout << "    1. AUTOFILL:     tile + pipeline only -> wave/warp auto-filled\n";
    std::cout << "    2. AUTOCORRECT:  wave(1,1,1) invalid -> corrected to (2,2,1)\n";
    std::cout << "    3. FULL:         all params explicit\n\n";

    GroupedConvKernelSetRegistry::instance().print();

    const auto& decl_set = GroupedConvKernelSetRegistry::instance().get("basic_conv_kernels");
    std::cout << "  'basic_conv_kernels': " << decl_set.size() << " declaration(s)\n";

    for(const auto& decl : decl_set.declarations())
    {
        print_grouped_conv_kernel_decl(decl);
    }

    // =========================================================================
    // Step 2: Build problem
    // =========================================================================
    std::cout << "\nStep 2: Build Problem\n";

    auto problem = GroupedConvProblemBuilder()
                       .batch(N)
                       .channels(C, K)
                       .groups(1)
                       .input_size(HW, HW)
                       .filter_size(3, 3)
                       .stride(1, 1)
                       .padding(1, 1)
                       .operation(GroupedConvOp::Forward)
                       .build();

    std::cout << "  " << problem.to_string() << "\n";
    std::cout << "  FLOPs: " << std::scientific << problem.get_flops() << "\n\n";

    // =========================================================================
    // Step 3: Register into registry and create dispatcher
    // =========================================================================
    std::cout << "Step 3: Register & Dispatch\n";

    GroupedConvRegistry registry;
    registry.set_name("basic_conv");
    registry.register_set(decl_set);
    std::cout << "  Registered " << registry.size() << " kernel(s)\n";

    GroupedConvDispatcher dispatcher(&registry);
    const auto* selected = dispatcher.select(problem);
    if(selected)
    {
        std::cout << "  Selected: " << selected->name() << "\n";
    }
    else
    {
        std::cout << "  No kernel matched (expected - placeholder run functions)\n";
    }

    // =========================================================================
    // Step 4: Export to JSON
    // =========================================================================
    std::cout << "\nStep 4: JSON Export\n";
    std::string json = registry.export_json(true);
    // Print first 400 chars
    std::cout << json.substr(0, std::min(json.size(), size_t(400))) << "\n  ...\n";

    // =========================================================================
    // Summary
    // =========================================================================
    utils::print_separator();
    std::cout << "GROUPED CONVOLUTION DECLARATION PATTERNS:\n";
    utils::print_separator();
    std::cout << R"(
  DECL_GROUPED_CONV_KERNEL_SET(name,
      .add(GroupedConvSig().dtype("fp16").layout("nhwc").conv_type("forward").dims(2),
           GroupedConvAlgo().tile(1, 128, 128).pipeline("compv4"),
           "gfx950")
  );

  1. AUTOFILL:     Specify tile + pipeline, system fills wave/warp/epilogue
  2. AUTOCORRECT:  Invalid wave/warp corrected to valid combos
  3. FULL:         All parameters explicit for production tuning
)";
    utils::print_separator();

    std::cout << "\n  Status: PASS (declarations registered and exported)\n";
    return 0;
}
