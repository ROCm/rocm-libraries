// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "_blocked_backend_test_support.hpp"

namespace blocked_backend_test {
void testOverlappingOutputIsRejectedAcrossBackends() {
    using namespace roc::host_numerics;

    constexpr size_t rows = 33;
    constexpr size_t reductionElements = 512;
    constexpr size_t columns = 64;

    std::vector<float> a(rows * reductionElements);
    for (size_t row = 0; row < rows; ++row)
        std::fill_n(a.begin() + row * reductionElements, reductionElements,
                    static_cast<float>(row + 1));
    const std::vector<float> b(reductionElements * columns, 1.0f);
    Tensor output(ScalarType::Float32, Layout(Shape{rows, columns}, {0, 0}));
    GemmTestCase problem = makeProblem(a, b, output, rows, reductionElements, columns);

    require(!queryGemmSupport(problem, GemmBackend::Blocked),
            "Blocked GEMM accepted overlapping destination elements.");
    require(!queryGemmSupport(problem, GemmBackend::Automatic),
            "Automatic GEMM accepted overlapping destination elements.");
}

void testOutputCannotAliasInputs() {
    using namespace roc::host_numerics;

    constexpr size_t extent = 2;
    const std::vector<float> a{1, 2, 3, 4};
    const std::vector<float> b{5, 6, 7, 8};

    GemmTestCase overlapsA =
        makeProblem(a, b, makeOutput(extent, extent, 0), extent, extent, extent);
    overlapsA.d = overlapsA.a;
    require(!queryGemmSupport(overlapsA, GemmBackend::Blocked),
            "GEMM accepted destination storage that overlaps A.");
}

}  // namespace blocked_backend_test
