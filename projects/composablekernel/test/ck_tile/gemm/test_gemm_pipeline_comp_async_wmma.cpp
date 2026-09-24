#include "test_gemm_pipeline_kernel_types.hpp"
#include "test_gemm_pipeline_wmma_base.hpp"
#include "gtest/gtest.h"

template <typename T>
class TestCkTileGemmPipelineCompAsyncWmma
    : public TestCkTileGemmPipelineWmmaBase<T, class TestCkTileGemmPipelineCompAsyncWmma<T>>
{
};

#define TEST_SUITE_NAME TestCkTileGemmPipelineCompAsyncWmma

TYPED_TEST_SUITE(TestCkTileGemmPipelineCompAsyncWmma, KernelTypesCompAsyncWmma);

#include "test_gemm_pipeline_ut_cases.inc"

// Unpadded shapes: K is a tile multiple chosen to hit each tail path of the async pipeline.
// num_loop 1 (tail One), 2 (tail Two, no hot loop), 3 (tail Three, no hot loop),
// 4 and 32 (tail Two + hot loop), 5 and 7 (tail Three + hot loop; 7 iterates the hot loop
// more than once). Any async K-tile over-read past the end of A/B or a missing async LDS
// fence in a tail shows up as wrong results or an illegal memory access.
// M = N = K = 1024 is covered by {1024, 1024} x num_loop 32.
TYPED_TEST(TEST_SUITE_NAME, UnpaddedTailCoverage)
{
    constexpr int KT = TestFixture::K_Tile;
    const std::vector<int> num_loops{1, 2, 3, 4, 5, 7, 32};
    const std::vector<std::pair<int, int>> MNs{{512, 512}, {512, 1024}, {1024, 512}, {1024, 1024}};
    for(const auto& [M, N] : MNs)
    {
        for(int nl : num_loops)
        {
            this->template Run<false, false, false>(M, N, nl * KT);
        }
    }
}

#undef TEST_SUITE_NAME
