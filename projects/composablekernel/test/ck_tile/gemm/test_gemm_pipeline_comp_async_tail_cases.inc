// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Unpadded shapes: K is a tile multiple chosen to hit each tail path of the async pipeline.
// num_loop 1 (tail One), 2 (tail Two, no hot loop), 3 (tail Three, no hot loop),
// 4 and 32 (tail Two + hot loop), 5 and 7 (tail Three + hot loop; 7 iterates the hot loop
// more than once). Any async K-tile over-read past the end of A/B or a missing async LDS
// fence in a tail shows up as wrong results or an illegal memory access.
// Keep k_batch = 1 so the loop counts are exact on both XDL and WMMA.
TYPED_TEST(TEST_SUITE_NAME, UnpaddedTailCoverage)
{
    constexpr int KT = TestFixture::K_Tile;
    const std::vector<int> num_loops{1, 2, 3, 4, 5, 7, 32};
    const std::vector<std::pair<int, int>> MNs{{512, 512}, {512, 1024}, {1024, 512}, {1024, 1024}};
    for(const auto& [M, N] : MNs)
    {
        for(int nl : num_loops)
        {
            SCOPED_TRACE(::testing::Message() << "M=" << M << ", N=" << N << ", num_loop=" << nl);
            this->template RunSingle<false, false, false, false>(M, N, nl * KT, 0, 0, 0, 1);
        }
    }
}
