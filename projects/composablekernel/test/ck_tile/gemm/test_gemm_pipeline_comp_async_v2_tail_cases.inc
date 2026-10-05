// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Tail coverage for the CompAsyncV2 pipeline (two prefetch stages).
// num_loop 1 (tail One, no hot loop), 2 (tail Two, no hot loop), 3, 5 and 7 (tail One + hot
// loop), 4 and 32 (tail Two + hot loop). A tail that skips a K tile, an async K-tile read past
// the end of A/B or a missing async LDS fence shows up as wrong results or a memory fault.

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

// Padded shapes: M is not a tile multiple and K ends half way into the last K tile, so
// num_loop = ceil(K / K_Tile) takes the same values as in the unpadded test. N is kept a
// tile multiple: a padded N combined with a padded K can fault in the async pipelines
// (CompAsync and CompAsyncV2 alike, independent of the tail handling tested here), which
// is tracked separately.
TYPED_TEST(TEST_SUITE_NAME, PaddedTailCoverage)
{
    constexpr int KT = TestFixture::K_Tile;
    const std::vector<int> num_loops{1, 2, 3, 4, 5, 7, 32};
    const std::vector<std::pair<int, int>> MNs{{300, 512}, {776, 256}};
    for(const auto& [M, N] : MNs)
    {
        for(int nl : num_loops)
        {
            this->template Run<true, true, true>(M, N, nl * KT - KT / 2);
        }
    }
}
