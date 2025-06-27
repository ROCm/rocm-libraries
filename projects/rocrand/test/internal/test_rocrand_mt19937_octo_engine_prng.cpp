#include <gtest/gtest.h>
#include <stdio.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_set>
#include <vector>

#include <rng/mt19937_octo_engine.hpp>

/* #################################################

                TEST HOST SIDE

   ###############################################*/

TEST(Mt19937OctoEngineTest, test_host_gather)
{
    // Check if a mt199370 Octo Engine state contatins the correct arrangement listed below

    /// Thread 0 has element   0, thread 1 has element 113, thread 2 has element 170,
    /// thread 3 had element 283, thread 4 has element 340, thread 5 has element 397,
    /// thread 6 has element 510, thread 7 has element 567.
    /// Thread i for i in [0, 7) has the following elements (ipt = items_per_thread):
    /// [  1 + ipt * i,   1 + ipt * (i + 1)), [398 + ipt * i, 398 + ipt * (i + 1)), [171 + ipt * i, 171 + ipt * (i + 1)),
    /// [568 + ipt * i, 568 + ipt * (i + 1)), [341 + ipt * i, 341 + ipt * (i + 1)), [114 + ipt * i, 114 + ipt * (i + 1)),
    /// [511 + ipt * i, 511 + ipt * (i + 1)), [284 + ipt * i, 284 + ipt * (i + 1)), [ 57 + ipt * i,  57 + ipt * (i + 1)),
    /// [454 + ipt * i, 454 + ipt * (i + 1)), [227 + ipt * i, 227 + ipt * (i + 1))
    ///

    namespace constants = rocrand_impl::host::mt19937_constants;

    std::vector<unsigned int> src(constants::n);
    std::iota(src.begin(), src.end(), 0);

    const std::vector<unsigned int> offsets = {1, 398, 171, 568, 341, 114, 511, 284, 57, 454, 227};
    const std::vector<unsigned int> special_elem = {0, 113, 170, 283, 340, 397, 510, 567};
    const unsigned int              ipt          = 7;
    const unsigned int              vpt          = 1 + ipt * 11;

    rocrand_impl::host::mt19937_octo_engine test_engine;

    for(size_t tid = 0; tid < 8; tid++)
    {
        dim3 t_idx = dim3(tid, 0, 0);

        test_engine.gather(src.data(), t_idx);

        std::vector<unsigned int> expected_items;

        for(const unsigned int& offset : offsets)
        {
            auto left  = offset + ipt * tid;
            auto right = (offset + ipt * (tid + 1)); // no need to -1 since insert is exclusive

            expected_items.insert(expected_items.begin(), src.begin() + left, src.begin() + right);
        }

        expected_items.insert(expected_items.begin(), special_elem[tid]);
        
        std::sort(expected_items.begin(), expected_items.end());
        
        std::vector<unsigned int> actual_items(vpt);

        for(size_t i = 0; i < vpt; i++)
            actual_items[i] = test_engine.get(i);

        std::sort(actual_items.begin(), actual_items.end());

        for(size_t i = 0; i < vpt; i++)
            ASSERT_EQ(expected_items[i], actual_items[i]);
    }
}