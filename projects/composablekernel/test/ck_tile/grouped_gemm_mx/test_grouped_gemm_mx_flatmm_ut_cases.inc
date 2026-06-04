// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

constexpr int init_method = 1;

TYPED_TEST(TestGroupedGemmMXFlatmm, Basic_4Groups)
{
    const int group_count = 4;

    std::vector<int> Ms;
    std::vector<int> Ns;
    std::vector<int> Ks;

    for(int i = 0; i < group_count; i++)
    {
        Ms.push_back(256 + 128 * i);
        Ns.push_back(256 + 256 * i);
        Ks.push_back(256 + 256 * i);
    }

    this->Run(Ms, Ns, Ks, group_count, init_method);
}

TYPED_TEST(TestGroupedGemmMXFlatmm, Single_Group)
{
    this->Run({512}, {512}, {512}, 1, init_method);
}

TYPED_TEST(TestGroupedGemmMXFlatmm, Two_Groups_Same_Size)
{
    this->Run({256, 256}, {256, 256}, {256, 256}, 2, init_method);
}

TYPED_TEST(TestGroupedGemmMXFlatmm, Eight_Groups)
{
    const int group_count = 8;

    std::vector<int> Ms;
    std::vector<int> Ns;
    std::vector<int> Ks;

    for(int i = 0; i < group_count; i++)
    {
        Ms.push_back(128 + 128 * (i % 4));
        Ns.push_back(256 + 256 * (i % 3));
        Ks.push_back(256 + 256 * (i % 5));
    }

    this->Run(Ms, Ns, Ks, group_count, init_method);
}
