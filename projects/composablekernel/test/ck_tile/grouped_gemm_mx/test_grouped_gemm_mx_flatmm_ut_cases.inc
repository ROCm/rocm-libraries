// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// =================================================================================
// Test cases for GroupedMXFlatmmKernel: 4 num_loop classes x 2 cases each = 8 tests.
//
// The MX FLATMM non-TDM pipeline (MXFlatmmPipelineAGmemBGmemCRegV1) bakes
// HasHotLoop and TailNum into the compiled kernel. The grouped wrapper can only
// launch one kernel instantiation per call, so all groups in a single Run() must
// share the same num_loop class (the fixture asserts this). Each class therefore
// needs its own test cases with K values that satisfy that class.
//
//   PrefetchStages = 2
//   BlockHasHotloop(num_loop)  = (num_loop > 2)
//   GetBlockLoopTailNum(n_loop) = (n_loop % 2 == 0) ? Even : Odd
//   K_Tile = 256, so num_loop = ceil(K / K_Tile)
//
//   | Class | HasHotLoop | TailNum | num_loop      | Valid K (K_Tile=256)      |
//   |-------|------------|---------|---------------|---------------------------|
//   |  A    | false      | Odd     | {1}           | {256}                     |
//   |  B    | false      | Even    | {2}           | {512}                     |
//   |  C    | true       | Odd     | {3, 5, 7, ...}| {768, 1280, 1792, ...}    |
//   |  D    | true       | Even    | {4, 6, 8, ...}| {1024, 1536, 2048, ...}   |
//
// Dimension constraints:
//   M must be a multiple of M_Tile = 128 (all configs).
//   N must be a multiple of 512 to satisfy MXfp4_FlatmmConfig16's N_Tile=512;
//     this is a superset of MXFlatmmConfigBase16 / MXFlatmmConfigBase32TDM
//     (N_Tile=256), so the same test cases cover both non-TDM and TDM binaries.
//   K must satisfy the class table above.
// =================================================================================

constexpr int init_method = 1;

// --- Class A: HasHotLoop=false, TailNum=Odd (num_loop=1, K=256) ---

TYPED_TEST(TestGroupedGemmMXFlatmm, ClassA_NoHotLoop_OddTail_SingleGroup)
{
    this->Run({128}, {512}, {256}, 1, init_method);
}

TYPED_TEST(TestGroupedGemmMXFlatmm, ClassA_NoHotLoop_OddTail_FourGroups)
{
    // All K=256 (only valid K for Class A); vary M and N.
    this->Run({128, 256, 384, 512},
              {512, 1024, 512, 1024},
              {256, 256, 256, 256},
              4,
              init_method);
}

// --- Class B: HasHotLoop=false, TailNum=Even (num_loop=2, K=512) ---

TYPED_TEST(TestGroupedGemmMXFlatmm, ClassB_NoHotLoop_EvenTail_SingleGroup)
{
    this->Run({128}, {512}, {512}, 1, init_method);
}

TYPED_TEST(TestGroupedGemmMXFlatmm, ClassB_NoHotLoop_EvenTail_FourGroups)
{
    // All K=512 (only valid K for Class B); vary M and N.
    this->Run({128, 256, 384, 512},
              {512, 1024, 512, 1024},
              {512, 512, 512, 512},
              4,
              init_method);
}

// --- Class C: HasHotLoop=true, TailNum=Odd (num_loop in {3,5,7,...}) ---

TYPED_TEST(TestGroupedGemmMXFlatmm, ClassC_HotLoop_OddTail_SingleGroup)
{
    // K=768 -> num_loop=3 (smallest valid for Class C).
    this->Run({128}, {512}, {768}, 1, init_method);
}

TYPED_TEST(TestGroupedGemmMXFlatmm, ClassC_HotLoop_OddTail_FourGroups)
{
    // num_loop values 3, 5, 7, 5 -- all odd and >= 3.
    this->Run({128, 256, 384, 512},
              {512, 1024, 512, 1024},
              {768, 1280, 1792, 1280},
              4,
              init_method);
}

// --- Class D: HasHotLoop=true, TailNum=Even (num_loop in {4,6,8,...}) ---

TYPED_TEST(TestGroupedGemmMXFlatmm, ClassD_HotLoop_EvenTail_SingleGroup)
{
    // K=1024 -> num_loop=4 (smallest valid for Class D).
    this->Run({128}, {512}, {1024}, 1, init_method);
}

TYPED_TEST(TestGroupedGemmMXFlatmm, ClassD_HotLoop_EvenTail_FourGroups)
{
    // num_loop values 4, 6, 8, 6 -- all even and >= 4.
    this->Run({128, 256, 384, 512},
              {512, 1024, 512, 1024},
              {1024, 1536, 2048, 1536},
              4,
              init_method);
}
