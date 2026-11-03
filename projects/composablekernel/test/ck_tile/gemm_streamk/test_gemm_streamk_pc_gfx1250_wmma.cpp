// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// GemmPipelineAgBgCrCompTDMProducerConsumer inside StreamKKernel with atomic reduction and the
// CShuffle epilogue: every workgroup adds its partial tile into C. Linear and tree reduction are
// in test_gemm_streamk_pc_reduction_gfx1250_wmma.cpp.

#include <cstdint>
#include <string>

#include "test_gemm_streamk_pc_gfx1250_fixture.hpp"

using PCStreamKAtomicTypes =
    ::testing::Types<PCStreamKParams<F16, Atomic, CShuffleEpilogue, NonPersistent>,
                     PCStreamKParams<F16, Atomic, CShuffleEpilogue, Persistent>,
                     PCStreamKParams<BF16, Atomic, CShuffleEpilogue, NonPersistent>,
                     PCStreamKParams<BF16, Atomic, CShuffleEpilogue, Persistent>>;

template <typename Params>
class TestGfx1250StreamKPCAtomic : public TestGfx1250StreamKPC<Params>
{
};

TYPED_TEST_SUITE(TestGfx1250StreamKPCAtomic, PCStreamKAtomicTypes);

TYPED_TEST(TestGfx1250StreamKPCAtomic, LoopCountsPerWorkgroup) { this->LoopCountsPerWorkgroup(); }

TYPED_TEST(TestGfx1250StreamKPCAtomic, SeveralSegmentsPerWorkgroup)
{
    this->SeveralSegmentsPerWorkgroup();
}

TYPED_TEST(TestGfx1250StreamKPCAtomic, RaggedMAndK) { this->RaggedMAndK(); }

TYPED_TEST(TestGfx1250StreamKPCAtomic, MixedDataParallelAndStreamK)
{
    this->MixedDataParallelAndStreamK();
}

TYPED_TEST(TestGfx1250StreamKPCAtomic, DeepKExact) { this->DeepKExact(); }

TYPED_TEST(TestGfx1250StreamKPCAtomic, WholeTileUnitsOverwriteC)
{
    this->WholeTileUnitsOverwriteC();
}

TYPED_TEST(TestGfx1250StreamKPCAtomic, DataParallelWhenTilesMultipleOfNumCu)
{
    this->DataParallelWhenTilesMultipleOfNumCu();
}

TYPED_TEST(TestGfx1250StreamKPCAtomic, RaggedNRejected)
{
    // TDM pipelines cannot pad, and with row-major C a partial N tile would make the CShuffle
    // store wrap into the next row.
    const auto M  = TestFixture::M_Tile;
    const auto N  = TestFixture::N_Tile;
    const auto K  = 8 * TestFixture::K_Tile;
    const auto* p = reinterpret_cast<void*>(std::uintptr_t{0x1000});
    auto* c       = reinterpret_cast<void*>(std::uintptr_t{0x1000});
    EXPECT_FALSE(TestFixture::IsArgSupported(TestFixture::MakeArgs(M, N + 1, K, p, p, c),
                                             TestFixture::kNumWorkgroups));
    EXPECT_TRUE(TestFixture::IsArgSupported(TestFixture::MakeArgs(M + 1, N, K, p, p, c),
                                            TestFixture::kNumWorkgroups));
}

TYPED_TEST(TestGfx1250StreamKPCAtomic, MisalignedPointersRejected)
{
    // TDM addresses global memory in elements from a byte base address.
    const auto M         = TestFixture::M_Tile;
    const auto N         = TestFixture::N_Tile;
    const auto K         = 8 * TestFixture::K_Tile;
    const auto* aligned  = reinterpret_cast<void*>(std::uintptr_t{0x1000});
    const auto* odd      = reinterpret_cast<void*>(std::uintptr_t{0x1001});
    auto* c_aligned      = reinterpret_cast<void*>(std::uintptr_t{0x1000});
    auto* c_odd          = reinterpret_cast<void*>(std::uintptr_t{0x1001});
    const auto supported = [&](const void* a, const void* b, void* c) {
        return TestFixture::IsArgSupported(TestFixture::MakeArgs(M, N, K, a, b, c),
                                           TestFixture::kNumWorkgroups);
    };
    EXPECT_TRUE(supported(aligned, aligned, c_aligned));
    EXPECT_FALSE(supported(odd, aligned, c_aligned));
    EXPECT_FALSE(supported(aligned, odd, c_aligned));
    EXPECT_FALSE(supported(aligned, aligned, c_odd));
}

TYPED_TEST(TestGfx1250StreamKPCAtomic, ResidencyCheckDoesNotApply)
{
    // Atomic workgroups never wait on each other, so a budget past the device's is still valid.
    const auto M  = TestFixture::M_Tile;
    const auto N  = TestFixture::N_Tile;
    const int wgs = TestFixture::DeviceBudget() + 1;
    const auto K  = 2 * wgs * TestFixture::K_Tile;
    ASSERT_EQ(TestFixture::Partition(M, N, K, wgs).get_sk_ctas(), wgs);
    const auto* p = reinterpret_cast<void*>(std::uintptr_t{0x1000});
    auto* c       = reinterpret_cast<void*>(std::uintptr_t{0x1000});
    EXPECT_TRUE(TestFixture::IsArgSupported(TestFixture::MakeArgs(M, N, K, p, p, c), wgs));
}

// StreamKKernel rejects atomic reduction with an epilogue that cannot accumulate through this
// trait; TdmEpilogue's TDM store overwrites C, so only the last partial tile would survive. This
// checks the trait, not the kernel's static_assert itself.
TYPED_TEST(TestGfx1250StreamKPCAtomic, TdmEpilogueCannotAccumulate)
{
    using EpilogueProblem = typename TestFixture::Base::
        template KernelFor<false, false, false, false, false>::EpilogueProblem;
    EXPECT_FALSE(ck_tile::epilogue_can_accumulate_v<ck_tile::TdmEpilogue<EpilogueProblem>>);
    EXPECT_TRUE(ck_tile::epilogue_can_accumulate_v<ck_tile::CShuffleEpilogue<EpilogueProblem>>);
}

// Host-only: instantiates the StreamKKernel members that no launch path above reaches, so the
// strict-warning build checks them.
TYPED_TEST(TestGfx1250StreamKPCAtomic, HostMembersInstantiate)
{
    using Kernel           = typename TestFixture::Kernel;
    const std::string name = Kernel::GetName();
    EXPECT_NE(name.find("atomic"), std::string::npos) << name;
    EXPECT_GT(Kernel::MaxOccupancyGridSize(ck_tile::stream_config{}).x, 0u);
}
