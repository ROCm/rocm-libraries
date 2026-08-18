/*
MIT License

Copyright (c) 2026 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <gtest/gtest.h>
#include <rpp/rpp.h>

#include <string>
#include <vector>

#include "framework/backend_memory.hpp"
#include "framework/config_param.hpp"
#include "framework/dtype_dispatch.hpp"
#include "framework/generic_tensor_setup.hpp"
#include "framework/skip_list.hpp"
#include "framework/tensor_setup.hpp"
#include "reference/transpose_ref.hpp"

using namespace rpptest;

namespace {

// The permutation is rank-dependent, so the grid carries the *kind* of permutation and each
// case materializes the concrete perm array for its own rank.
enum class PermKind { Identity, Reverse, RotateLeft };

struct TransposeParams {
    PermKind kind;
    std::string name() const {
        switch (kind) {
            case PermKind::Identity:   return "Identity";
            case PermKind::Reverse:    return "Reverse";
            case PermKind::RotateLeft: return "RotateLeft";
        }
        return "UNK";
    }
};

std::vector<Rpp32u> make_perm(PermKind kind, Rpp32u nDim) {
    std::vector<Rpp32u> perm(nDim);
    for (Rpp32u k = 0; k < nDim; ++k) {
        switch (kind) {
            case PermKind::Identity:   perm[k] = k; break;
            case PermKind::Reverse:    perm[k] = nDim - 1 - k; break;
            case PermKind::RotateLeft: perm[k] = (k + 1) % nDim; break;
        }
    }
    return perm;
}

template <typename T>
void run_transpose(const NdConfig& cfg, const TransposeParams& p) {
    const NdDims srcDims = nd_extents(cfg.nDim);
    const std::vector<Rpp32u> perm = make_perm(p.kind, cfg.nDim);
    const NdDims dstDims = transpose_dst_dims(srcDims, perm);

    // Descriptors are device-addressable for HIP: the ND kernels read dims/strides on device.
    GenericDescriptor srcDesc(cfg.backend, srcDims, cfg.dtypeIn);
    GenericDescriptor dstDesc(cfg.backend, dstDims, cfg.dtypeIn);

    const std::size_t count = generic_element_count(*srcDesc);
    const std::size_t srcBytes = generic_byte_size(*srcDesc, cfg.dtypeIn);
    const std::size_t dstBytes = generic_byte_size(*dstDesc, cfg.dtypeIn);

    // (1) Host golden model. The op writes every output element, so golden needs no pre-seeding.
    std::vector<T> input(count), golden(count), actual(count);
    fill_input_nd<T>(input.data(), *srcDesc, cfg.dtypeIn);
    transpose_reference<T>(input.data(), golden.data(), *srcDesc, *dstDesc, perm.data());

    // (2) permTensor and roiTensor live in host-accessible (pinned for HIP) memory. permTensor
    // holds nDim values shared by the whole batch (the batch axis is not permuted).
    PinnedArray<Rpp32u> permTensor(cfg.backend, perm.size());
    for (std::size_t i = 0; i < perm.size(); ++i) permTensor[i] = perm[i];

    const std::vector<Rpp32u> roiVec = make_nd_roi_tensor(srcDims);
    PinnedArray<Rpp32u> roi(cfg.backend, roiVec.size());
    for (std::size_t i = 0; i < roiVec.size(); ++i) roi[i] = roiVec[i];

    // (3) Run RPP on the configured backend.
    DeviceTensor src(cfg.backend, srcBytes), dst(cfg.backend, dstBytes);
    src.write(input.data(), srcBytes);

    RppHandle handle(cfg.backend, srcDims[0]);
    ASSERT_EQ(rppt_transpose(src.ptr(), srcDesc.get(), dst.ptr(), dstDesc.get(), permTensor.data(),
                             roi.data(), handle.get(), cfg.backend),
              RPP_SUCCESS);

    handle.sync();  // drain the op's stream before copying results back
    dst.read(actual.data(), dstBytes);

    // (4) Compare the whole output tensor bit-exactly: transpose only moves elements.
    EXPECT_TRUE(compare_nd<T>(actual.data(), golden.data(), *dstDesc, 0.0, 0.0));
}

}  // namespace

// Full name: Misc_Geometric/TransposeTest.Correctness/<Backend>_<DType>to<DType>_<Rank>_<Perm>_<Shape>
// (the shape token is the *source* shape; the destination shape is that shape permuted).
class TransposeTest : public SkipListTest<NdWithParams<TransposeParams>> {};

TEST_P(TransposeTest, Correctness) {
    const NdConfig cfg = GetParam().cfg;
    const TransposeParams p = GetParam().op;
    dispatch_dtype<DType::U8, DType::I8, DType::F16, DType::F32>(cfg.dtypeIn, [&](auto tag) {
        run_transpose<Element<decltype(tag)>>(cfg, p);
    });
}

// Every case is bit-exact (tolerance 0 on both terms, every dtype): transpose performs no
// arithmetic and no dtype conversion, so a single differing element is a real defect.
//
// The three permutation kinds are chosen to separate the failure modes an axis/stride mix-up
// produces: Identity is a pure copy (any diff means the op is broken outright), Reverse moves
// every axis, and RotateLeft is a non-involutive permutation (applying it twice is not the
// identity), which catches an implementation that inverts the perm -- reading source axis k
// from output axis perm[k] instead of the documented converse. The two are indistinguishable
// under Identity and Reverse, both of which are self-inverse.
//
// nd_extents() gives every axis a distinct length, so a swapped pair of axes cannot coincide
// with the correct answer, and a wrong-extent destination fails to even allocate identically.
//
// Note the ND descriptors must be device-addressable on HIP (GenericDescriptor pins them): at
// rank >= 4 the ND kernels read dims/strides on the device. Undocumented and rank-dependent.
INSTANTIATE_TEST_SUITE_P(Misc_Geometric, TransposeTest,
                         ::testing::ValuesIn(nd_with_params<TransposeParams>(
                             make_nd_configs({DType::U8, DType::F16, DType::F32, DType::I8},
                                             {2, 3, 4}),
                             {{PermKind::Identity}, {PermKind::Reverse},
                              {PermKind::RotateLeft}})),
                         nd_op_config_name<TransposeParams>);
