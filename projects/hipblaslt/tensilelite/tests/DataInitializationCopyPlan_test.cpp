// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <any>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include <hip/hip_runtime.h>

#include <Tensile/Utils.hpp>

#include "ClientProblemFactory.hpp"
#include "DataInitialization.hpp"
#include "DataInitializationTestUtils.hpp"

using namespace TensileLite;
using namespace TensileLite::Client;

namespace
{
    using TensileLite::testing::PlainProblemSpec;
    using TensileLite::testing::makePlainProblem;

    class CopyPlanDataInitialization : public DataInitialization
    {
    public:
        using DataInitialization::DataInitialization;
        using TensorCopyOp        = DataInitialization::TensorCopyOp;
        using TensorCopyPlan      = DataInitialization::TensorCopyPlan;
        using TensorCopyPlanKind  = DataInitialization::TensorCopyPlanKind;
        using PristineUnit        = DataInitialization::PristineUnit;
        using DataInitialization::copyInputs;
        using DataInitialization::effectiveStreamForOp;
        using DataInitialization::executeTensorCopyPlan;
        using DataInitialization::planInputCopies;
        using DataInitialization::planOutputResetCopyOps;
        using DataInitialization::resetOutput;

        std::vector<void*>& gpuPtrs()
        {
            return m_gpuPtrs;
        }

        std::vector<void**>& gpuBatchPtrs()
        {
            return m_gpuBatchPtrs;
        }

        std::vector<size_t>& maxElements()
        {
            return m_maxElements;
        }

        std::vector<std::vector<size_t>>& groupedOffsets()
        {
            return m_groupedOffsets;
        }

        bool altSlotsReady() const
        {
            return m_altSlotsReady;
        }

        auto const& slotState(size_t slot) const
        {
            return m_gpuInputSlots.at(slot);
        }

        PristineUnit const& pristineUnit(size_t tensorIndex,
                                         ContractionProblemGemm const& problem) const
        {
            auto const& desc = problem.tensors().at(tensorIndex);
            auto const& units = m_vdata.at(tensorIndex).pristine;
            auto        it    = units.find(desc.dataType());
            if(it == units.end())
            {
                throw std::runtime_error("Missing pristine unit for tensor index.");
            }
            return it->second;
        }
    };

    ::testing::AssertionResult hasHipDevice()
    {
        int        deviceCount = 0;
        hipError_t err         = hipGetDeviceCount(&deviceCount);
        if(err != hipSuccess)
        {
            return ::testing::AssertionFailure()
                   << "hipGetDeviceCount failed: " << hipGetErrorString(err);
        }

        if(deviceCount <= 0)
            return ::testing::AssertionFailure() << "No HIP devices available";

        return ::testing::AssertionSuccess();
    }

    Client::po::variables_map makeBaseArgs(std::vector<std::vector<size_t>> problemSizes)
    {
        auto args = TensileLite::testing::buildBaseDataInitArgs(std::move(problemSizes));
        TensileLite::testing::detail::setDataInitArg(args,
                                                     "num-elements-to-validate",
                                                     std::any(int(1)));
        return args;
    }

    Client::po::variables_map makeRingArgs(std::vector<std::vector<size_t>> problemSizes)
    {
        return TensileLite::testing::buildRingArgs(std::move(problemSizes), 1);
    }

    Client::po::variables_map makeGuardPageBackArgs(std::vector<std::vector<size_t>> problemSizes,
                                                    bool swizzleTensorA = false,
                                                    bool swizzleTensorB = false)
    {
        auto args = makeBaseArgs(std::move(problemSizes));
        TensileLite::testing::detail::setDataInitArg(args,
                                                     "bounds-check",
                                                     std::any(BoundsCheckMode::GuardPageBack));
        TensileLite::testing::detail::setDataInitArg(args,
                                                     "swizzle-tensor-a",
                                                     std::any(swizzleTensorA));
        TensileLite::testing::detail::setDataInitArg(args,
                                                     "swizzle-tensor-b",
                                                     std::any(swizzleTensorB));
        return args;
    }

    ContractionProblemGemm makeBatchProblem(size_t m,
                                            size_t n,
                                            size_t k,
                                            size_t batch,
                                            bool   swizzleTensorA = false,
                                            bool   swizzleTensorB = false)
    {
        PlainProblemSpec spec;
        spec.m     = m;
        spec.n     = n;
        spec.k     = k;
        spec.batch = batch;

        auto problem = makePlainProblem(spec);
        problem.setSwizzleTensorA(swizzleTensorA);
        problem.setSwizzleTensorB(swizzleTensorB);
        return problem;
    }

    size_t swizzleMiK(rocisa::DataType dt, size_t& miKv)
    {
        switch(dt)
        {
        case rocisa::DataType::Float:
        case rocisa::DataType::Double:
            miKv = 1;
            return 4;
        case rocisa::DataType::XFloat32:
            miKv = 2;
            return 8;
        case rocisa::DataType::Half:
        case rocisa::DataType::BFloat16:
            miKv = 4;
            return 16;
        case rocisa::DataType::Int8:
        case rocisa::DataType::Float8_fnuz:
        case rocisa::DataType::BFloat8_fnuz:
        case rocisa::DataType::Float8BFloat8_fnuz:
        case rocisa::DataType::BFloat8Float8_fnuz:
        case rocisa::DataType::Float8:
        case rocisa::DataType::BFloat8:
        case rocisa::DataType::Float8BFloat8:
        case rocisa::DataType::BFloat8Float8:
        case rocisa::DataType::E8:
        case rocisa::DataType::E5M3:
            miKv = 8;
            return 32;
        default:
            throw std::runtime_error("unsupported datatype for swizzling");
        }
    }

    ptrdiff_t expectedGuardBackPadding(TensorDescriptor const& desc)
    {
        size_t miKv  = 0;
        size_t miK   = swizzleMiK(desc.dataType(), miKv);
        size_t packK = 16 / miKv / rocisa::GetElementSize(desc.dataType());

        auto const k         = desc.sizes()[0];
        auto const m_n       = desc.sizes()[1];
        auto const b         = desc.sizes()[2];
        auto const swizzleK  = miK * packK;
        auto const paddedMN  = (m_n + 16 - 1) / 16 * 16;
        auto const paddedK   = (k + swizzleK - 1) / swizzleK * swizzleK;
        auto const allocated = desc.totalAllocatedElements();

        return static_cast<ptrdiff_t>(paddedMN * paddedK * b - allocated);
    }
} // namespace

TEST(DataInitializationCopyPlan, ExecutorPreservesD2DOnlyStreamForwarding)
{
    using Op = CopyPlanDataInitialization::TensorCopyOp;

    Op h2h{};
    h2h.kind = hipMemcpyHostToHost;
    Op h2d{};
    h2d.kind = hipMemcpyHostToDevice;
    Op d2d{};
    d2d.kind = hipMemcpyDeviceToDevice;

    hipStream_t const sentinel = reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(0x1234));

    EXPECT_EQ(CopyPlanDataInitialization::effectiveStreamForOp(h2h, sentinel), nullptr);
    EXPECT_EQ(CopyPlanDataInitialization::effectiveStreamForOp(h2d, sentinel), nullptr);
    EXPECT_EQ(CopyPlanDataInitialization::effectiveStreamForOp(d2d, sentinel), sentinel);
}

TEST(DataInitializationCopyPlan, ExecutorReturnsTensorIndexedResults)
{
    auto hipDevice = hasHipDevice();
    if(!hipDevice)
    {
        GTEST_SKIP() << hipDevice.message();
    }

    auto problem = makeBatchProblem(32, 24, 16, 4);
    auto args    = makeBaseArgs({{32, 24, 4, 16}});

    ClientProblemFactory             factory(args);
    CopyPlanDataInitialization       dataInit(args, factory);
    auto const plan = dataInit.planInputCopies(problem, hipMemcpyDeviceToDevice);
    auto const results = dataInit.executeTensorCopyPlan(plan, nullptr);

    ASSERT_EQ(results.size(), plan.opsByTensor.size());
    ASSERT_EQ(plan.opsByTensor.size(), problem.tensors().size());

    size_t missingCount = 0;
    for(size_t i = 0; i < plan.opsByTensor.size(); ++i)
    {
        if(plan.opsByTensor[i])
        {
            EXPECT_NE(results[i], nullptr);
        }
        else
        {
            EXPECT_EQ(results[i], nullptr);
            ++missingCount;
        }
    }

    EXPECT_GT(missingCount, 0u);
}

TEST(DataInitializationCopyPlan, InputPlanIsPureAndMatchesViewMetadata)
{
    auto hipDevice = hasHipDevice();
    if(!hipDevice)
    {
        GTEST_SKIP() << hipDevice.message();
    }

    auto problem = makeBatchProblem(32, 24, 16, 4);
    auto args    = makeBaseArgs({{32, 24, 4, 16}});

    ClientProblemFactory       factory(args);
    CopyPlanDataInitialization  dataInit(args, factory);
    auto const plan = dataInit.planInputCopies(problem, hipMemcpyDeviceToDevice);
    auto const expectedResults = dataInit.executeTensorCopyPlan(plan, nullptr);

    ASSERT_EQ(plan.opsByTensor.size(), problem.tensors().size());
    ASSERT_EQ(expectedResults.size(), plan.opsByTensor.size());

    std::vector<void*>              ptrs(problem.tensors().size(), reinterpret_cast<void*>(0x1));
    std::vector<void**>             batchPtrs(problem.tensors().size(),
                                  reinterpret_cast<void**>(0x2));
    std::vector<size_t>             maxElements(problem.tensors().size(), 9999);
    std::vector<std::vector<size_t>> offsets(problem.tensors().size(), std::vector<size_t>{7, 7});

    auto ptrsBefore      = ptrs;
    auto batchPtrsBefore = batchPtrs;
    auto maxBefore       = maxElements;
    auto offsetsBefore   = offsets;

    EXPECT_EQ(ptrs, ptrsBefore);
    EXPECT_EQ(batchPtrs, batchPtrsBefore);
    EXPECT_EQ(maxElements, maxBefore);
    EXPECT_EQ(offsets, offsetsBefore);

    dataInit.copyInputs(ptrs,
                        batchPtrs,
                        maxElements,
                        offsets,
                        problem,
                        hipMemcpyDeviceToDevice);

    std::vector<void*>              expectedPtrs;
    std::vector<void**>             expectedBatchPtrs;
    std::vector<size_t>             expectedMaxElements;
    std::vector<std::vector<size_t>> expectedOffsets;
    expectedPtrs.reserve(plan.opsByTensor.size());
    expectedBatchPtrs.reserve(plan.opsByTensor.size());
    expectedMaxElements.reserve(plan.opsByTensor.size());
    expectedOffsets.reserve(plan.opsByTensor.size());

    for(size_t i = 0; i < plan.opsByTensor.size(); ++i)
    {
        auto const& maybeOp = plan.opsByTensor[i];
        if(maybeOp)
        {
            auto const& op       = *maybeOp;
            auto const& pristine = dataInit.pristineUnit(i, problem);

            ASSERT_NE(op.descriptor, nullptr);
            EXPECT_EQ(op.tensorIndex, i);
            EXPECT_EQ(op.descriptor, &problem.tensors().at(i));
            EXPECT_EQ(op.kind, hipMemcpyDeviceToDevice);
            EXPECT_EQ(op.planKind, CopyPlanDataInitialization::TensorCopyPlanKind::Plain);
            EXPECT_NE(op.dst, nullptr);
            EXPECT_NE(op.src, nullptr);
            EXPECT_EQ(op.maxElements, pristine.maxElements);
            EXPECT_EQ(op.batchPtr, pristine.gpuInput.batch.get());
            EXPECT_EQ(op.groupedOffsets, pristine.groupedGemmOffsets);

            expectedPtrs.push_back(expectedResults[i]);
            expectedBatchPtrs.push_back(op.batchPtr);
            expectedMaxElements.push_back(op.maxElements);
            expectedOffsets.push_back(op.groupedOffsets);
        }
        else
        {
            expectedPtrs.push_back(nullptr);
            expectedBatchPtrs.push_back(nullptr);
            expectedMaxElements.push_back(0);
            expectedOffsets.emplace_back();
        }
    }

    EXPECT_EQ(ptrs, expectedPtrs);
    EXPECT_EQ(batchPtrs, expectedBatchPtrs);
    EXPECT_EQ(maxElements, expectedMaxElements);
    EXPECT_EQ(offsets, expectedOffsets);
}

TEST(DataInitializationCopyPlan, InputGuardPageBackPlansGuardBackOps)
{
    auto hipDevice = hasHipDevice();
    if(!hipDevice)
    {
        GTEST_SKIP() << hipDevice.message();
    }

    auto problem = makeBatchProblem(17, 19, 23, 4);
    auto args    = makeGuardPageBackArgs({{17, 19, 4, 23}});

    ClientProblemFactory             factory(args);
    CopyPlanDataInitialization       dataInit(args, factory);
    auto const plan = dataInit.planInputCopies(problem, hipMemcpyDeviceToDevice);

    ASSERT_EQ(plan.opsByTensor.size(), problem.tensors().size());

    for(size_t i = 0; i < plan.opsByTensor.size(); ++i)
    {
        auto const& maybeOp = plan.opsByTensor[i];
        if(!maybeOp)
            continue;

        auto const& op       = *maybeOp;
        auto const& pristine = dataInit.pristineUnit(i, problem);

        EXPECT_EQ(op.tensorIndex, i);
        EXPECT_EQ(op.descriptor, &problem.tensors().at(i));
        EXPECT_EQ(op.kind, hipMemcpyDeviceToDevice);
        EXPECT_EQ(op.planKind, CopyPlanDataInitialization::TensorCopyPlanKind::GuardBack);
        EXPECT_NE(op.dst, nullptr);
        EXPECT_NE(op.src, nullptr);
        EXPECT_EQ(op.bad, nullptr);
        EXPECT_EQ(op.maxElements, pristine.maxElements);
        EXPECT_EQ(op.batchPtr, pristine.gpuInput.batch.get());
        EXPECT_EQ(op.groupedOffsets, pristine.groupedGemmOffsets);
        EXPECT_EQ(op.customPadding, -1);
    }
}

TEST(DataInitializationCopyPlan, InputGuardPageBackSwizzledTensorsUseCustomPadding)
{
    auto hipDevice = hasHipDevice();
    if(!hipDevice)
    {
        GTEST_SKIP() << hipDevice.message();
    }

    auto problem = makeBatchProblem(17, 19, 23, 4, true, true);
    auto args    = makeGuardPageBackArgs({{17, 19, 4, 23}}, true, true);

    ClientProblemFactory             factory(args);
    CopyPlanDataInitialization       dataInit(args, factory);
    auto const plan = dataInit.planInputCopies(problem, hipMemcpyDeviceToDevice);

    ASSERT_EQ(plan.opsByTensor.size(), problem.tensors().size());

    for(size_t i = 0; i < plan.opsByTensor.size(); ++i)
    {
        auto const& maybeOp = plan.opsByTensor[i];
        if(!maybeOp)
            continue;

        auto const& op = *maybeOp;
        if(i == ContractionProblemGemm::TENSOR::A || i == ContractionProblemGemm::TENSOR::B)
        {
            EXPECT_EQ(op.planKind, CopyPlanDataInitialization::TensorCopyPlanKind::GuardBack);
            EXPECT_EQ(op.customPadding, expectedGuardBackPadding(*op.descriptor));
        }
    }
}

TEST(DataInitializationCopyPlan, OutputResetPlanOnlyTargetsOutputs)
{
    auto hipDevice = hasHipDevice();
    if(!hipDevice)
    {
        GTEST_SKIP() << hipDevice.message();
    }

    auto problem = makeBatchProblem(32, 24, 16, 4);
    auto args    = makeGuardPageBackArgs({{32, 24, 4, 16}});

    ClientProblemFactory             factory(args);
    CopyPlanDataInitialization       dataInit(args, factory);

    auto inputs = dataInit.prepareGPUInputs(static_cast<ContractionProblem const*>(&problem));
    ASSERT_NE(inputs, nullptr);

    auto beforePtrs      = dataInit.gpuPtrs();
    auto beforeBatchPtrs = dataInit.gpuBatchPtrs();
    auto beforeMax       = dataInit.maxElements();
    auto beforeOffsets   = dataInit.groupedOffsets();

    auto const plan = dataInit.planOutputResetCopyOps(problem, hipMemcpyDeviceToDevice);
    auto const expectedResults = dataInit.executeTensorCopyPlan(plan, nullptr);

    ASSERT_EQ(plan.opsByTensor.size(), problem.tensors().size());
    ASSERT_EQ(expectedResults.size(), plan.opsByTensor.size());

    dataInit.resetOutput(dataInit.gpuPtrs(),
                         dataInit.gpuBatchPtrs(),
                         dataInit.maxElements(),
                         dataInit.groupedOffsets(),
                         problem,
                         hipMemcpyDeviceToDevice);

    std::vector<void*>              expectedPtrs      = beforePtrs;
    std::vector<void**>             expectedBatchPtrs = beforeBatchPtrs;
    std::vector<size_t>             expectedMax       = beforeMax;
    std::vector<std::vector<size_t>> expectedOffsets   = beforeOffsets;

    for(size_t i = 0; i < plan.opsByTensor.size(); ++i)
    {
        auto const& desc    = problem.tensors().at(i);
        auto const& maybeOp = plan.opsByTensor[i];
        if(!desc.isOutput())
        {
            EXPECT_FALSE(maybeOp.has_value());
            continue;
        }

        ASSERT_TRUE(maybeOp.has_value());
        auto const& op = *maybeOp;
        EXPECT_EQ(op.tensorIndex, i);
        EXPECT_EQ(op.planKind, CopyPlanDataInitialization::TensorCopyPlanKind::Plain);
        EXPECT_NE(op.dst, nullptr);
        EXPECT_NE(op.src, nullptr);
        EXPECT_EQ(op.dst, expectedResults[i]);
        EXPECT_NE(op.dst, beforePtrs[i]);

        expectedPtrs[i]      = expectedResults[i];
        expectedBatchPtrs[i] = op.batchPtr;
        expectedMax[i]       = op.maxElements;
        expectedOffsets[i]   = op.groupedOffsets;
    }

    EXPECT_EQ(dataInit.gpuPtrs(), expectedPtrs);
    EXPECT_EQ(dataInit.gpuBatchPtrs(), expectedBatchPtrs);
    EXPECT_EQ(dataInit.maxElements(), expectedMax);
    EXPECT_EQ(dataInit.groupedOffsets(), expectedOffsets);
}

TEST(DataInitializationCopyPlan, InputPlanUsesDefaultAndExplicitGpuSlots)
{
    auto hipDevice = hasHipDevice();
    if(!hipDevice)
    {
        GTEST_SKIP() << hipDevice.message();
    }

    auto problem = makeBatchProblem(32, 24, 16, 4);
    auto args    = makeRingArgs({{32, 24, 4, 16}});

    ClientProblemFactory      factory(args);
    CopyPlanDataInitialization dataInit(args, factory);

    auto inputs = dataInit.prepareGPUInputs(static_cast<ContractionProblem const*>(&problem));
    ASSERT_NE(inputs, nullptr);
    ASSERT_TRUE(dataInit.altSlotsReady());

    auto const& slot0 = dataInit.slotState(0);
    auto const& slot1 = dataInit.slotState(1);
    ASSERT_TRUE(slot0.populated());
    ASSERT_TRUE(slot1.populated());

    auto const defaultPlan = dataInit.planInputCopies(problem, hipMemcpyDeviceToDevice);
    auto const explicitPlan
        = dataInit.planInputCopies(problem, hipMemcpyDeviceToDevice, 1);

    ASSERT_EQ(defaultPlan.opsByTensor.size(), problem.tensors().size());
    ASSERT_EQ(explicitPlan.opsByTensor.size(), problem.tensors().size());

    for(size_t i = 0; i < problem.tensors().size(); ++i)
    {
        auto const& maybeDefault = defaultPlan.opsByTensor[i];
        auto const& maybeExplicit = explicitPlan.opsByTensor[i];
        ASSERT_TRUE(maybeDefault.has_value());
        ASSERT_TRUE(maybeExplicit.has_value());

        auto const& opDefault  = *maybeDefault;
        auto const& opExplicit = *maybeExplicit;
        auto const& pristine   = dataInit.pristineUnit(i, problem);

        EXPECT_EQ(opDefault.dst, slot0.ptrs.at(i));
        EXPECT_EQ(opDefault.dst, pristine.gpuInput.current.get());
        EXPECT_EQ(opDefault.src, pristine.gpuInput.valid.get());
        EXPECT_EQ(opDefault.batchPtr, slot0.batchPtrs.at(i));
        EXPECT_EQ(opDefault.batchPtr, pristine.gpuInput.batch.get());

        EXPECT_EQ(opExplicit.dst, slot1.ptrs.at(i));
        EXPECT_EQ(opExplicit.src, pristine.gpuInput.valid.get());
        EXPECT_EQ(opExplicit.batchPtr, slot1.batchPtrs.at(i));
        EXPECT_NE(opExplicit.dst, opDefault.dst);
        EXPECT_NE(opExplicit.batchPtr, opDefault.batchPtr);
    }
}

TEST(DataInitializationCopyPlan, OutputResetPlanUsesDefaultAndExplicitGpuSlots)
{
    auto hipDevice = hasHipDevice();
    if(!hipDevice)
    {
        GTEST_SKIP() << hipDevice.message();
    }

    auto problem = makeBatchProblem(32, 24, 16, 4);
    auto args    = makeRingArgs({{32, 24, 4, 16}});

    ClientProblemFactory      factory(args);
    CopyPlanDataInitialization dataInit(args, factory);

    auto inputs = dataInit.prepareGPUInputs(static_cast<ContractionProblem const*>(&problem));
    ASSERT_NE(inputs, nullptr);
    ASSERT_TRUE(dataInit.altSlotsReady());

    auto const& slot0 = dataInit.slotState(0);
    auto const& slot1 = dataInit.slotState(1);
    ASSERT_TRUE(slot0.populated());
    ASSERT_TRUE(slot1.populated());

    auto const defaultPlan = dataInit.planOutputResetCopyOps(problem, hipMemcpyDeviceToDevice);
    auto const explicitPlan
        = dataInit.planOutputResetCopyOps(problem, hipMemcpyDeviceToDevice, 1);

    ASSERT_EQ(defaultPlan.opsByTensor.size(), problem.tensors().size());
    ASSERT_EQ(explicitPlan.opsByTensor.size(), problem.tensors().size());

    for(size_t i = 0; i < problem.tensors().size(); ++i)
    {
        auto const& desc = problem.tensors().at(i);
        auto const& maybeDefault = defaultPlan.opsByTensor[i];
        auto const& maybeExplicit = explicitPlan.opsByTensor[i];

        if(!desc.isOutput())
        {
            EXPECT_FALSE(maybeDefault.has_value());
            EXPECT_FALSE(maybeExplicit.has_value());
            continue;
        }

        ASSERT_TRUE(maybeDefault.has_value());
        ASSERT_TRUE(maybeExplicit.has_value());

        auto const& opDefault  = *maybeDefault;
        auto const& opExplicit = *maybeExplicit;
        auto const& pristine   = dataInit.pristineUnit(i, problem);

        EXPECT_EQ(opDefault.dst, slot0.ptrs.at(i));
        EXPECT_EQ(opDefault.src, pristine.gpuInput.valid.get());
        EXPECT_EQ(opDefault.batchPtr, slot0.batchPtrs.at(i));
        EXPECT_EQ(opDefault.batchPtr, pristine.gpuInput.batch.get());

        EXPECT_EQ(opExplicit.dst, slot1.ptrs.at(i));
        EXPECT_EQ(opExplicit.src, pristine.gpuInput.valid.get());
        EXPECT_EQ(opExplicit.batchPtr, slot1.batchPtrs.at(i));
        EXPECT_NE(opExplicit.dst, opDefault.dst);
        EXPECT_NE(opExplicit.batchPtr, opDefault.batchPtr);
    }
}
