// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>

#include "compilation/ICompiledProgram.hpp"
#include "compilation/IKernelCompiler.hpp"
#include "compilation/IRunnableKernel.hpp"

namespace hip_kernel_provider::batchnorm::test
{

/// Declared alignment, in bytes, by tensor name.
using AlignmentByName = std::map<std::string, int64_t>;

/// Re-serializes a graph with each named tensor declaring the given alignment. The builders in the
/// test SDK always declare the default alignment (16 bytes), so a graph with another alignment has
/// to be patched after the fact. Tensors that are not named keep the alignment they declared.
inline flatbuffers::DetachedBuffer withNamedAlignments(const void* graphBuffer,
                                                       const AlignmentByName& alignmentByName)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    const auto graphT = std::unique_ptr<GraphT>(GetGraph(graphBuffer)->UnPack());
    size_t patched = 0;
    for(auto& tensor : graphT->tensors)
    {
        const auto found = alignmentByName.find(tensor->name);
        if(found != alignmentByName.end())
        {
            tensor->alignment = found->second;
            ++patched;
        }
    }
    EXPECT_GE(patched, alignmentByName.size()) << "a named tensor is not in the graph";

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(CreateGraph(builder, graphT.get()));
    return builder.Release();
}

/// Asserts that the named tensors of the graph declare exactly the given alignments. Guards the
/// acceptance checks against a re-serialization that silently kept the default alignment, which
/// would make them pass for the wrong reason.
inline void expectDeclaredAlignments(const void* graphBuffer,
                                     const AlignmentByName& alignmentByName)
{
    const auto* graph = hipdnn_flatbuffers_sdk::data_objects::GetGraph(graphBuffer);
    for(const auto* tensor : *graph->tensors())
    {
        const auto found = alignmentByName.find(tensor->name()->str());
        if(found != alignmentByName.end())
        {
            EXPECT_EQ(tensor->alignment(), found->second) << "tensor " << found->first;
        }
    }
}

/// Whether @p isApplicable accepts the graph once the named tensors declare the given alignments.
///
/// @param isApplicable Callable taking `const IGraph&` and returning bool.
template <typename IsApplicableFn>
bool isApplicableWithAlignments(const flatbuffers::FlatBufferBuilder& graph,
                                const AlignmentByName& alignments,
                                IsApplicableFn&& isApplicable)
{
    const auto patched = withNamedAlignments(graph.GetBufferPointer(), alignments);
    expectDeclaredAlignments(patched.data(), alignments);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper(patched.data(),
                                                                             patched.size());
    return isApplicable(wrapper);
}

/// Names of the tensors a plan accesses through a device pointer: the graph's non-virtual tensors
/// that are not passed by value.
inline std::vector<std::string> deviceTensorNames(const void* graphBuffer)
{
    std::vector<std::string> names;
    const auto* graph = hipdnn_flatbuffers_sdk::data_objects::GetGraph(graphBuffer);
    for(const auto* tensor : *graph->tensors())
    {
        if(!tensor->virtual_() && !hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(tensor))
        {
            names.push_back(tensor->name()->str());
        }
    }
    return names;
}

/// Asserts that @p isApplicable accepts @p valid as built (default alignment) and still accepts it
/// when every device tensor declares its element size, then declines it once any one device tensor
/// declares @p belowElementBytes, which is under the element size. The accepted baseline is what
/// makes the declines mean something: a check that refuses every graph cannot pass.
///
/// @param elementBytes Element size of the graph's device tensors, all of one data type.
/// @param belowElementBytes An alignment under @p elementBytes.
/// @param isApplicable Callable taking `const IGraph&` and returning bool.
template <typename IsApplicableFn>
void expectDeclinedBelowElementAlignment(const flatbuffers::FlatBufferBuilder& valid,
                                         int64_t elementBytes,
                                         int64_t belowElementBytes,
                                         IsApplicableFn&& isApplicable)
{
    {
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper baseline(
            valid.GetBufferPointer(), valid.GetSize());
        EXPECT_TRUE(isApplicable(baseline)) << "the graph at default alignment must be accepted";
    }

    const auto names = deviceTensorNames(valid.GetBufferPointer());
    ASSERT_FALSE(names.empty());

    AlignmentByName everyTensorAtElementSize;
    for(const auto& name : names)
    {
        everyTensorAtElementSize[name] = elementBytes;
    }
    EXPECT_TRUE(isApplicableWithAlignments(valid, everyTensorAtElementSize, isApplicable))
        << "element-size alignment was declined";

    for(const auto& name : names)
    {
        EXPECT_FALSE(isApplicableWithAlignments(valid, {{name, belowElementBytes}}, isApplicable))
            << "tensor " << name << " declaring " << belowElementBytes << " bytes was accepted";
    }
}

/// A kernel that accepts every launch setting and does nothing. Hand-written rather than a gmock
/// mock: the plans only configure kernels here, and none of the tests that use it verify the calls.
class RecordingKernel : public compilation::IRunnableKernel
{
public:
    void setBlockSize(unsigned int /*x*/, unsigned int /*y*/, unsigned int /*z*/) override {}
    void setGridSize(unsigned int /*x*/, unsigned int /*y*/, unsigned int /*z*/) override {}
    void setSharedMemBytes(unsigned int /*bytes*/) override {}

protected:
    void launchImpl(hipStream_t /*stream*/, void** /*args*/) const override {}
};

/// A compiled program that hands back as many kernels as the plan asks for.
class RecordingProgram : public compilation::ICompiledProgram
{
public:
    std::unique_ptr<compilation::IRunnableKernel>
        getKernel(const std::string& /*name*/) const override
    {
        return std::make_unique<RecordingKernel>();
    }
};

/// A kernel compiler that records the options of every compile it is asked for.
class RecordingCompiler : public compilation::IKernelCompiler
{
public:
    std::unique_ptr<compilation::ICompiledProgram>
        compile(const std::string& /*source*/,
                const std::vector<std::string>& options) const override
    {
        _options = options;
        ++_compileCount;
        return std::make_unique<RecordingProgram>();
    }

    const std::vector<std::string>& options() const
    {
        return _options;
    }

    size_t compileCount() const
    {
        return _compileCount;
    }

private:
    mutable std::vector<std::string> _options;
    mutable size_t _compileCount = 0;
};

/// Runs @p compile against a recording kernel compiler, asserts that it compiled exactly once, and
/// returns the options of that compile.
///
/// @param compile Callable taking `const IKernelCompiler&` that compiles one plan.
template <typename CompileFn>
std::vector<std::string> compileAndCaptureOptions(CompileFn&& compile)
{
    const RecordingCompiler compiler;
    compile(compiler);
    EXPECT_EQ(compiler.compileCount(), 1U);
    return compiler.options();
}

/// Whether the recorded compile options contain @p option, such as "-DHIP_PLUGIN_BN_VARIANT=2".
inline bool hasCompileOption(const std::vector<std::string>& options, const std::string& option)
{
    return std::find(options.begin(), options.end(), option) != options.end();
}

} // namespace hip_kernel_provider::batchnorm::test
