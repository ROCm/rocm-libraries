// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/kernel_ingestor_engine/packs/PointwiseAddDispatchHandler.hpp"

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <string>

#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>

#include "compilation/KernelCompileOptions.hpp"
#include "engines/kernel_ingestor_engine/packs/PointwiseAddMatchers.hpp"
#include "engines/kernel_ingestor_engine/packs/PointwiseAddSymbols.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

/// Scratch reported by the larger-block kernel. Arbitrary: its only job is to make the
/// engine's max-across-survivors answer observably non-zero.
constexpr size_t LARGE_BLOCK_WORKSPACE_BYTES = 1024;
constexpr int64_t LARGE_BLOCK_SIZE = 256;

/// The compiled kernel plus the operand uids it launches with: everything read from the
/// graph, resolved once, owning nothing that points back into it.
class PreparedPointwiseAdd : public PreparedDispatch
{
public:
    PreparedPointwiseAdd(std::unique_ptr<compilation::ICompiledProgram> program,
                         std::unique_ptr<compilation::IRunnableKernel> kernel,
                         PointwiseAddBinding binding)
        : _program(std::move(program))
        , _kernel(std::move(kernel))
        , _binding(binding)
    {
    }

    const compilation::IRunnableKernel& kernel() const
    {
        return *_kernel;
    }

    const PointwiseAddBinding& binding() const
    {
        return _binding;
    }

private:
    // The runnable kernel is a view into its program's module, so the program must
    // outlive it; both are held here for the plan's lifetime.
    std::unique_ptr<compilation::ICompiledProgram> _program;
    std::unique_ptr<compilation::IRunnableKernel> _kernel;
    PointwiseAddBinding _binding;
};

/// The C++ type the kernel is compiled for, from the kernel's dtype metadata.
std::string elementTypeFor(const KernelDefinition& kernel)
{
    const auto& dtype = kernel.getStringMetadata(std::string(DTYPE_FIELD));
    if(dtype == "FLOAT")
    {
        return "float";
    }
    if(dtype == "HALF")
    {
        return "_Float16";
    }

    // Unreachable via matching, which admits only dtypes this pack declares. Reported
    // rather than defaulted: a silent fallback would compile the wrong kernel.
    throw hipdnn_plugin_sdk::HipdnnPluginException(
        HIPDNN_PLUGIN_STATUS_BAD_PARAM,
        "kernel '" + kernel.kernelId + "' declares unsupported dtype '" + dtype + "'");
}

const data_objects::TensorAttributes& firstInput(const MatchContext& context,
                                                 const PointwiseAddBinding& binding)
{
    const auto& tensors = context.graph.getTensorMap();
    auto it = tensors.find(binding.inputA);
    if(it == tensors.end() || it->second == nullptr)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "matched pointwise add has no tensor for uid " + std::to_string(binding.inputA));
    }
    return *it->second;
}

} // namespace

size_t PointwiseAddDispatchHandler::workspaceBytes(const KernelDefinition& kernel) const
{
    return kernel.getIntMetadata(std::string(BLOCK_SIZE_FIELD)) == LARGE_BLOCK_SIZE
               ? LARGE_BLOCK_WORKSPACE_BYTES
               : 0;
}

std::unique_ptr<PreparedDispatch>
    PointwiseAddDispatchHandler::prepare(const MatchContext& context,
                                         const KernelDefinition& kernel) const
{
    // Re-reads the operand uids the matcher bound, rather than re-deriving them here
    // with a second notion of what this graph looks like.
    const auto binding = pointwiseAddBinding(context);

    const auto blockSize
        = static_cast<unsigned int>(kernel.getIntMetadata(std::string(BLOCK_SIZE_FIELD)));

    compilation::KernelCompileOptions options(&firstInput(context, binding),
                                              context.deviceProperties);
    options.add("HIP_PLUGIN_POINTWISE_ADD_TYPE", elementTypeFor(kernel));
    options.add("HIP_PLUGIN_POINTWISE_ADD_BLOCK_SIZE", blockSize);

    auto program = _kernelCompiler.compile(kernel.sourceFile, options);
    auto runnableKernel = program->getKernel(kernel.entryPoint);

    // One element, so one workgroup; the block size comes from kernel metadata, which is
    // how a per-kernel launch quantity reaches a dispatch descriptor shared by the pack.
    runnableKernel->setBlockSize(blockSize, 1, 1);
    runnableKernel->setGridSize(1, 1, 1);

    return std::make_unique<PreparedPointwiseAdd>(
        std::move(program), std::move(runnableKernel), binding);
}

void PointwiseAddDispatchHandler::launch(const Handle& handle,
                                         const PreparedDispatch& prepared,
                                         const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                         uint32_t numDeviceBuffers,
                                         void* /*workspace*/) const
{
    const auto& preparedAdd = dynamic_cast<const PreparedPointwiseAdd&>(prepared);
    const auto& binding = preparedAdd.binding();

    const auto inputA
        = hipdnn_plugin_sdk::findDeviceBuffer(binding.inputA, deviceBuffers, numDeviceBuffers);
    const auto inputB
        = hipdnn_plugin_sdk::findDeviceBuffer(binding.inputB, deviceBuffers, numDeviceBuffers);
    const auto output
        = hipdnn_plugin_sdk::findDeviceBuffer(binding.output, deviceBuffers, numDeviceBuffers);

    preparedAdd.kernel().launch(handle.getStream(), inputA.ptr, inputB.ptr, output.ptr);
}

void registerPointwiseAddDispatch(const PointwiseAddDispatchHandler& handler)
{
    DispatchRegistry<Handle>::registerSymbol(std::string(DISPATCH_SYMBOL), &handler);
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
