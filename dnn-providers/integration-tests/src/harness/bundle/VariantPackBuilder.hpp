// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include "harness/VariantPack.hpp"
#include "harness/bundle/IntegrationTestBundle.hpp"
#include "harness/bundle/OutputComparison.hpp"
#include "harness/input-init/TensorAliases.hpp"

namespace hipdnn_integration_tests::bundle::detail
{

/// The device could not be made to hold the output buffers: an allocation, a memset or
/// the wait for it failed. That is a fault in the harness or the device, not in any
/// executor, so a caller that runs a reference must not blame the reference for it.
class DeviceOutputError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

// Assembles the uid -> buffer map an executor is handed for one bundle.
//
// Shared by both harnesses: the engine harness builds one to run the engine, the
// reference harness builds one to run a reference. Its own translation unit so the
// golden-data binary can link one pure function without dragging in the engine
// harness.
//
// Inputs that are also outputs are skipped -- the output allocation owns that uid.
// `useDevice` selects host or device pointers; a runtime-pass-by-value input is
// always passed by value regardless.
//
// Each view in `aliases` gets its owner's pointer, advanced by the alias's byte
// offset. Throws std::invalid_argument when an owner has no entry, or when a view
// already has one as an input or an output.
VariantPack buildVariantPack(
    TensorMap& inputs,
    OutputTensors& outputs,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorAttributes,
    const std::vector<int64_t>& outputTensorUids,
    bool useDevice,
    const std::vector<TensorAlias>& aliases = {});

/// The output buffers one bundle is run into, each prefilled with the sentinel
/// value so a tensor nobody wrote is visibly untouched rather than plausibly zero.
///
/// `onDevice` says the buffers will be handed to an executor as device pointers. Their
/// sentinel is then written straight into device memory where the element type allows
/// it, rather than filled on the host and uploaded; the rest are filled on the host as
/// before. False never touches the device. A device failure while `onDevice` throws
/// DeviceOutputError.
///
/// Shared for the same reason buildVariantPack() is: both harnesses allocate the
/// same buffers from the same attributes, and two copies of that would drift.
///
/// A ragged output shares its offset tensor from `loadedTensors`, so it has the
/// same layout as the golden output it is compared against. So does its seq_len tensor:
/// `lengthUids` maps a ragged tensor uid to its node's seq_len uid (raggedLengthUids()),
/// and the tensor comes from `loadedTensors` too. Throws std::invalid_argument when
/// either is not loaded.
OutputTensors allocateSentinelOutputs(
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorAttributes,
    const std::vector<int64_t>& outputTensorUids,
    const TensorMap& loadedTensors,
    const std::unordered_map<int64_t, int64_t>& lengthUids,
    bool onDevice);

// Moves the engine's view of some tensors to a copy that starts a given number of bytes
// past an address aligned to K_START_OFFSET_BASE_ALIGNMENT. The tensors themselves do
// not move.
//
// The constructor makes one copy for each entry of `startOffsets`, of the output or else
// the input with that uid. It copies the tensor's bytes from its current variant-pack
// pointer and sets that pointer to the copy. It then sets the pointer of each alias view
// whose owner moved to the owner's new pointer plus the alias's byte offset. Throws
// std::invalid_argument when a uid appears twice or has no tensor or no variant-pack
// entry, and DeviceOutputError when a device call fails. After a throw, do not use the
// variant pack.
//
// waitForEngine() waits for the device work that the engine started on the moved copies
// and returns the result. It returns hipSuccess at once when no tensor moved or the
// copies are on the host. A failure there is the engine's.
//
// copyOutputsBack() copies each moved output back into its tensor. Call it after
// waitForEngine(). The destructor releases the copies, so the variant pack must not be
// used after it.
class StartOffsetStaging
{
public:
    StartOffsetStaging(VariantPack& variantPack,
                       TensorMap& inputs,
                       OutputTensors& outputs,
                       const std::vector<TensorStartOffset>& startOffsets,
                       const std::vector<TensorAlias>& aliases,
                       bool useDevice);
    ~StartOffsetStaging() = default;

    StartOffsetStaging(const StartOffsetStaging&) = delete;
    StartOffsetStaging& operator=(const StartOffsetStaging&) = delete;
    StartOffsetStaging(StartOffsetStaging&&) = delete;
    StartOffsetStaging& operator=(StartOffsetStaging&&) = delete;

    hipError_t waitForEngine() const;
    void copyOutputsBack();

private:
    struct StagedOutput
    {
        void* tensorData;
        const void* stagedData;
        std::size_t bytes;
    };

    void* stage(void* source, std::size_t bytes, int64_t startOffset);

    bool _useDevice;
    std::vector<hipdnn_data_sdk::utilities::Workspace<>> _deviceCopies;
    std::vector<std::vector<std::byte>> _hostCopies;
    std::vector<StagedOutput> _outputs;
};

/// Tell each tensor which side now holds the fresh data. Without it the comparison
/// reads the stale copy — silently, and in whichever direction is wrong.
void markOutputsModified(OutputTensors& outputs, bool device);

} // namespace hipdnn_integration_tests::bundle::detail
