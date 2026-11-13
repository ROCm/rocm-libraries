// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include "harness/BundleMetadata.hpp"
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
    bool onDevice,
    // The case asked for the non-finite mask: float, half, bfloat16 and double outputs are
    // filled with a NaN carrying a distinctive payload (NonFiniteMask.hpp), so an element the
    // engine never wrote still fails when NaN is matched against NaN. Only the engine run's
    // outputs ask for it; a reference's keep the plain sentinel.
    bool payloadNanSentinel = false);

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

/// Which side of a tensor's declared alignment N a placed pointer lands on.
enum class PlacedAt
{
    EXACT, ///< aligned to N and not to 2N
    UNDER, ///< aligned to N/2 and not to N, so a backend enforcing N must reject it
};

/// The address a pointer is moved to, given the base of the shadow buffer it moves into.
///
/// EXACT: alignUp(base, 2N) + N, so `address % N == 0` and `address % 2N == N`.
/// UNDER: alignUp(base, 2N) + N/2, so `address % N == N/2`. Needs N >= 2; throws
/// std::invalid_argument for a smaller N, and for N <= 0.
/// Pure arithmetic: the only part of placement that needs no device.
std::uintptr_t placedAddress(std::uintptr_t base, std::int64_t alignment, PlacedAt at);

/// Bytes a shadow buffer needs beyond the tensor itself so that placedAddress() stays
/// inside it whatever the base is: alignUp(base, 2N) <= base + 2N - 1, plus at most N more.
std::uint64_t placementSlack(std::int64_t alignment);

/// A placement that could not be set up: a tensor that cannot be placed, or a device
/// failure while building the shadows. A fault in the harness or the device, never the
/// engine's.
class PlacementError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

/// One device buffer of the engine run that a placement may move.
struct PlacementCandidate
{
    int64_t uid = 0;
    hipdnn_data_sdk::utilities::ITensor* tensor = nullptr;
    bool isOutput = false;
    std::int64_t alignment = 16; ///< the tensor's declared alignment N
};

/// The buffers of an engine run's variant pack a placement may move, and the ones it has to
/// leave where they are.
struct PlacementPlan
{
    std::vector<PlacementCandidate> candidates;
    /// "uid N: why", for each buffer that stays in place: a pass-by-value scalar (a host
    /// pointer), an input of the sub-byte packed set, a tensor with no declared alignment.
    std::vector<std::string> leftInPlace;
};

/// Sorts the entries of `variantPack` into candidates and left-in-place buffers.
///
/// `inputs` is the set the pack was built from. `packedInputs` says that set is the sub-byte
/// packed copy, whose buffers are left in place. Throws std::invalid_argument for a pack
/// entry that has no tensor in `inputs` or `outputs`.
PlacementPlan planPlacement(
    const VariantPack& variantPack,
    TensorMap& inputs,
    OutputTensors& outputs,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorAttributes,
    bool packedInputs);

/// RAII shadows of an engine run's device buffers, each placed at a chosen alignment.
///
/// Each placed tensor gets a device allocation of `bytes + placementSlack(N)`; the pointer
/// handed to the engine is placedAddress() inside it, and `variantPack` is rewritten to it.
/// The tensor's current device contents are copied into the shadow, so an input is intact
/// and an output still holds the sentinel the harness wrote. After the engine ran,
/// copyBackOutputs() copies the output shadows over the original buffers, which is what the
/// comparison reads. The placed pointers exist only for the engine run: a reference run is
/// never placed.
///
/// EXACT places every candidate at its own declared alignment. UNDER places only
/// `underUid`, which must be a candidate whose alignment is at least 2.
class PlacedVariantPack
{
public:
    PlacedVariantPack() = default;
    PlacedVariantPack(VariantPack& variantPack,
                      const std::vector<PlacementCandidate>& candidates,
                      CasePlacement placement,
                      std::optional<int64_t> underUid);
    ~PlacedVariantPack();

    PlacedVariantPack(const PlacedVariantPack&) = delete;
    PlacedVariantPack& operator=(const PlacedVariantPack&) = delete;
    PlacedVariantPack(PlacedVariantPack&&) noexcept = default;
    PlacedVariantPack& operator=(PlacedVariantPack&& other) noexcept;

    /// Waits for the engine, copies every output shadow over its original buffer, and waits
    /// again. Throws PlacementError on a device failure.
    void copyBackOutputs();

    /// The uids that were moved, in the order they were placed.
    const std::vector<int64_t>& placedUids() const
    {
        return _placedUids;
    }

private:
    struct Shadow
    {
        void* base = nullptr; ///< what hipMalloc returned
        void* original = nullptr; ///< the tensor's own device buffer
        void* placed = nullptr; ///< what the engine was handed
        std::size_t bytes = 0;
        bool isOutput = false;
    };

    void release() noexcept;

    std::vector<Shadow> _shadows;
    std::vector<int64_t> _placedUids;
};

} // namespace hipdnn_integration_tests::bundle::detail
