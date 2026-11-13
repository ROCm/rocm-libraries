// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include "harness/input-init/InputFillRecipes.hpp"
#include "harness/input-init/TensorAliases.hpp"

namespace hipdnn_integration_tests
{

using InputTensorMap
    = std::unordered_map<int64_t, std::shared_ptr<hipdnn_data_sdk::utilities::ITensor>>;

struct FillResult
{
    enum class Status
    {
        OK,
        /// The harness cannot fill this graph's inputs.
        UNSUPPORTED,
        /// A recipe cannot fill its tensor as written, for example a VALUES list
        /// with a value out of range for the tensor's data type.
        INVALID,
    };

    bool filled = false;
    std::string reason;
    /// How many tensors were generated on the device rather than the host.
    std::size_t deviceFilled = 0;
    Status status = Status::UNSUPPORTED;

    static FillResult ok(std::size_t deviceFilled = 0)
    {
        return {true, {}, deviceFilled, Status::OK};
    }
    static FillResult unsupported(std::string why)
    {
        return {false, std::move(why), 0, Status::UNSUPPORTED};
    }
    static FillResult invalid(std::string why)
    {
        return {false, std::move(why), 0, Status::INVALID};
    }
};

// Fill used for an input of this data type when neither the test nor the op
// registers one. Covers every DataType except UNSET, which throws, so a new
// data type cannot fall through to a range that misses its encodings.
FillRecipe defaultFillFor(hipdnn_flatbuffers_sdk::data_objects::DataType dataType);

/// A device fill failed: generator creation, the scaling kernel's compile, a scratch
/// allocation, a launch, or the wait for them. A fault in the harness or the device,
/// never in the graph under test, so a caller reports it as such instead of skipping.
class DeviceInputError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

/// Generates large inputs straight into device memory with rocRAND, instead of by the host's
/// serial RNG, which costs seconds per tensor at the largest shapes. Takes the FREE, NONZERO,
/// ZEROS, LARGE, SUBNORMAL and SPECIAL kinds: rocRAND draws the uniform floats and a
/// hipRTC kernel turns them into the kind's values in the tensor's own data type. A tensor
/// filled this way is device-resident afterwards, so the first non-const host access
/// migrates it; a const access cannot.
///
/// The values are not the host fill's: the stream differs, and rocRAND's uniform draw
/// is in (0, 1], so a range is (lo, hi] rather than [lo, hi). Each kind keeps its host value
/// contract (range, sign, magnitude, the exact planted share of SPECIAL), but a NONZERO draw
/// that rounds to zero is redrawn once, not up to 64 times. A failure that cannot be
/// reproduced should be re-run with the device fill off (hostInputFillForced()).
///
/// Owns the rocRAND generator, created on the first fill that needs one, so its
/// lifetime is the owner's rather than the process's. Not thread-safe. Without rocRAND
/// it never accepts a fill, and every tensor is filled on the host.
class DeviceInputFiller
{
public:
    DeviceInputFiller();
    ~DeviceInputFiller();

    DeviceInputFiller(const DeviceInputFiller&) = delete;
    DeviceInputFiller& operator=(const DeviceInputFiller&) = delete;
    DeviceInputFiller(DeviceInputFiller&&) = delete;
    DeviceInputFiller& operator=(DeviceInputFiller&&) = delete;

    /// Fewest allocated elements a tensor needs to be filled on the device. Below this
    /// the fixed per-launch rocRAND cost exceeds the host loop.
    static constexpr std::size_t minElements()
    {
        return std::size_t{1} << 14;
    }

    /// Starts the fill of `tensor` on the device and returns true, or returns false and
    /// leaves the tensor untouched when the recipe is FIXED (a constant, cheaper on the host),
    /// when the tensor is too small to be worth it, or when it is of a type or rank the device
    /// fill does not cover. FREE covers float, half, bfloat16 and double; the other kinds
    /// cover float, half, bfloat16 and the four fp8 formats. A kind the tensor's data type
    /// cannot hold is also declined, so the host fill reports it as unsupported. Elements
    /// are written through the tensor's strides; the bytes of a padded tensor between its
    /// elements are left as they were. The fill is left in flight: nothing may read
    /// the tensor until waitForFills() returns. Throws DeviceInputError if the device
    /// work fails.
    bool tryFill(hipdnn_data_sdk::utilities::ITensor& tensor,
                 const FillRecipe& recipe,
                 unsigned int seed);

    /// Blocks until every fill started by tryFill() has finished. Throws
    /// DeviceInputError if one failed.
    void waitForFills();

    /// Whether this build can generate on the device at all, which is whether rocRAND
    /// was available when FillInputs.cpp was compiled.
    static bool isSupported();

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

/// Whether HIPDNN_TEST_HOST_INPUT_FILL asks for every input to be filled on the host: the
/// variable is set to anything but "0". The harness then withholds the DeviceInputFiller.
bool hostInputFillForced();

/// Fills every owned input of `graph`. With a `device` filler, the inputs it takes (see
/// DeviceInputFiller::tryFill) are generated on the device; with nullptr, which is what
/// most unit tests use since they have no device, everything is filled on the host. All
/// device fills have finished by the time this returns or throws. Throws DeviceInputError
/// when a device fill fails.
///
/// A VALUES recipe is always filled on the host. A recipe that cannot fill its tensor
/// as written makes the result invalid, with the uid and the reason.
///
/// After the fill, the O and stats inputs of each SDPA backward node are replaced by
/// the forward of its Q, K and V (see deriveSdpaBackwardForwardOutputs()), whatever
/// recipe they were filled with. Likewise the index input of each max-pooling resample
/// backward node is replaced by the argmax of a forward (see
/// deriveResampleBackwardIndex()). A node whose forward cannot be derived makes the
/// result unsupported, with the reason.
///
/// Each view in `aliases` keeps its place in the seed sequence, so no other tensor's
/// contents change, but it is not filled and need not be in `inputs`. Each owner has
/// its whole host allocation zeroed, then is filled once on the host. A recipe set()
/// on a view makes the result invalid. Throws std::invalid_argument when an owner is
/// missing from `ownedUids` or from `inputs`, or when fillDerivesInputs() is true for
/// the graph.
FillResult fillInputs(const hipdnn_flatbuffers_sdk::data_objects::Graph& graph,
                      InputTensorMap& inputs,
                      const std::vector<int64_t>& ownedUids,
                      InputFillRecipes& recipes,
                      DeviceInputFiller* device,
                      const std::vector<TensorAlias>& aliases = {});

/// Whether fillInputs() replaces some inputs of `graph` after the fill with values
/// derived from other inputs. Such a fill reads every input as its own tensor.
bool fillDerivesInputs(const hipdnn_flatbuffers_sdk::data_objects::Graph& graph);

} // namespace hipdnn_integration_tests
