// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <memory>
#include <unordered_map>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/ApplicabilityUtils.hpp>

using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_flatbuffers_sdk::utilities;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

namespace
{

using TensorMap = std::unordered_map<int64_t, const TensorAttributes*>;

// Builds standalone TensorAttributes FlatBuffers and exposes them as the
// const-pointer map consumed by the applicability helpers. Each tensor is
// serialized into its own buffer which is kept alive for the lifetime of the
// builder so the pointers stored in the map remain valid.
class TensorMapBuilder
{
public:
    // Adds a non-ragged (dense) tensor with the given uid.
    void addDense(int64_t uid)
    {
        addImpl(uid, flatbuffers::nullopt);
    }

    // Adds a ragged tensor with the given uid, pointing at raggedOffsetUid.
    void addRagged(int64_t uid, int64_t raggedOffsetUid)
    {
        addImpl(uid, flatbuffers::Optional<int64_t>(raggedOffsetUid));
    }

    const TensorMap& map() const
    {
        return _map;
    }

private:
    void addImpl(int64_t uid, flatbuffers::Optional<int64_t> raggedOffset)
    {
        auto fbb = std::make_unique<flatbuffers::FlatBufferBuilder>();

        const std::vector<int64_t> dims = {2, 3};
        const std::vector<int64_t> strides = {3, 1};

        auto offset = CreateTensorAttributesDirect(*fbb,
                                                   uid,
                                                   "t",
                                                   DataType::FLOAT,
                                                   &strides,
                                                   &dims,
                                                   /*virtual_=*/false,
                                                   TensorValue::NONE,
                                                   /*value=*/0,
                                                   false,
                                                   raggedOffset);
        fbb->Finish(offset);

        _map[uid] = flatbuffers::GetRoot<TensorAttributes>(fbb->GetBufferPointer());
        _builders.push_back(std::move(fbb));
    }

    std::vector<std::unique_ptr<flatbuffers::FlatBufferBuilder>> _builders;
    TensorMap _map;
};

} // namespace

TEST(TestApplicabilityUtils, HasNoRaggedTensorIdsReturnsTrueForEmptyMap)
{
    const TensorMap emptyMap;
    EXPECT_TRUE(hasNoRaggedTensorIds(emptyMap));
}

TEST(TestApplicabilityUtils, HasNoRaggedTensorIdsReturnsTrueWhenNoTensorIsRagged)
{
    TensorMapBuilder builder;
    builder.addDense(1);
    builder.addDense(2);
    builder.addDense(3);

    EXPECT_TRUE(hasNoRaggedTensorIds(builder.map()));
}

TEST(TestApplicabilityUtils, HasNoRaggedTensorIdsReturnsFalseWhenAnyTensorIsRagged)
{
    TensorMapBuilder builder;
    builder.addDense(1);
    builder.addRagged(2, /*raggedOffsetUid=*/3);
    builder.addDense(3);

    EXPECT_FALSE(hasNoRaggedTensorIds(builder.map()));
}

TEST(TestApplicabilityUtils, ListUnsupportedRaggedTensorIdsReturnsEmptyWhenNoTensorIsRagged)
{
    TensorMapBuilder builder;
    builder.addDense(1);
    builder.addDense(2);

    EXPECT_THAT(listUnsupportedRaggedTensorIds(builder.map()), IsEmpty());
}

TEST(TestApplicabilityUtils, ListUnsupportedRaggedTensorIdsReturnsAllRaggedWhenNoneSupported)
{
    TensorMapBuilder builder;
    builder.addRagged(1, /*raggedOffsetUid=*/10);
    builder.addDense(2);
    builder.addRagged(3, /*raggedOffsetUid=*/11);

    // With an empty supported list every ragged tensor is unsupported.
    EXPECT_THAT(listUnsupportedRaggedTensorIds(builder.map()), UnorderedElementsAre(1, 3));
}

TEST(TestApplicabilityUtils, ListUnsupportedRaggedTensorIdsExcludesSupportedRaggedTensors)
{
    TensorMapBuilder builder;
    builder.addRagged(1, /*raggedOffsetUid=*/10);
    builder.addRagged(2, /*raggedOffsetUid=*/11);

    // Both tensors are ragged but both are declared supported.
    EXPECT_THAT(listUnsupportedRaggedTensorIds(builder.map(), /*supportedRaggedIds=*/{1, 2}),
                IsEmpty());
}

TEST(TestApplicabilityUtils, ListUnsupportedRaggedTensorIdsReturnsOnlyUnsupportedRaggedTensors)
{
    TensorMapBuilder builder;
    builder.addRagged(1, /*raggedOffsetUid=*/10); // ragged + supported
    builder.addRagged(2, /*raggedOffsetUid=*/11); // ragged + unsupported
    builder.addDense(3); // dense
    builder.addRagged(4, /*raggedOffsetUid=*/12); // ragged + unsupported

    EXPECT_THAT(listUnsupportedRaggedTensorIds(builder.map(), /*supportedRaggedIds=*/{1}),
                UnorderedElementsAre(2, 4));
}

TEST(TestApplicabilityUtils, ListUnsupportedRaggedTensorIdsIgnoresSupportedIdsThatAreNotRagged)
{
    TensorMapBuilder builder;
    builder.addDense(1); // dense, listed as supported
    builder.addRagged(2, /*raggedOffsetUid=*/11); // ragged + unsupported

    // Supported ids that are dense (1) or absent (99) must not affect the result.
    EXPECT_THAT(listUnsupportedRaggedTensorIds(builder.map(), /*supportedRaggedIds=*/{1, 99}),
                UnorderedElementsAre(2));
}
