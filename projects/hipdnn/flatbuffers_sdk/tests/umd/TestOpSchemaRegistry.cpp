// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// RFC 0018 Phase 0: proves the generated op-schema registry. Two claims:
//   1. Shape + optionality parity -- the registry lists SDPA's input/output tensor
//      names and scalar attributes with the optionality the schema `= null`
//      defaults imply (matching the generated header's Optional<T> fields).
//   2. Accessor value round-trip -- the generated typed readers, driven off a
//      live SdpaAttributes flatbuffer, return the exact stored values. Registry
//      shape alone is necessary but not sufficient; this proves the accessors
//      read the right field (the binding).

#include <cstdint>
#include <cstring>
#include <string_view>

#include <gtest/gtest.h>

#include <flatbuffers/flatbuffers.h>

#include <hipdnn_flatbuffers_sdk/umd/op_schema_registry_generated.hpp>

namespace fb = flatbuffers;
namespace umd = hipdnn_flatbuffers_sdk::umd;
namespace data = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

const umd::InputTensorBinding* findInputTensor(const umd::OpSchemaEntry* e, std::string_view name)
{
    for(std::size_t i = 0; i < e->inputTensorCount; ++i)
        if(e->inputTensors[i].name == name)
            return &e->inputTensors[i];
    return nullptr;
}

const umd::OutputTensorBinding* findOutputTensor(const umd::OpSchemaEntry* e, std::string_view name)
{
    for(std::size_t i = 0; i < e->outputTensorCount; ++i)
        if(e->outputTensors[i].name == name)
            return &e->outputTensors[i];
    return nullptr;
}

const umd::AttrBinding* findAttr(const umd::OpSchemaEntry* e, std::string_view name)
{
    for(std::size_t i = 0; i < e->attributeCount; ++i)
        if(e->attributes[i].name == name)
            return &e->attributes[i];
    return nullptr;
}

const umd::OpSchemaEntry* sdpaEntry()
{
    return umd::lookupOpByName("sdpa_fwd");
}

// A minimal SDPA attribute table with known values; optionals left unset stay
// absent. Owned by the returned builder; read via GetRoot on its buffer.
fb::FlatBufferBuilder buildSdpa()
{
    fb::FlatBufferBuilder b;
    data::SdpaAttributesBuilder sb(b);
    sb.add_q_tensor_uid(101);
    sb.add_k_tensor_uid(102);
    sb.add_v_tensor_uid(103);
    sb.add_o_tensor_uid(201);
    sb.add_dropout_probability(0.25f); // optional present
    sb.add_alibi_mask(true); // required bool
    // attn_mask / page_table_* / generate_stats left unset -> absent
    b.Finish(sb.Finish());
    return b;
}

} // namespace

TEST(UmdOpSchemaRegistry, SdpaEntryResolvesByNameAndType)
{
    const auto* byName = umd::lookupOpByName("sdpa_fwd"); // the umd_opcode shorthand
    const auto* byType = umd::lookupOpByType(data::NodeAttributes::SdpaAttributes);
    ASSERT_NE(byName, nullptr);
    ASSERT_NE(byType, nullptr);
    EXPECT_EQ(byName, byType);
    EXPECT_EQ(byName->opcode, "sdpa_fwd");
    EXPECT_EQ(byName->tableName, "SdpaAttributes");
    EXPECT_EQ(umd::lookupOpByName("SdpaAttributes"), nullptr); // table name is not the key
    EXPECT_EQ(umd::lookupOpByName("NoSuchOp"), nullptr);
}

TEST(UmdOpSchemaRegistry, SdpaRequiredOperandsAndResult)
{
    const auto* e = sdpaEntry();
    ASSERT_NE(e, nullptr);
    for(std::string_view name : {"q", "k", "v"})
    {
        const auto* op = findInputTensor(e, name);
        ASSERT_NE(op, nullptr) << name;
        EXPECT_FALSE(op->optional) << name;
    }
    const auto* o = findOutputTensor(e, "o");
    ASSERT_NE(o, nullptr);
    EXPECT_FALSE(o->optional);
}

TEST(UmdOpSchemaRegistry, SdpaOptionalOperandsClassifiedOptional)
{
    const auto* e = sdpaEntry();
    ASSERT_NE(e, nullptr);
    for(std::string_view name : {"attn_mask", "page_table_k", "page_table_v", "scale", "seed"})
    {
        const auto* op = findInputTensor(e, name);
        ASSERT_NE(op, nullptr) << name;
        EXPECT_TRUE(op->optional) << name;
    }
    // Optional results too.
    for(std::string_view name : {"stats", "amax_s", "amax_o"})
    {
        const auto* r = findOutputTensor(e, name);
        ASSERT_NE(r, nullptr) << name;
        EXPECT_TRUE(r->optional) << name;
    }
}

// Optionality parity with the schema `= null` defaults / generated Optional<T>.
TEST(UmdOpSchemaRegistry, SdpaScalarAttributeOptionalityParity)
{
    const auto* e = sdpaEntry();
    ASSERT_NE(e, nullptr);

    // `= null` scalars are optional.
    for(std::string_view name : {"generate_stats",
                                 "dropout_probability",
                                 "attn_scale_value",
                                 "left_bound",
                                 "right_bound",
                                 "max_seq_len_kv"})
    {
        const auto* a = findAttr(e, name);
        ASSERT_NE(a, nullptr) << name;
        EXPECT_TRUE(a->optional) << name;
    }
    // Defaulted (non-null) scalars/enums are required.
    for(std::string_view name : {"alibi_mask",
                                 "padding_mask",
                                 "causal_mask",
                                 "causal_mask_bottom_right",
                                 "diagonal_alignment"})
    {
        const auto* a = findAttr(e, name);
        ASSERT_NE(a, nullptr) << name;
        EXPECT_FALSE(a->optional) << name;
    }
}

TEST(UmdOpSchemaRegistry, SdpaScalarAttributeTypes)
{
    const auto* e = sdpaEntry();
    ASSERT_NE(e, nullptr);
    EXPECT_EQ(findAttr(e, "alibi_mask")->type, umd::AttrType::Bool);
    EXPECT_EQ(findAttr(e, "dropout_probability")->type, umd::AttrType::Float);
    EXPECT_EQ(findAttr(e, "left_bound")->type, umd::AttrType::Int);
    EXPECT_EQ(findAttr(e, "diagonal_alignment")->type, umd::AttrType::Dtype);
}

TEST(UmdOpSchemaRegistry, AccessorRoundTripReadsLiveValues)
{
    const auto* e = sdpaEntry();
    ASSERT_NE(e, nullptr);
    auto b = buildSdpa();
    const auto* attrs = fb::GetRoot<data::SdpaAttributes>(b.GetBufferPointer());
    const void* a = attrs;

    std::int64_t uid = 0;
    ASSERT_TRUE(findInputTensor(e, "q")->read(a, uid));
    EXPECT_EQ(uid, 101);
    ASSERT_TRUE(findInputTensor(e, "k")->read(a, uid));
    EXPECT_EQ(uid, 102);
    ASSERT_TRUE(findInputTensor(e, "v")->read(a, uid));
    EXPECT_EQ(uid, 103);
    ASSERT_TRUE(findOutputTensor(e, "o")->read(a, uid));
    EXPECT_EQ(uid, 201);
}

TEST(UmdOpSchemaRegistry, AbsentOptionalOperandReadsFalse)
{
    const auto* e = sdpaEntry();
    ASSERT_NE(e, nullptr);
    auto b = buildSdpa();
    const auto* attrs = fb::GetRoot<data::SdpaAttributes>(b.GetBufferPointer());
    const void* a = attrs;

    std::int64_t uid = -1;
    EXPECT_FALSE(findInputTensor(e, "attn_mask")->read(a, uid));
    EXPECT_FALSE(findInputTensor(e, "page_table_k")->read(a, uid));
}

TEST(UmdOpSchemaRegistry, ScalarReadersReflectPresenceAndValue)
{
    const auto* e = sdpaEntry();
    ASSERT_NE(e, nullptr);
    auto b = buildSdpa();
    const auto* attrs = fb::GetRoot<data::SdpaAttributes>(b.GetBufferPointer());
    const void* a = attrs;

    const umd::ScalarValue alibi = findAttr(e, "alibi_mask")->read(a);
    EXPECT_EQ(alibi.type, umd::AttrType::Bool);
    EXPECT_TRUE(alibi.present);
    EXPECT_TRUE(alibi.b);

    const umd::ScalarValue drop = findAttr(e, "dropout_probability")->read(a);
    EXPECT_EQ(drop.type, umd::AttrType::Float);
    EXPECT_TRUE(drop.present);
    EXPECT_DOUBLE_EQ(drop.f, 0.25);

    const umd::ScalarValue stats = findAttr(e, "generate_stats")->read(a);
    EXPECT_EQ(stats.type, umd::AttrType::Bool);
    EXPECT_FALSE(stats.present); // optional, unset

    const umd::ScalarValue align = findAttr(e, "diagonal_alignment")->read(a);
    EXPECT_EQ(align.type, umd::AttrType::Dtype);
    EXPECT_TRUE(align.present);
    ASSERT_NE(align.dtype, nullptr);
    EXPECT_STREQ(align.dtype, "TOP_LEFT");
}
