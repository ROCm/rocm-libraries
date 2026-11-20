// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include "tests/engines/kernel_ingestor_engine/packs/UnpulledModelsRecord.hpp"

/**
 * @file TestUnpulledModelsRecord.cpp
 * @brief The reader of hkp_pack's skip record (hkp-unpulled-models.record), which the
 *        model-binding census and the GPU cold cases consult before accepting a missing
 *        trained model: only a record that names the model's UHD id excuses it, and a missing,
 *        malformed or differently shaped record excuses nothing.
 */
namespace
{

using hip_kernel_provider::kernel_ingestor_engine::testing::UNPULLED_RECORD;
using hip_kernel_provider::kernel_ingestor_engine::testing::unpulledRecordNames;

constexpr const char* RANKER = "18bb432e-d871-4290-9ae8-f5111aaf2ffe";

class TestUnpulledModelsRecord : public ::testing::Test
{
protected:
    void SetUp() override
    {
        const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
        _root = std::filesystem::temp_directory_path()
                / (std::string("hkp_unpulled_record_") + info->name());
        std::filesystem::remove_all(_root);
        std::filesystem::create_directories(_root);
    }

    void TearDown() override
    {
        std::error_code error;
        std::filesystem::remove_all(_root, error);
    }

    void writeRecord(const std::string& text) const
    {
        std::ofstream(_root / UNPULLED_RECORD) << text;
    }

    std::filesystem::path _root;
};

TEST_F(TestUnpulledModelsRecord, NamesTheModelsTheRecordLists)
{
    writeRecord(std::string(R"({"unpulled_models": [{"uhd_id": ")") + RANKER + R"("}]})");
    EXPECT_TRUE(unpulledRecordNames(_root, RANKER));

    writeRecord(std::string(R"({"unpulled_models": [{"uhd_id": "other"}, {"uhd_id": ")") + RANKER
                + R"("}]})");
    EXPECT_TRUE(unpulledRecordNames(_root, RANKER));
    EXPECT_TRUE(unpulledRecordNames(_root, "other"));
}

TEST_F(TestUnpulledModelsRecord, DoesNotNameAModelTheRecordOmits)
{
    writeRecord(R"({"unpulled_models": [{"uhd_id": "other"}]})");
    EXPECT_FALSE(unpulledRecordNames(_root, RANKER));
}

TEST_F(TestUnpulledModelsRecord, NoRecordNamesNothing)
{
    EXPECT_FALSE(unpulledRecordNames(_root, RANKER));
}

TEST_F(TestUnpulledModelsRecord, AMalformedOrDifferentlyShapedRecordNamesNothing)
{
    for(const char* text :
        {"not json",
         R"([])",
         R"({})",
         R"({"unpulled_models": "18bb432e-d871-4290-9ae8-f5111aaf2ffe"})",
         R"({"unpulled_models": ["18bb432e-d871-4290-9ae8-f5111aaf2ffe"]})",
         R"({"unpulled_models": [{"id": "18bb432e-d871-4290-9ae8-f5111aaf2ffe"}]})"})
    {
        writeRecord(text);
        EXPECT_FALSE(unpulledRecordNames(_root, RANKER)) << "record: " << text;
    }
}

} // namespace
