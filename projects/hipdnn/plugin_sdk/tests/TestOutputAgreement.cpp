// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/types/Bfloat16.hpp>
#include <hipdnn_data_sdk/types/Fp4E2M1.hpp>
#include <hipdnn_data_sdk/types/Fp6E2M3.hpp>
#include <hipdnn_data_sdk/types/Fp6E3M2.hpp>
#include <hipdnn_data_sdk/types/Fp8E4M3.hpp>
#include <hipdnn_data_sdk/types/Fp8E4M3Fnuz.hpp>
#include <hipdnn_data_sdk/types/Fp8E5M2.hpp>
#include <hipdnn_data_sdk/types/Fp8E5M2Fnuz.hpp>
#include <hipdnn_data_sdk/types/Fp8E8M0.hpp>
#include <hipdnn_data_sdk/types/Half.hpp>
#include <hipdnn_plugin_sdk/OutputAgreement.hpp>

/**
 * @file TestOutputAgreement.cpp
 * @brief Unit tests for OutputAgreement.hpp: the output fill rule and the element compare.
 *
 * Not guarded on HIPDNN_ENABLE_KERNEL_INGESTOR: the header under test is not.
 */
namespace
{

using namespace hipdnn_plugin_sdk::output_agreement;
namespace types = hipdnn_data_sdk::types;

constexpr int FIRST_DATA_TYPE = static_cast<int>(DataType::MIN);
constexpr int LAST_DATA_TYPE = static_cast<int>(DataType::MAX);

std::vector<DataType> allDataTypes()
{
    std::vector<DataType> result;
    for(int value = FIRST_DATA_TYPE; value <= LAST_DATA_TYPE; ++value)
    {
        result.push_back(static_cast<DataType>(value));
    }
    return result;
}

std::string typeName(DataType dataType)
{
    return hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(dataType);
}

/// numeric_limits<T>::has_quiet_NaN of the data_sdk type that stores @p dataType; false for
/// UNSET and for INT4, which has no element type.
bool typeHasQuietNaN(DataType dataType)
{
    switch(dataType)
    {
    case DataType::FLOAT:
        return std::numeric_limits<float>::has_quiet_NaN;
    case DataType::DOUBLE:
        return std::numeric_limits<double>::has_quiet_NaN;
    case DataType::HALF:
        return std::numeric_limits<types::half>::has_quiet_NaN;
    case DataType::BFLOAT16:
        return std::numeric_limits<types::bfloat16>::has_quiet_NaN;
    case DataType::FP8_E4M3:
        return std::numeric_limits<types::fp8_e4m3>::has_quiet_NaN;
    case DataType::FP8_E5M2:
        return std::numeric_limits<types::fp8_e5m2>::has_quiet_NaN;
    case DataType::FP8_E8M0:
        return std::numeric_limits<types::fp8_e8m0>::has_quiet_NaN;
    case DataType::FP8_E4M3_FNUZ:
        return std::numeric_limits<types::fp8_e4m3_fnuz>::has_quiet_NaN;
    case DataType::FP8_E5M2_FNUZ:
        return std::numeric_limits<types::fp8_e5m2_fnuz>::has_quiet_NaN;
    case DataType::FP4_E2M1:
        return std::numeric_limits<types::fp4_e2m1>::has_quiet_NaN;
    case DataType::FP6_E2M3:
        return std::numeric_limits<types::fp6_e2m3>::has_quiet_NaN;
    case DataType::FP6_E3M2:
        return std::numeric_limits<types::fp6_e3m2>::has_quiet_NaN;
    case DataType::UINT8:
        return std::numeric_limits<uint8_t>::has_quiet_NaN;
    case DataType::INT8:
        return std::numeric_limits<int8_t>::has_quiet_NaN;
    case DataType::INT32:
        return std::numeric_limits<int32_t>::has_quiet_NaN;
    case DataType::INT64:
        return std::numeric_limits<int64_t>::has_quiet_NaN;
    case DataType::BOOLEAN:
        return std::numeric_limits<bool>::has_quiet_NaN;
    default:
        return false;
    }
}

/// Builds a T from the bytes of fillByte(@p dataType) and asks T's own isnan.
template <typename T>
bool fillBytesDecodeToNaN(DataType dataType)
{
    std::array<uint8_t, sizeof(T)> bytes;
    std::memset(bytes.data(), fillByte(dataType), bytes.size());
    T value;
    std::memcpy(static_cast<void*>(&value), bytes.data(), sizeof(T));
    using std::isnan;
    return isnan(value);
}

bool fillDecodesToNaNByTypesOwnIsnan(DataType dataType)
{
    switch(dataType)
    {
    case DataType::FLOAT:
        return fillBytesDecodeToNaN<float>(dataType);
    case DataType::DOUBLE:
        return fillBytesDecodeToNaN<double>(dataType);
    case DataType::HALF:
        return fillBytesDecodeToNaN<types::half>(dataType);
    case DataType::BFLOAT16:
        return fillBytesDecodeToNaN<types::bfloat16>(dataType);
    case DataType::FP8_E4M3:
        return fillBytesDecodeToNaN<types::fp8_e4m3>(dataType);
    case DataType::FP8_E5M2:
        return fillBytesDecodeToNaN<types::fp8_e5m2>(dataType);
    case DataType::FP8_E8M0:
        return fillBytesDecodeToNaN<types::fp8_e8m0>(dataType);
    case DataType::FP8_E4M3_FNUZ:
        return fillBytesDecodeToNaN<types::fp8_e4m3_fnuz>(dataType);
    case DataType::FP8_E5M2_FNUZ:
        return fillBytesDecodeToNaN<types::fp8_e5m2_fnuz>(dataType);
    default:
        ADD_FAILURE() << typeName(dataType) << " has no NaN type in this table";
        return false;
    }
}

template <typename T>
std::vector<uint8_t> imageOf(const std::vector<T>& values)
{
    std::vector<uint8_t> image(values.size() * sizeof(T));
    std::memcpy(image.data(), values.data(), image.size());
    return image;
}

std::vector<uint8_t> halfImage(const std::vector<float>& values)
{
    std::vector<types::half> halves;
    halves.reserve(values.size());
    for(const float value : values)
    {
        halves.emplace_back(value);
    }
    return imageOf(halves);
}

std::vector<uint8_t> bfloat16Image(const std::vector<float>& values)
{
    std::vector<types::bfloat16> halves;
    halves.reserve(values.size());
    for(const float value : values)
    {
        halves.emplace_back(value);
    }
    return imageOf(halves);
}

constexpr float FLOAT_NAN = std::numeric_limits<float>::quiet_NaN();
constexpr float FLOAT_INF = std::numeric_limits<float>::infinity();

} // namespace

// ---- fill rule -------------------------------------------------------------------------

TEST(TestOutputAgreement, FillByteIs0x80ForTheFnuzTypesAnd0xFFForEveryOtherType)
{
    for(const DataType dataType : allDataTypes())
    {
        const bool fnuz
            = dataType == DataType::FP8_E4M3_FNUZ || dataType == DataType::FP8_E5M2_FNUZ;
        EXPECT_EQ(fillByte(dataType), fnuz ? 0x80 : 0xFF) << typeName(dataType);
    }
}

TEST(TestOutputAgreement, HasNaNEncodingMatchesTheDataSdkTypeForEveryDataType)
{
    for(const DataType dataType : allDataTypes())
    {
        EXPECT_EQ(hasNaNEncoding(dataType), typeHasQuietNaN(dataType)) << typeName(dataType);
    }
}

TEST(TestOutputAgreement, TheFillBytesOfEveryNaNCapableTypeDecodeToNaNByThatTypesIsnan)
{
    int checked = 0;
    for(const DataType dataType : allDataTypes())
    {
        if(!hasNaNEncoding(dataType))
        {
            continue;
        }
        EXPECT_TRUE(fillDecodesToNaNByTypesOwnIsnan(dataType)) << typeName(dataType);
        ++checked;
    }
    // float, double, half, bfloat16 and the five FP8 types
    EXPECT_EQ(checked, 9);
}

TEST(TestOutputAgreement, AnImageOfOnlyFillBytesIsUntouchedForANaNCapableType)
{
    int checked = 0;
    for(const DataType dataType : allDataTypes())
    {
        if(!hasNaNEncoding(dataType))
        {
            continue;
        }
        const std::vector<uint8_t> image(64, fillByte(dataType));
        EXPECT_TRUE(imageUntouched(dataType, image)) << typeName(dataType);
        ++checked;
    }
    EXPECT_EQ(checked, 9);
}

TEST(TestOutputAgreement, OneDifferingByteMakesTheImageTouchedAtAnyPosition)
{
    for(const DataType dataType : {DataType::FLOAT, DataType::FP8_E4M3_FNUZ})
    {
        for(const size_t position : {size_t{0}, size_t{31}, size_t{63}})
        {
            std::vector<uint8_t> image(64, fillByte(dataType));
            image[position] = static_cast<uint8_t>(image[position] ^ 0x01U);
            EXPECT_FALSE(imageUntouched(dataType, image)) << typeName(dataType) << " " << position;
        }
    }
}

TEST(TestOutputAgreement, AnEmptyImageIsNeverUntouched)
{
    EXPECT_FALSE(imageUntouched(DataType::FLOAT, {}));
    EXPECT_FALSE(imageUntouched(DataType::FP8_E4M3_FNUZ, {}));
}

TEST(TestOutputAgreement, ATypeWithoutANaNEncodingIsNeverUntouched)
{
    for(const DataType dataType : allDataTypes())
    {
        if(hasNaNEncoding(dataType))
        {
            continue;
        }
        const std::vector<uint8_t> image(64, fillByte(dataType));
        EXPECT_FALSE(imageUntouched(dataType, image)) << typeName(dataType);
    }
}

// ---- tables ----------------------------------------------------------------------------

TEST(TestOutputAgreement, NumericKindAndElementBytesCoverTheDecodableTypesOnly)
{
    struct Row
    {
        DataType dataType;
        NumericKind kind;
        size_t bytes;
    };
    const std::array<Row, 19> expected{{
        {DataType::DOUBLE, NumericKind::FLOAT64, 8},
        {DataType::FLOAT, NumericKind::FLOAT32, 4},
        {DataType::HALF, NumericKind::FLOAT16, 2},
        {DataType::BFLOAT16, NumericKind::BFLOAT16, 2},
        {DataType::INT8, NumericKind::SIGNED_INTEGER, 1},
        {DataType::INT32, NumericKind::SIGNED_INTEGER, 4},
        {DataType::INT64, NumericKind::SIGNED_INTEGER, 8},
        {DataType::UINT8, NumericKind::UNSIGNED_INTEGER, 1},
        {DataType::BOOLEAN, NumericKind::UNSIGNED_INTEGER, 1},
        {DataType::UNSET, NumericKind::NONE, 0},
        {DataType::FP8_E4M3, NumericKind::NONE, 0},
        {DataType::FP8_E5M2, NumericKind::NONE, 0},
        {DataType::FP8_E8M0, NumericKind::NONE, 0},
        {DataType::FP8_E4M3_FNUZ, NumericKind::NONE, 0},
        {DataType::FP8_E5M2_FNUZ, NumericKind::NONE, 0},
        {DataType::FP4_E2M1, NumericKind::NONE, 0},
        {DataType::FP6_E2M3, NumericKind::NONE, 0},
        {DataType::FP6_E3M2, NumericKind::NONE, 0},
        {DataType::INT4, NumericKind::NONE, 0},
    }};
    for(const auto& row : expected)
    {
        EXPECT_EQ(numericKind(row.dataType), row.kind) << typeName(row.dataType);
        EXPECT_EQ(elementBytes(row.dataType), row.bytes) << typeName(row.dataType);
    }
}

TEST(TestOutputAgreement, ToleranceAndFloorTables)
{
    EXPECT_EQ(agreementTolerance(NumericKind::FLOAT64), 1e-12);
    EXPECT_EQ(agreementTolerance(NumericKind::FLOAT32), 1e-5);
    EXPECT_EQ(agreementTolerance(NumericKind::FLOAT16), 2e-2);
    EXPECT_EQ(agreementTolerance(NumericKind::BFLOAT16), 6e-2);
    EXPECT_EQ(agreementTolerance(NumericKind::SIGNED_INTEGER), 0.0);
    EXPECT_EQ(agreementTolerance(NumericKind::UNSIGNED_INTEGER), 0.0);
    EXPECT_EQ(agreementTolerance(NumericKind::NONE), 0.0);

    EXPECT_EQ(agreementFloor(NumericKind::FLOAT64), std::ldexp(1.0, -52));
    EXPECT_EQ(agreementFloor(NumericKind::FLOAT32), std::ldexp(1.0, -23));
    EXPECT_EQ(agreementFloor(NumericKind::FLOAT16), std::ldexp(1.0, -10));
    EXPECT_EQ(agreementFloor(NumericKind::BFLOAT16), std::ldexp(1.0, -7));
    EXPECT_EQ(agreementFloor(NumericKind::SIGNED_INTEGER), 0.0);
    EXPECT_EQ(agreementFloor(NumericKind::UNSIGNED_INTEGER), 0.0);
    EXPECT_EQ(agreementFloor(NumericKind::NONE), 0.0);
}

TEST(TestOutputAgreement, DecodeElementReadsEachStorageType)
{
    EXPECT_EQ(decodeElement(DataType::DOUBLE, imageOf<double>({1.0, -2.5}), 1), -2.5);
    EXPECT_EQ(decodeElement(DataType::FLOAT, imageOf<float>({1.0F, 3.25F}), 1), 3.25);
    EXPECT_EQ(decodeElement(DataType::HALF, halfImage({1.0F, 0.5F}), 1), 0.5);
    EXPECT_EQ(decodeElement(DataType::BFLOAT16, bfloat16Image({1.0F, -4.0F}), 1), -4.0);
    EXPECT_EQ(decodeElement(DataType::INT64, imageOf<int64_t>({1, -7}), 1), -7.0);
    EXPECT_EQ(decodeElement(DataType::INT32, imageOf<int32_t>({1, -70000}), 1), -70000.0);
    EXPECT_EQ(decodeElement(DataType::INT8, imageOf<int8_t>({1, -3}), 1), -3.0);
    EXPECT_EQ(decodeElement(DataType::UINT8, imageOf<uint8_t>({1, 200}), 1), 200.0);
    EXPECT_EQ(decodeElement(DataType::BOOLEAN, imageOf<uint8_t>({0, 1}), 1), 1.0);
}

// ---- firstDisagreement -----------------------------------------------------------------

TEST(TestOutputAgreement, Float32AgreesWithinTheToleranceAndDisagreesOutsideIt)
{
    const auto base = imageOf<float>({100.0F, 1.0F});
    const auto within = imageOf<float>({100.0F + 5e-4F, 1.0F});
    const auto outside = imageOf<float>({100.0F + 5e-2F, 1.0F});

    EXPECT_EQ(firstDisagreement("o", DataType::FLOAT, base, within), "");
    EXPECT_NE(firstDisagreement("o", DataType::FLOAT, base, outside), "");
}

TEST(TestOutputAgreement, TheMessageNamesTheOutputTheFirstIndexAndBothValues)
{
    const auto left = imageOf<float>({1.0F, 2.0F, 3.0F});
    const auto right = imageOf<float>({1.0F, 2.0F, 4.0F});

    const std::string message = firstDisagreement("dy", DataType::FLOAT, left, right);

    EXPECT_EQ(message.rfind("dy[2]: ", 0), 0U) << message;
    EXPECT_NE(message.find('3'), std::string::npos) << message;
    EXPECT_NE(message.find('4'), std::string::npos) << message;
    EXPECT_NE(message.find("(tolerance "), std::string::npos) << message;
}

TEST(TestOutputAgreement, TheFirstDisagreementWinsWhenSeveralElementsDiffer)
{
    const auto left = imageOf<float>({1.0F, 2.0F, 3.0F});
    const auto right = imageOf<float>({1.0F, 9.0F, 8.0F});

    EXPECT_EQ(firstDisagreement("o", DataType::FLOAT, left, right).rfind("o[1]: ", 0), 0U);
}

TEST(TestOutputAgreement, FirstDisagreementAtReportsTheIndexTheValuesAndTheTolerance)
{
    const auto left = imageOf<float>({1.0F, 2.0F, 3.0F});
    const auto right = imageOf<float>({1.0F, 2.0F, 4.0F});

    const auto found = firstDisagreementAt(DataType::FLOAT, left, right);

    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(found->index, 2U);
    EXPECT_EQ(found->left, 3.0);
    EXPECT_EQ(found->right, 4.0);
    // 1e-5 * max(3, 4) + 2^-23 * 4: the relative term plus one ulp at the images' scale.
    EXPECT_DOUBLE_EQ(found->tolerance, (1e-5 * 4.0) + (0x1p-23 * 4.0));
}

TEST(TestOutputAgreement, FirstDisagreementAtIsEmptyWhenTheImagesAgreeOrCannotBeCompared)
{
    const auto base = imageOf<float>({100.0F, 1.0F});
    EXPECT_FALSE(firstDisagreementAt(DataType::FLOAT, base, base).has_value());
    EXPECT_FALSE(firstDisagreementAt(DataType::FLOAT, base, imageOf<float>({100.0F + 5e-4F, 1.0F}))
                     .has_value());
    EXPECT_FALSE(firstDisagreementAt(DataType::FLOAT, base, imageOf<float>({100.0F})).has_value());
    EXPECT_FALSE(firstDisagreementAt(
                     DataType::FP8_E4M3, std::vector<uint8_t>{1, 2}, std::vector<uint8_t>{3, 4})
                     .has_value());
}

TEST(TestOutputAgreement, FirstDisagreementAtReportsANonFiniteElementAgainstAFiniteOne)
{
    const auto found = firstDisagreementAt(
        DataType::FLOAT, imageOf<float>({1.0F, FLOAT_NAN}), imageOf<float>({1.0F, 2.0F}));

    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(found->index, 1U);
    EXPECT_TRUE(std::isnan(found->left));
    EXPECT_EQ(found->right, 2.0);
}

TEST(TestOutputAgreement, Float64UsesATighterToleranceThanFloat32)
{
    const auto base = imageOf<double>({1.0});
    const auto nearby = imageOf<double>({1.0 + 1e-9});

    EXPECT_NE(firstDisagreement("o", DataType::DOUBLE, base, nearby), "");
    EXPECT_EQ(firstDisagreement(
                  "o", DataType::FLOAT, imageOf<float>({1.0F}), imageOf<float>({1.0F + 1e-7F})),
              "");
}

TEST(TestOutputAgreement, Float16AndBfloat16UseTheirOwnTolerances)
{
    // 1.0 vs 1.01: inside half's 2e-2, outside float's 1e-5; 1.0 vs 1.05 is outside half's.
    EXPECT_EQ(firstDisagreement("o", DataType::HALF, halfImage({1.0F}), halfImage({1.01F})), "");
    EXPECT_NE(firstDisagreement("o", DataType::HALF, halfImage({1.0F}), halfImage({1.05F})), "");
    // bfloat16: 1.0 vs 1.03 inside 6e-2, 1.0 vs 1.125 outside.
    EXPECT_EQ(
        firstDisagreement("o", DataType::BFLOAT16, bfloat16Image({1.0F}), bfloat16Image({1.03F})),
        "");
    EXPECT_NE(
        firstDisagreement("o", DataType::BFLOAT16, bfloat16Image({1.0F}), bfloat16Image({1.125F})),
        "");
}

TEST(TestOutputAgreement, IntegerKindsAreExact)
{
    EXPECT_EQ(
        firstDisagreement("o", DataType::INT32, imageOf<int32_t>({5, 6}), imageOf<int32_t>({5, 6})),
        "");
    EXPECT_NE(
        firstDisagreement("o", DataType::INT32, imageOf<int32_t>({5, 6}), imageOf<int32_t>({5, 7})),
        "");
    EXPECT_NE(firstDisagreement("o", DataType::INT8, imageOf<int8_t>({-1}), imageOf<int8_t>({1})),
              "");
    EXPECT_NE(
        firstDisagreement("o", DataType::UINT8, imageOf<uint8_t>({0}), imageOf<uint8_t>({255})),
        "");
    EXPECT_NE(
        firstDisagreement("o", DataType::BOOLEAN, imageOf<uint8_t>({0}), imageOf<uint8_t>({1})),
        "");
    EXPECT_NE(firstDisagreement("o", DataType::INT64, imageOf<int64_t>({1}), imageOf<int64_t>({2})),
              "");
}

TEST(TestOutputAgreement, ImagesOfDifferentSizeAreNotCompared)
{
    EXPECT_EQ(firstDisagreement(
                  "o", DataType::FLOAT, imageOf<float>({1.0F}), imageOf<float>({1.0F, 2.0F})),
              "");
}

TEST(TestOutputAgreement, ADtypeWithoutADecodeIsNeverCompared)
{
    const std::vector<uint8_t> left(4, 0x01);
    const std::vector<uint8_t> right(4, 0x02);

    EXPECT_EQ(firstDisagreement("o", DataType::FP8_E4M3, left, right), "");
    EXPECT_EQ(firstDisagreement("o", DataType::INT4, left, right), "");
    EXPECT_EQ(firstDisagreement("o", DataType::UNSET, left, right), "");
}

TEST(TestOutputAgreement, TheScaleCoversBothImages)
{
    // At index 0 the values differ by 1e-6 against a magnitude of 1e-3: far outside the
    // relative term alone (1e-5 * 1e-3 = 1e-8). The absolute term is 2^-23 * scale, so the
    // verdict depends on the largest finite magnitude over BOTH images. A single image's
    // scale (1e-3) gives 1.2e-10 and disagrees; the pair's scale (1e3) gives 1.2e-4 and agrees.
    const auto left = imageOf<float>({1e-3F, 1e-3F});
    const auto right = imageOf<float>({1e-3F + 1e-6F, 1e3F});
    const auto rightSmall = imageOf<float>({1e-3F + 1e-6F, 1e-3F});

    EXPECT_EQ(firstDisagreement("o", DataType::FLOAT, left, right).rfind("o[1]: ", 0), 0U)
        << "index 0 must agree under the pair's scale; index 1 is the real difference";
    EXPECT_EQ(firstDisagreement("o", DataType::FLOAT, left, rightSmall).rfind("o[0]: ", 0), 0U);
}

TEST(TestOutputAgreement, TheScaleIgnoresNonFiniteElements)
{
    // An infinity must not widen the absolute term to infinity and swallow a real difference.
    const auto left = imageOf<float>({FLOAT_INF, 1.0F});
    const auto right = imageOf<float>({FLOAT_INF, 2.0F});

    EXPECT_EQ(firstDisagreement("o", DataType::FLOAT, left, right).rfind("o[1]: ", 0), 0U);
}

TEST(TestOutputAgreement, TheSameNonFiniteKindAtTheSamePositionAgrees)
{
    const auto left = imageOf<float>({FLOAT_NAN, FLOAT_INF, -FLOAT_INF, 1.0F});
    const auto right = imageOf<float>({FLOAT_NAN, FLOAT_INF, -FLOAT_INF, 1.0F});
    EXPECT_EQ(firstDisagreement("o", DataType::FLOAT, left, right), "");

    // Not byte-identical, so the byte short-circuit does not apply: a NaN with another payload.
    auto payload = left;
    payload[0] = static_cast<uint8_t>(payload[0] ^ 0x01U);
    EXPECT_EQ(firstDisagreement("o", DataType::FLOAT, left, payload), "");
}

TEST(TestOutputAgreement, NaNAgainstFiniteDisagreesInBothDirections)
{
    const auto nan = imageOf<float>({FLOAT_NAN});
    const auto finite = imageOf<float>({1.0F});

    EXPECT_NE(firstDisagreement("o", DataType::FLOAT, nan, finite), "");
    EXPECT_NE(firstDisagreement("o", DataType::FLOAT, finite, nan), "");
}

TEST(TestOutputAgreement, InfinityAgainstFiniteOrAgainstTheOppositeInfinityDisagrees)
{
    EXPECT_NE(firstDisagreement(
                  "o", DataType::FLOAT, imageOf<float>({FLOAT_INF}), imageOf<float>({1.0F})),
              "");
    EXPECT_NE(firstDisagreement(
                  "o", DataType::FLOAT, imageOf<float>({FLOAT_INF}), imageOf<float>({-FLOAT_INF})),
              "");
    EXPECT_NE(firstDisagreement(
                  "o", DataType::FLOAT, imageOf<float>({FLOAT_NAN}), imageOf<float>({FLOAT_INF})),
              "");
}

// ---- firstNonFinite --------------------------------------------------------------------

TEST(TestOutputAgreement, FirstNonFiniteIsEmptyWhenEveryElementIsFinite)
{
    EXPECT_EQ(firstNonFinite("o", DataType::FLOAT, imageOf<float>({1.0F, -2.0F, 0.0F})), "");
    EXPECT_EQ(firstNonFinite("o", DataType::INT32, imageOf<int32_t>({1, 2})), "");
    EXPECT_EQ(firstNonFinite("o", DataType::FLOAT, {}), "");
}

TEST(TestOutputAgreement, FirstNonFiniteNamesTheFirstNaNOrInfinity)
{
    const std::string nanMessage
        = firstNonFinite("y", DataType::FLOAT, imageOf<float>({1.0F, FLOAT_NAN, FLOAT_INF}));
    EXPECT_EQ(nanMessage.rfind("y[1]: ", 0), 0U) << nanMessage;

    const std::string infMessage
        = firstNonFinite("y", DataType::HALF, halfImage({1.0F, 2.0F, -FLOAT_INF}));
    EXPECT_EQ(infMessage.rfind("y[2]: ", 0), 0U) << infMessage;
}

TEST(TestOutputAgreement, FirstNonFiniteIgnoresADtypeWithoutADecode)
{
    EXPECT_EQ(firstNonFinite("o", DataType::FP8_E4M3, std::vector<uint8_t>(4, 0xFF)), "");
}

TEST(TestOutputAgreement, FirstNonFiniteAtReportsTheIndexAndTheValue)
{
    EXPECT_FALSE(firstNonFiniteAt(DataType::FLOAT, imageOf<float>({1.0F, 2.0F})).has_value());
    EXPECT_FALSE(firstNonFiniteAt(DataType::FP8_E4M3, std::vector<uint8_t>(4, 0xFF)).has_value());

    const auto found
        = firstNonFiniteAt(DataType::FLOAT, imageOf<float>({1.0F, -FLOAT_INF, FLOAT_NAN}));
    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(found->index, 1U);
    EXPECT_EQ(found->value, -std::numeric_limits<double>::infinity());
}
