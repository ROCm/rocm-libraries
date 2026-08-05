// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <roc/host_validation/tensor.hpp>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

uint16_t bytesToUint16(std::span<const std::byte> bytes) {
    return static_cast<uint16_t>(std::to_integer<uint8_t>(bytes[0])) |
           static_cast<uint16_t>(std::to_integer<uint8_t>(bytes[1])) << 8;
}
}  // namespace

int main() {
    using namespace roc::host_validation;

    const std::array<float, 16> fp4Expected{
        0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    std::vector<std::byte> fp4Raw(8);
    for (uint8_t index = 0; index < 16; index += 2)
        fp4Raw[index / 2] = static_cast<std::byte>(index | ((index + 1) << 4));
    const Tensor fp4Decoded = Tensor::fromStorage(
        ScalarType::Float4E2M1, Layout::contiguous(Shape{16}), std::move(fp4Raw));
    for (size_t index = 0; index < fp4Expected.size(); ++index)
        require(fp4Decoded.view().loadAs<float>({index}) == fp4Expected[index],
                "FP4 exhaustive decode mismatch.");

    Tensor fp4Encoded(ScalarType::Float4E2M1, Shape{16});
    for (size_t index = 0; index < fp4Expected.size(); ++index)
        fp4Encoded.mutableView().storeFrom({index}, fp4Expected[index]);
    for (uint8_t index = 0; index < 16; ++index) {
        const uint8_t byte = std::to_integer<uint8_t>(fp4Encoded.storage()[index / 2]);
        const uint8_t raw = (index & 1) ? byte >> 4 : byte & 0xf;
        require(raw == index, "FP4 exhaustive encode mismatch.");
    }

    std::vector<std::byte> int4Raw(8);
    for (uint8_t index = 0; index < 16; index += 2)
        int4Raw[index / 2] = static_cast<std::byte>(index | ((index + 1) << 4));
    const Tensor int4 = Tensor::fromStorage(
        ScalarType::Int4, Layout::contiguous(Shape{16}), std::move(int4Raw));
    for (uint8_t index = 0; index < 16; ++index) {
        const int32_t expected = index < 8 ? index : static_cast<int32_t>(index) - 16;
        require(int4.view().loadAs<int32_t>({index}) == expected,
                "Int4 exhaustive decode mismatch.");
    }

    for (uint32_t raw = 0; raw < 64; ++raw) {
        std::vector<std::byte> storage(1, static_cast<std::byte>(raw));
        const Tensor value = Tensor::fromStorage(
            ScalarType::Float6E2M3, Layout::contiguous(Shape{1}), std::move(storage));
        Tensor roundTrip(ScalarType::Float6E2M3, Shape{1});
        roundTrip.mutableView().storeFrom({0}, value.view().loadAs<float>({0}));
        require((std::to_integer<uint8_t>(roundTrip.storage()[0]) & 0x3f) == raw,
                "FP6 exhaustive round-trip mismatch.");
    }

    for (uint32_t raw = 0; raw <= 0xffff; ++raw) {
        std::vector<std::byte> storage{
            static_cast<std::byte>(raw & 0xff),
            static_cast<std::byte>(raw >> 8),
        };
        const Tensor value = Tensor::fromStorage(
            ScalarType::Float16, Layout::contiguous(Shape{1}), std::move(storage));
        const float decoded = value.view().loadAs<float>({0});
        Tensor roundTrip(ScalarType::Float16, Shape{1});
        roundTrip.mutableView().storeFrom({0}, decoded);
        const uint16_t encoded = bytesToUint16(roundTrip.storage());
        if (std::isnan(decoded)) {
            require((encoded & 0x7c00U) == 0x7c00U && (encoded & 0x03ffU) != 0,
                    "Float16 NaN did not remain NaN.");
        } else {
            require(encoded == raw, "Float16 exhaustive round-trip mismatch.");
        }
    }

    std::vector<std::byte> e8Raw{std::byte{0}, std::byte{1}, std::byte{127},
                                 std::byte{128}, std::byte{254}, std::byte{255}};
    const Tensor e8 = Tensor::fromStorage(
        ScalarType::E8M0, Layout::contiguous(Shape{6}), std::move(e8Raw));
    require(e8.view().loadAs<float>({0}) == 0.0f, "E8M0 zero mismatch.");
    require(e8.view().loadAs<float>({1}) == std::ldexp(1.0f, -126),
            "E8M0 minimum mismatch.");
    require(e8.view().loadAs<float>({2}) == 1.0f, "E8M0 unity mismatch.");
    require(e8.view().loadAs<float>({3}) == 2.0f, "E8M0 exponent mismatch.");
    require(std::isnan(e8.view().loadAs<float>({5})), "E8M0 NaN mismatch.");

    return 0;
}
