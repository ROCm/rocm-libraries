// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GUARD_MIOPEN_CONV_HEURISTICS_LGBM_BINARY_HPP
#define GUARD_MIOPEN_CONV_HEURISTICS_LGBM_BINARY_HPP

#include <miopen/config.h>
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

namespace miopen {
namespace ai {
namespace lgbm {

// Bumped whenever the on-disk layout of lgbm_rank.bin / lgbm_pcfg.bin changes.
// A file whose version does not match is rejected (the picker abstains) rather
// than mis-parsed. Kept in lockstep with script/convert_lgbm_binary.py.
inline constexpr std::uint32_t kBinaryFormatVersion = 1;

// Cursor over an in-memory buffer holding one of the LGBM binary assets
// (lgbm_rank.bin / lgbm_pcfg.bin). All multi-byte fields are little-endian,
// which every ROCm host target is; the values are memcpy'd out rather than read
// through a reinterpreted struct so there is no alignment or padding dependency.
//
// Reads are bounds-checked: any read past the end leaves the returned value
// zero/empty and latches Ok() to false. Callers walk the format optimistically
// and check Ok() once at the end (a truncated or corrupt file then abstains
// instead of reading out of bounds).
class BinReader
{
public:
    BinReader(const char* data, std::size_t size) : data_(data), size_(size) {}

    bool Ok() const { return ok_; }
    std::size_t Pos() const { return pos_; }

    template <typename T>
    T ReadScalar()
    {
        static_assert(std::is_trivially_copyable<T>::value, "scalar must be trivially copyable");
        T v{};
        if(!Take(sizeof(T)))
            return v;
        std::memcpy(&v, data_ + pos_ - sizeof(T), sizeof(T));
        return v;
    }

    std::uint8_t ReadU8() { return ReadScalar<std::uint8_t>(); }
    std::uint16_t ReadU16() { return ReadScalar<std::uint16_t>(); }
    std::uint32_t ReadU32() { return ReadScalar<std::uint32_t>(); }
    std::uint64_t ReadU64() { return ReadScalar<std::uint64_t>(); }
    std::int32_t ReadI32() { return ReadScalar<std::int32_t>(); }
    std::int64_t ReadI64() { return ReadScalar<std::int64_t>(); }
    double ReadF64() { return ReadScalar<double>(); }

    // Read `count` little-endian scalars into a vector in one shot.
    template <typename T>
    std::vector<T> ReadArray(std::size_t count)
    {
        static_assert(std::is_trivially_copyable<T>::value, "element must be trivially copyable");
        std::vector<T> out;
        if(count == 0)
            return out;
        if(!Take(count * sizeof(T)))
            return out;
        out.resize(count);
        std::memcpy(out.data(), data_ + pos_ - count * sizeof(T), count * sizeof(T));
        return out;
    }

    // u16 length-prefixed UTF-8 string.
    std::string ReadString()
    {
        const std::uint16_t len = ReadU16();
        if(!Take(len))
            return {};
        return {data_ + pos_ - len, len};
    }

    // Compare the next `n` bytes against a magic literal without advancing on a
    // mismatch beyond consuming them (used once at the header).
    bool ReadMagic(const char* magic, std::size_t n)
    {
        if(!Take(n))
            return false;
        return std::memcmp(data_ + pos_ - n, magic, n) == 0;
    }

    // Reposition to an absolute offset (used to seek to a directory section).
    void SeekTo(std::size_t pos)
    {
        if(pos > size_)
        {
            ok_ = false;
            return;
        }
        pos_ = pos;
    }

private:
    bool Take(std::size_t n)
    {
        if(!ok_ || n > size_ - pos_)
        {
            ok_ = false;
            return false;
        }
        pos_ += n;
        return true;
    }

    const char* data_;
    std::size_t size_;
    std::size_t pos_ = 0;
    bool ok_         = true;
};

} // namespace lgbm
} // namespace ai
} // namespace miopen

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
#endif // GUARD_MIOPEN_CONV_HEURISTICS_LGBM_BINARY_HPP
