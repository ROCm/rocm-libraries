// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// ============================================================================
// Minimal, dependency-free NumPy .npy v1.0 reader/writer for the standalone
// SDPA gpu_ref vs selectable PyTorch MATH/AOTriton reference driver.
//
// Reader: parses C-contiguous, little-endian arrays of dtype '<f4', '<f2',
// '<u2' or '|u1' and returns the raw bytes plus shape. No dtype conversion is
// performed; the caller reinterprets the raw bytes (e.g. '<u2' bits as bf16,
// '|u1' bits as fp8).
//
// Writer: emits a C-contiguous, little-endian '<f4' (fp32) array. The header is
// a v1.0 header dict padded with spaces so that the total header length is a
// multiple of 64 and terminated with '\n', exactly as numpy.lib.format expects.
// ============================================================================

#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sdpa_harness::npy
{

// Supported on-disk element types.
enum class DType
{
    F4, // '<f4' little-endian fp32
    F2, // '<f2' little-endian fp16
    U2, // '<u2' little-endian uint16 (raw bf16 bits)
    U1 // '|u1' single-byte uint8 (raw fp8 bits)
};

// Result of reading an .npy file: raw little-endian bytes, logical shape and dtype.
struct NpyArray
{
    std::vector<char> data;
    std::vector<int64_t> shape;
    DType dtype{DType::F4};

    // Product of all shape dimensions.
    int64_t elementCount() const
    {
        int64_t count = 1;
        for(const int64_t dim : shape)
        {
            count *= dim;
        }
        return count;
    }
};

namespace detail
{

// Byte size of one element of the given dtype.
inline size_t dtypeSize(DType dtype)
{
    switch(dtype)
    {
    case DType::F4:
        return 4;
    case DType::F2:
        return 2;
    case DType::U2:
        return 2;
    case DType::U1:
        return 1;
    default:
        throw std::runtime_error("NpyIO: unknown dtype");
    }
}

// Map a numpy 'descr' string to a DType, throwing on anything unsupported.
inline DType parseDescr(const std::string& descr)
{
    if(descr == "<f4")
    {
        return DType::F4;
    }
    if(descr == "<f2")
    {
        return DType::F2;
    }
    if(descr == "<u2")
    {
        return DType::U2;
    }
    // Single-byte dtype has no meaningful byte order; numpy emits '|u1'.
    if(descr == "|u1" || descr == "<u1" || descr == "u1")
    {
        return DType::U1;
    }
    throw std::runtime_error("NpyIO: unsupported descr '" + descr
                             + "' (only '<f4', '<f2', '<u2', '|u1' are supported)");
}

// Extract the value of a single-quoted string field (e.g. 'descr':'<f4') from
// the header dict. Returns the unquoted value.
inline std::string extractQuoted(const std::string& header, const std::string& key)
{
    const std::string needle = "'" + key + "'";
    const size_t keyPos = header.find(needle);
    if(keyPos == std::string::npos)
    {
        throw std::runtime_error("NpyIO: header missing key '" + key + "'");
    }
    const size_t colon = header.find(':', keyPos + needle.size());
    if(colon == std::string::npos)
    {
        throw std::runtime_error("NpyIO: malformed header near key '" + key + "'");
    }
    const size_t open = header.find('\'', colon + 1);
    if(open == std::string::npos)
    {
        throw std::runtime_error("NpyIO: malformed header value for key '" + key + "'");
    }
    const size_t close = header.find('\'', open + 1);
    if(close == std::string::npos)
    {
        throw std::runtime_error("NpyIO: malformed header value for key '" + key + "'");
    }
    return header.substr(open + 1, close - open - 1);
}

// Parse fortran_order (expects exactly True/False).
inline bool extractFortranOrder(const std::string& header)
{
    const std::string needle = "'fortran_order'";
    const size_t keyPos = header.find(needle);
    if(keyPos == std::string::npos)
    {
        throw std::runtime_error("NpyIO: header missing 'fortran_order'");
    }
    if(header.find("True", keyPos) != std::string::npos
       && header.find("True", keyPos) < header.find("False", keyPos))
    {
        return true;
    }
    if(header.find("False", keyPos) != std::string::npos)
    {
        return false;
    }
    throw std::runtime_error("NpyIO: malformed 'fortran_order'");
}

// Parse the shape tuple, e.g. "(2, 4, 8, 16)" or "(8,)" or "()".
inline std::vector<int64_t> extractShape(const std::string& header)
{
    const std::string needle = "'shape'";
    const size_t keyPos = header.find(needle);
    if(keyPos == std::string::npos)
    {
        throw std::runtime_error("NpyIO: header missing 'shape'");
    }
    const size_t open = header.find('(', keyPos);
    const size_t close = header.find(')', open);
    if(open == std::string::npos || close == std::string::npos || close < open)
    {
        throw std::runtime_error("NpyIO: malformed shape tuple");
    }

    std::vector<int64_t> shape;
    const std::string inner = header.substr(open + 1, close - open - 1);
    std::stringstream ss(inner);
    std::string token;
    while(std::getline(ss, token, ','))
    {
        // Strip whitespace.
        size_t begin = token.find_first_not_of(" \t");
        if(begin == std::string::npos)
        {
            continue; // empty token (trailing comma)
        }
        const size_t end = token.find_last_not_of(" \t");
        const std::string trimmed = token.substr(begin, end - begin + 1);
        if(trimmed.empty())
        {
            continue;
        }
        try
        {
            shape.push_back(static_cast<int64_t>(std::stoll(trimmed)));
        }
        catch(const std::exception&)
        {
            throw std::runtime_error("NpyIO: non-integer in shape tuple: '" + trimmed + "'");
        }
    }
    return shape;
}

} // namespace detail

// Read a v1.0 .npy file. Throws std::runtime_error on any unsupported or
// malformed input.
inline NpyArray read(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if(!in)
    {
        throw std::runtime_error("NpyIO: cannot open file for reading: " + path);
    }

    char magic[6] = {};
    in.read(magic, 6);
    if(in.gcount() != 6 || std::memcmp(magic, "\x93NUMPY", 6) != 0)
    {
        throw std::runtime_error("NpyIO: not a .npy file (bad magic): " + path);
    }

    unsigned char version[2] = {};
    in.read(reinterpret_cast<char*>(version), 2);
    if(in.gcount() != 2 || version[0] != 1 || version[1] != 0)
    {
        throw std::runtime_error("NpyIO: only .npy version 1.0 is supported: " + path);
    }

    unsigned char lenBytes[2] = {};
    in.read(reinterpret_cast<char*>(lenBytes), 2);
    if(in.gcount() != 2)
    {
        throw std::runtime_error("NpyIO: truncated header length: " + path);
    }
    const size_t headerLen
        = static_cast<size_t>(lenBytes[0]) | (static_cast<size_t>(lenBytes[1]) << 8U);

    std::string header(headerLen, '\0');
    in.read(header.data(), static_cast<std::streamsize>(headerLen));
    if(static_cast<size_t>(in.gcount()) != headerLen)
    {
        throw std::runtime_error("NpyIO: truncated header: " + path);
    }

    NpyArray result;
    result.dtype = detail::parseDescr(detail::extractQuoted(header, "descr"));
    if(detail::extractFortranOrder(header))
    {
        throw std::runtime_error("NpyIO: fortran_order arrays are not supported: " + path);
    }
    result.shape = detail::extractShape(header);

    const int64_t elementCount = result.elementCount();
    const size_t byteCount = static_cast<size_t>(elementCount) * detail::dtypeSize(result.dtype);
    result.data.resize(byteCount);
    in.read(result.data.data(), static_cast<std::streamsize>(byteCount));
    if(static_cast<size_t>(in.gcount()) != byteCount)
    {
        throw std::runtime_error("NpyIO: truncated data payload: " + path);
    }

    return result;
}

namespace detail
{

// Build a v1.0 header dict for an '<f4' array of the given shape, padded so the
// total header (magic + version + 2-byte length + dict) is a multiple of 64 and
// the dict is terminated with '\n'.
inline std::string buildF4Header(const std::vector<int64_t>& shape)
{
    std::ostringstream dict;
    dict << "{'descr': '<f4', 'fortran_order': False, 'shape': (";
    for(size_t i = 0; i < shape.size(); ++i)
    {
        dict << shape[i];
        // numpy always emits a trailing comma for rank-1 tuples, and a comma
        // separator otherwise. Emitting "n," for every dim is valid Python and
        // matches numpy's output for rank-1; for rank>1 a trailing comma is also
        // accepted by numpy's parser.
        if(i + 1 < shape.size())
        {
            dict << ", ";
        }
    }
    if(shape.size() == 1)
    {
        dict << ",";
    }
    dict << ")}";

    std::string header = dict.str();

    // Total preamble = 6 (magic) + 2 (version) + 2 (header length) = 10 bytes.
    // The header dict + trailing '\n' must make the grand total a multiple of 64.
    constexpr size_t preamble = 10;
    const size_t unpadded = preamble + header.size() + 1; // +1 for '\n'
    const size_t padded = ((unpadded + 63) / 64) * 64;
    const size_t padding = padded - unpadded;
    header.append(padding, ' ');
    header.push_back('\n');
    return header;
}

} // namespace detail

// Write an '<f4' (fp32) C-contiguous array of the given shape. `data` must point
// to at least (product of shape) floats.
inline void writeF4(const std::string& path, const float* data, const std::vector<int64_t>& shape)
{
    std::ofstream out(path, std::ios::binary);
    if(!out)
    {
        throw std::runtime_error("NpyIO: cannot open file for writing: " + path);
    }

    const std::string header = detail::buildF4Header(shape);
    const size_t headerLen = header.size();
    if(headerLen > 0xFFFFU)
    {
        throw std::runtime_error("NpyIO: header too large for v1.0 format: " + path);
    }

    out.write("\x93NUMPY", 6);
    const char version[2] = {1, 0};
    out.write(version, 2);
    const unsigned char lenBytes[2] = {static_cast<unsigned char>(headerLen & 0xFFU),
                                       static_cast<unsigned char>((headerLen >> 8U) & 0xFFU)};
    out.write(reinterpret_cast<const char*>(lenBytes), 2);
    out.write(header.data(), static_cast<std::streamsize>(headerLen));

    int64_t elementCount = 1;
    for(const int64_t dim : shape)
    {
        elementCount *= dim;
    }
    out.write(reinterpret_cast<const char*>(data),
              static_cast<std::streamsize>(static_cast<size_t>(elementCount) * sizeof(float)));

    if(!out)
    {
        throw std::runtime_error("NpyIO: failed while writing data payload: " + path);
    }
}

} // namespace sdpa_harness::npy
