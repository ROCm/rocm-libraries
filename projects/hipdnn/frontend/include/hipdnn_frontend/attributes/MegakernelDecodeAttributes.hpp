// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file MegakernelDecodeAttributes.hpp
 * @brief Attributes for a precompiled megakernel decode operation
 */

#pragma once

#include <cstdint>
#include <hipdnn_frontend/Error.hpp>
#include <limits>
#include <optional>
#include <string>
#include <utility>

namespace hipdnn_frontend::graph
{

/**
 * @brief Configuration for a precompiled megakernel decode operation
 *
 * Set all fields. Copy the binary selection fields from the matching engine manifest
 * entry: model identifier, source fingerprint (`src`), GPU architecture (`arch`),
 * compute unit count (`cus`), key/value cache layout (`kv`) and block size (`block`),
 * batch size (`M`), tensor parallel size (`tp`), routing mode (`rt`), and whether
 * the binary includes the language model head (`lm_head`). Set `nlayers` to the
 * number of layers to run (a positive 32-bit integer).
 *
 * @see Graph::megakernel_decode()
 */
class MegakernelDecodeAttributes
{
public:
    // NOLINTBEGIN(readability-identifier-naming)
    MegakernelDecodeAttributes& set_model(std::string value)
    {
        _model = std::move(value);
        return *this;
    }
    const std::string& get_model() const
    {
        return _model;
    }

    MegakernelDecodeAttributes& set_src(std::string value)
    {
        _src = std::move(value);
        return *this;
    }
    const std::string& get_src() const
    {
        return _src;
    }

    MegakernelDecodeAttributes& set_arch(std::string value)
    {
        _arch = std::move(value);
        return *this;
    }
    const std::string& get_arch() const
    {
        return _arch;
    }

    MegakernelDecodeAttributes& set_cus(int64_t value)
    {
        _cus = value;
        return *this;
    }
    int64_t get_cus() const
    {
        return _cus;
    }

    MegakernelDecodeAttributes& set_kv(std::string value)
    {
        _kv = std::move(value);
        return *this;
    }
    const std::string& get_kv() const
    {
        return _kv;
    }

    MegakernelDecodeAttributes& set_block(int64_t value)
    {
        _block = value;
        return *this;
    }
    int64_t get_block() const
    {
        return _block;
    }

    MegakernelDecodeAttributes& set_m(int64_t value)
    {
        _m = value;
        return *this;
    }
    int64_t get_m() const
    {
        return _m;
    }

    MegakernelDecodeAttributes& set_tp(int64_t value)
    {
        _tp = value;
        return *this;
    }
    int64_t get_tp() const
    {
        return _tp;
    }

    MegakernelDecodeAttributes& set_rt(int64_t value)
    {
        _rt = value;
        return *this;
    }
    std::optional<int64_t> get_rt() const
    {
        return _rt;
    }

    MegakernelDecodeAttributes& set_lm_head(bool value)
    {
        _lmHead = value;
        return *this;
    }
    std::optional<bool> get_lm_head() const
    {
        return _lmHead;
    }

    MegakernelDecodeAttributes& set_nlayers(int64_t value)
    {
        _nlayers = value;
        return *this;
    }
    int64_t get_nlayers() const
    {
        return _nlayers;
    }
    // NOLINTEND(readability-identifier-naming)

    /** @brief Check that all required decode attributes have valid values. */
    Error validate() const
    {
        if(_model.empty() || _src.empty() || _arch.empty() || _kv.empty() || !_rt.has_value()
           || !_lmHead.has_value())
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "Megakernel decode requires model, src, arch, kv, rt, and lm_head"};
        }
        if(_cus <= 0 || _block <= 0 || _m <= 0 || _tp <= 0)
        {
            return {ErrorCode::INVALID_VALUE,
                    "Megakernel decode requires positive cus, block, M, and tp"};
        }
        if(_nlayers <= 0 || _nlayers > std::numeric_limits<int32_t>::max())
        {
            return {ErrorCode::INVALID_VALUE,
                    "Megakernel decode requires nlayers to be a positive 32-bit integer"};
        }
        return {};
    }

private:
    std::string _model;
    std::string _src;
    std::string _arch;
    int64_t _cus = 0;
    std::string _kv;
    int64_t _block = 0;
    int64_t _m = 0;
    int64_t _tp = 0;
    std::optional<int64_t> _rt;
    std::optional<bool> _lmHead;
    int64_t _nlayers = 0;
};

} // namespace hipdnn_frontend::graph
