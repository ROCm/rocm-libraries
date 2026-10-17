// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend, used under the MIT license:
//   include/cudnn_frontend_Rng.h
//     Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// License text and pinned upstream version: THIRD_PARTY_LICENSES.md
// (installed to share/doc/hipdnn_frontend).

/**
 * @file cudnn_frontend_Rng.h
 * @brief cuDNN FE v0.x `RngDesc_v8` and its builder, on the shim's backend C API.
 */
#pragma once

#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include <hipdnn_compatibility/cudnn/cudnn_frontend_utils.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_version.h>

namespace hipdnn_frontend::compatibility::cudnn_frontend
{
static_assert(CUDNN_FRONTEND_VERSION == 12400,
              "cuDNN FE version pin changed: re-diff this file against upstream "
              "include/cudnn_frontend_Rng.h at the new tag, then update this assert.");

// NOLINTBEGIN(readability-identifier-naming,modernize-use-override,modernize-use-equals-delete):
// ported verbatim; upstream's spelling and shapes are the source-compatible surface.

class RngDesc_v8 : public BackendDescriptor
{
public:
    friend class RngDescBuilder_v8;
    std::string describe() const override
    {
        std::stringstream ss;
#if(CUDNN_VERSION >= 8700)
#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
        ss << "CUDNN_BACKEND_RNG_DESCRIPTOR: " << "Distribution Type: "
           << detail::enumJson(distribution)
#else
        ss << "CUDNN_BACKEND_RNG_DESCRIPTOR: " << "Distribution Type: "
           << static_cast<int>(distribution)
#endif
           << ", Normal Distribution Mean: " << normal_dist_mean
           << ", Normal Distribution Standard Deviation: " << normal_dist_std_dev
           << ", Uniform Distribution Maximum: " << uniform_dist_max
           << ", Uniform Distribution Minimum: " << uniform_dist_min
           << ", Bernoulli Distribution Probability: " << bernoulli_dist_probability;
#endif
        return ss.str();
    }

    RngDesc_v8(RngDesc_v8&& from) = default;
    RngDesc_v8& operator=(RngDesc_v8&&) = default;

    ~RngDesc_v8() = default;

    double getNormalDistMean() const
    {
        return normal_dist_mean;
    }

    double getNormalDistStdDev() const
    {
        return normal_dist_std_dev;
    }

    double getUniformDistMax() const
    {
        return uniform_dist_max;
    }

    double getUniformDistMin() const
    {
        return uniform_dist_min;
    }

    double getBernoulliDistProbability() const
    {
        return bernoulli_dist_probability;
    }

    RngDistribution_t getDistribution() const
    {
        return distribution;
    }

private:
    RngDesc_v8() = default;
    RngDesc_v8(const RngDesc_v8&) = delete;
    RngDesc_v8& operator=(const RngDesc_v8&) = delete;

    double normal_dist_mean = -1;
    double normal_dist_std_dev = -1;
    double uniform_dist_max = -1;
    double uniform_dist_min = -1;
    double bernoulli_dist_probability = -1;

    RngDistribution_t distribution = RngDistribution_t::NOT_SET;
};

class RngDescBuilder_v8
{
public:
    auto setRngDistribution(RngDistribution_t distribution_) -> RngDescBuilder_v8&
    {
        m_RngDesc.distribution = distribution_;
        return *this;
    }

#if(CUDNN_VERSION >= 8700)
    auto setRngDistribution(cudnnRngDistribution_t distribution_) -> RngDescBuilder_v8&
    {
        m_RngDesc.distribution = detail::convert_from_cudnn_type(distribution_);
        return *this;
    }

#endif

    auto setNormalDistParams(double normal_dist_mean_,
                             double normal_dist_std_dev_) -> RngDescBuilder_v8&
    {
        m_RngDesc.normal_dist_mean = normal_dist_mean_;
        m_RngDesc.normal_dist_std_dev = normal_dist_std_dev_;
        return *this;
    }

    auto setNormalDistMean(double normal_dist_mean_) -> RngDescBuilder_v8&
    {
        m_RngDesc.normal_dist_mean = normal_dist_mean_;
        return *this;
    }

    auto setNormalDistStdDev(double normal_dist_std_dev_) -> RngDescBuilder_v8&
    {
        m_RngDesc.normal_dist_std_dev = normal_dist_std_dev_;
        return *this;
    }

    auto setUniformDistParams(double uniform_dist_max_,
                              double uniform_dist_min_) -> RngDescBuilder_v8&
    {
        m_RngDesc.uniform_dist_max = uniform_dist_max_;
        m_RngDesc.uniform_dist_min = uniform_dist_min_;
        return *this;
    }

    auto setUniformDistMax(double uniform_dist_max_) -> RngDescBuilder_v8&
    {
        m_RngDesc.uniform_dist_max = uniform_dist_max_;
        return *this;
    }

    auto setUniformDistMin(double uniform_dist_min_) -> RngDescBuilder_v8&
    {
        m_RngDesc.uniform_dist_min = uniform_dist_min_;
        return *this;
    }

    auto setBernoulliDistProbability(double bernoulli_dist_probability_) -> RngDescBuilder_v8&
    {
        m_RngDesc.bernoulli_dist_probability = bernoulli_dist_probability_;
        return *this;
    }

    RngDesc_v8&& build()
    {
#if(CUDNN_VERSION >= 8700)
        NV_CUDNN_FE_DYNAMIC_CHECK_BACKEND_DESCRIPTOR(
            8700, m_RngDesc, "CUDNN_BACKEND_RNG_DESCRIPTOR: Requires cudnn 8.7.0");

        auto status = m_RngDesc.initialize_managed_backend_pointer(CUDNN_BACKEND_RNG_DESCRIPTOR);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(
                &m_RngDesc, status, "CUDNN_BACKEND_RNG_DESCRIPTOR: cudnnCreate Failed");
            return std::move(m_RngDesc);
        }

        cudnnRngDistribution_t cudnn_rng_distribution;
        status = detail::convert_to_cudnn_type(m_RngDesc.distribution, cudnn_rng_distribution);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(
                &m_RngDesc,
                status,
                "CUDNN_BACKEND_RNG_DESCRIPTOR: SetAttribute CUDNN_ATTR_RNG_DISTRIBUTION Failed");
            return std::move(m_RngDesc);
        }

        status = detail::set_attribute(m_RngDesc.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_RNG_DISTRIBUTION,
                                       CUDNN_TYPE_RNG_DISTRIBUTION,
                                       1,
                                       &cudnn_rng_distribution);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(
                &m_RngDesc,
                status,
                "CUDNN_BACKEND_RNG_DESCRIPTOR: SetAttribute CUDNN_ATTR_RNG_DISTRIBUTION Failed");
            return std::move(m_RngDesc);
        }

        status = detail::set_attribute(m_RngDesc.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_RNG_NORMAL_DIST_MEAN,
                                       CUDNN_TYPE_DOUBLE,
                                       1,
                                       &(m_RngDesc.normal_dist_mean));
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_RngDesc,
                                          status,
                                          "CUDNN_BACKEND_RNG_DESCRIPTOR: SetAttribute "
                                          "CUDNN_ATTR_RNG_NORMAL_DIST_MEAN Failed");
            return std::move(m_RngDesc);
        }

        status = detail::set_attribute(m_RngDesc.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_RNG_NORMAL_DIST_STANDARD_DEVIATION,
                                       CUDNN_TYPE_DOUBLE,
                                       1,
                                       &(m_RngDesc.normal_dist_std_dev));
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_RngDesc,
                                          status,
                                          "CUDNN_BACKEND_RNG_DESCRIPTOR: SetAttribute "
                                          "CUDNN_ATTR_RNG_NORMAL_DIST_STANDARD_DEVIATION Failed");
            return std::move(m_RngDesc);
        }

        status = detail::set_attribute(m_RngDesc.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_RNG_UNIFORM_DIST_MAXIMUM,
                                       CUDNN_TYPE_DOUBLE,
                                       1,
                                       &(m_RngDesc.uniform_dist_max));
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_RngDesc,
                                          status,
                                          "CUDNN_BACKEND_RNG_DESCRIPTOR: SetAttribute "
                                          "CUDNN_ATTR_RNG_UNIFORM_DIST_MAXIMUM Failed");
            return std::move(m_RngDesc);
        }

        status = detail::set_attribute(m_RngDesc.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_RNG_UNIFORM_DIST_MINIMUM,
                                       CUDNN_TYPE_DOUBLE,
                                       1,
                                       &(m_RngDesc.uniform_dist_min));
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_RngDesc,
                                          status,
                                          "CUDNN_BACKEND_RNG_DESCRIPTOR: SetAttribute "
                                          "CUDNN_ATTR_RNG_UNIFORM_DIST_MINIMUM Failed");
            return std::move(m_RngDesc);
        }

        status = detail::set_attribute(m_RngDesc.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY,
                                       CUDNN_TYPE_DOUBLE,
                                       1,
                                       &(m_RngDesc.bernoulli_dist_probability));
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(&m_RngDesc,
                                          status,
                                          "CUDNN_BACKEND_RNG_DESCRIPTOR: SetAttribute "
                                          "CUDNN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY Failed");
            return std::move(m_RngDesc);
        }

        status = detail::finalize(m_RngDesc.pointer->get_backend_descriptor());
        if(status != CUDNN_STATUS_SUCCESS)
        {
            set_error_and_throw_exception(
                &m_RngDesc, status, "CUDNN_BACKEND_RNG_DESCRIPTOR: cudnnFinalize Failed");
            return std::move(m_RngDesc);
        }
        CUDNN_FE_LOG_LABEL_ENDL(m_RngDesc);
        return std::move(m_RngDesc);
#else
        set_error_and_throw_exception(
            &m_RngDesc,
            CUDNN_STATUS_NOT_SUPPORTED,
            "CUDNN_RNG_DESCRIPTOR: Rng only supported in cuDNN v8.7 or later");
        return std::move(m_RngDesc);
#endif
    }

    explicit RngDescBuilder_v8() = default;
    ~RngDescBuilder_v8() = default;
    RngDescBuilder_v8(RngDescBuilder_v8&&) = delete;
    RngDescBuilder_v8(const RngDescBuilder_v8&) = delete;
    RngDescBuilder_v8& operator=(const RngDescBuilder_v8&) = delete;

private:
    RngDesc_v8 m_RngDesc;
};

using RngDesc = RngDesc_v8;
using RngDescBuilder = RngDescBuilder_v8;

// NOLINTEND(readability-identifier-naming,modernize-use-override,modernize-use-equals-delete)

} // namespace hipdnn_frontend::compatibility::cudnn_frontend
