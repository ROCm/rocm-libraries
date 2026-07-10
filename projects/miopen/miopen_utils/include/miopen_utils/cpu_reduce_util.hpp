/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef GUARD_CPU_REDUCE_UTIL_HPP
#define GUARD_CPU_REDUCE_UTIL_HPP

#include <miopen_utils/tensor_holder.hpp>
#include <cstddef>
#include <half/half.hpp>
#include <limits>
#include <cmath>
#include <cassert>
#include <ratio>
#include <stdexcept>
#include <string>
#include <miopen/miopen.h>
#include <common_utils/reduce_common.hpp>

namespace reduce {

template <typename T>
static inline bool float_equal_one(T);

static inline bool float_equal_one(float x) { return x == 1.0f; };

static inline bool float_equal_one(double x) { return x == 1.0; };

static inline bool float_equal_one(half_float::half x)
{
    return x == convert_type<half_float::half>(1.0f);
};

template <typename T>
static inline bool float_equal_zero(T x);

static inline bool float_equal_zero(float x) { return x == 0.0f; };

static inline bool float_equal_zero(double x) { return x == 0.0; };

static inline bool float_equal_zero(half_float::half x)
{
    return x == convert_type<half_float::half>(0.0f);
};

template <typename SizeT>
static inline void build_radix(const std::vector<SizeT>& lens, std::vector<std::size_t>& radix)
{
    const std::size_t D = lens.size();
    radix.assign(D, 1);
    for(std::size_t d = D; d-- > 1;)
        radix[d - 1] = radix[d] * static_cast<std::size_t>(lens[d]); // radix[d] = Π_{k>d} lens[k]
}

// i -> memory offset using lens-radix + actual strides
template <typename SizeT>
static inline std::size_t linear_to_offset_by_lens_strides(std::size_t i,
                                                           const std::vector<SizeT>& lens,
                                                           const std::vector<std::size_t>& radix,
                                                           const std::vector<SizeT>& strides)
{
    std::size_t off = 0;
    for(std::size_t d = 0; d < lens.size(); ++d)
    {
        const std::size_t idx_d = (i / radix[d]) % static_cast<std::size_t>(lens[d]);
        off += idx_d * static_cast<std::size_t>(strides[d]);
    }
    return off;
}

template <typename compType>
static inline std::function<void(compType&)> PreUnaryOpFn(miopenReduceTensorOp_t op_, std::size_t)
{
    using std::abs;

    switch(op_)
    {
    case MIOPEN_REDUCE_TENSOR_NORM1: return ([&](compType& a_) { a_ = abs(a_); });
    case MIOPEN_REDUCE_TENSOR_NORM2: return ([&](compType& a_) { a_ = a_ * a_; });
    case MIOPEN_REDUCE_TENSOR_AMAX: return ([&](compType& a_) { a_ = abs(a_); });

    case MIOPEN_REDUCE_TENSOR_AVG:
    case MIOPEN_REDUCE_TENSOR_ADD:
    case MIOPEN_REDUCE_TENSOR_MUL:
    case MIOPEN_REDUCE_TENSOR_MIN:
    case MIOPEN_REDUCE_TENSOR_MAX: return ([&](compType&) {});
    }

    throw std::runtime_error(std::string(__FUNCTION__) +
                             ": using undefined Reduction operation is not permitted");
};

template <typename compType>
static inline std::function<void(compType&)> PosUnaryOpFn(miopenReduceTensorOp_t op_,
                                                          std::size_t divider)
{
    using std::sqrt;

    switch(op_)
    {
    case MIOPEN_REDUCE_TENSOR_NORM2: return ([&](compType& a_) { a_ = sqrt(a_); });

    case MIOPEN_REDUCE_TENSOR_AVG:
        return ([&, divider](compType& a_) {
            a_ = a_ / convert_type<compType>(static_cast<float>(divider));
        });

    case MIOPEN_REDUCE_TENSOR_ADD:
    case MIOPEN_REDUCE_TENSOR_NORM1:
    case MIOPEN_REDUCE_TENSOR_MUL:
    case MIOPEN_REDUCE_TENSOR_MIN:
    case MIOPEN_REDUCE_TENSOR_MAX:
    case MIOPEN_REDUCE_TENSOR_AMAX: return ([&](compType&) {});
    }

    throw std::runtime_error(std::string(__FUNCTION__) +
                             ": using undefined Reduction operation is not permitted");
};

template <typename compType>
static inline std::function<void(compType&, compType)> ReduceOpFn(miopenReduceTensorOp_t op_)
{
    switch(op_)
    {
    case MIOPEN_REDUCE_TENSOR_ADD:
    case MIOPEN_REDUCE_TENSOR_AVG:
    case MIOPEN_REDUCE_TENSOR_NORM1:
    case MIOPEN_REDUCE_TENSOR_NORM2: return ([&](compType& a_, compType b_) { a_ = a_ + b_; });

    case MIOPEN_REDUCE_TENSOR_MUL: return ([&](compType& a_, compType b_) { a_ = a_ * b_; });

    case MIOPEN_REDUCE_TENSOR_MIN:
        return ([&](compType& a_, compType b_) {
            if(a_ > b_)
                a_ = b_;
        });

    case MIOPEN_REDUCE_TENSOR_MAX:
    case MIOPEN_REDUCE_TENSOR_AMAX:
        return ([&](compType& a_, compType b_) {
            if(a_ < b_)
                a_ = b_;
        });
    }

    throw std::runtime_error(std::string(__FUNCTION__) +
                             ": using undefined Reduction operation is not permitted");
};

template <typename compType>
static inline std::function<void(compType&, compType, bool& changed)>
ReduceOpFn2(miopenReduceTensorOp_t op_)
{
    switch(op_)
    {
    case MIOPEN_REDUCE_TENSOR_MIN:
        return ([&](compType& a_, compType b_, bool& changed) {
            if(a_ > b_)
            {
                a_      = b_;
                changed = true;
            }
            else
            {
                changed = false;
            }
        });

    case MIOPEN_REDUCE_TENSOR_MAX:
    case MIOPEN_REDUCE_TENSOR_AMAX:
        return ([&](compType& a_, compType b_, bool& changed) {
            if(a_ < b_)
            {
                a_      = b_;
                changed = true;
            }
            else
            {
                changed = false;
            }
        });

    case MIOPEN_REDUCE_TENSOR_ADD:
    case MIOPEN_REDUCE_TENSOR_MUL:
    case MIOPEN_REDUCE_TENSOR_AVG:
    case MIOPEN_REDUCE_TENSOR_NORM1:
    case MIOPEN_REDUCE_TENSOR_NORM2: return (std::function<void(compType&, compType, bool&)>{});
    };

    throw std::runtime_error(std::string(__FUNCTION__) +
                             ": using undefined Reduction operation is not permitted");
};

template <typename compType>
static inline compType ReduceOpZeroVal(miopenReduceTensorOp_t op_)
{
    switch(op_)
    {
    case MIOPEN_REDUCE_TENSOR_ADD:
    case MIOPEN_REDUCE_TENSOR_AVG:
    case MIOPEN_REDUCE_TENSOR_NORM1:
    case MIOPEN_REDUCE_TENSOR_NORM2: return (convert_type<compType>(0.0f));

    case MIOPEN_REDUCE_TENSOR_MUL: return (convert_type<compType>(1.0f));

    case MIOPEN_REDUCE_TENSOR_MIN: return (std::numeric_limits<compType>::max());

    case MIOPEN_REDUCE_TENSOR_MAX: return (std::numeric_limits<compType>::lowest());
    case MIOPEN_REDUCE_TENSOR_AMAX: return (convert_type<compType>(0.0f));
    }

    throw std::runtime_error(std::string(__FUNCTION__) +
                             ": using undefined Reduction operation is not permitted");
};

template <typename compType, typename reduceOpT>
static inline void binop_with_nan_check(miopenNanPropagation_t nanOpt,
                                        reduceOpT&& opReduce,
                                        compType& accuVal,
                                        compType currVal)
{
    using std::isnan;

    if(nanOpt == MIOPEN_NOT_PROPAGATE_NAN)
    {
        opReduce(accuVal, currVal);
    }
    else
    {
        if(isnan(currVal))
            accuVal = currVal;
        else
            opReduce(accuVal, currVal);
    };
};

template <typename compType, typename reduceOpT>
static inline void binop_with_nan_check2(miopenNanPropagation_t nanOpt,
                                         reduceOpT&& opReduce,
                                         compType& accuVal,
                                         compType currVal,
                                         int& accuIndex,
                                         int currIndex)
{
    using std::isnan;

    if(nanOpt == MIOPEN_NOT_PROPAGATE_NAN)
    {
        bool changed;

        opReduce(accuVal, currVal, changed);

        if(changed)
            accuIndex = currIndex;
    }
    else
    {
        if(isnan(currVal))
        {
            accuVal   = currVal;
            accuIndex = currIndex;
        }
        else
        {
            bool changed;

            opReduce(accuVal, currVal, changed);

            if(changed)
                accuIndex = currIndex;
        };
    };
};

}; // end of namespace reduce

template <typename T>
std::vector<std::vector<T>> get_all_indexes(const std::vector<T>& lens)
{
    const std::size_t D = lens.size();
    assert(D > 0);

    std::size_t N = 1;
    for(const auto L : lens)
        N *= static_cast<std::size_t>(L);

    std::vector<std::vector<T>> out;
    out.resize(N);
    for(auto& row : out)
        row.resize(D);

    std::vector<std::size_t> stride(D, 1);
    for(std::size_t d = D; d-- > 1;)
        stride[d - 1] = stride[d] * static_cast<std::size_t>(lens[d]);

    for(std::size_t r = 0; r < N; ++r)
    {
        for(std::size_t d = 0; d < D; ++d)
            out[r][d] = static_cast<T>((r / stride[d]) % static_cast<std::size_t>(lens[d]));
    }

    return out;
}

template <typename T>
static inline T
linear_to_offset(size_t li, const std::vector<T>& lens, const std::vector<T>& strides)
{
    T off = 0;
    for(int d = int(lens.size()) - 1; d >= 0; --d)
    {
        const T idx = li % lens[d];
        li /= lens[d];
        off += idx * strides[d];
    }
    return off;
}

template <typename T>
T get_offset_from_index(const std::vector<T>& strides, const std::vector<T>& index)
{
    T offset = 0;

    assert(strides.size() == index.size());

    for(int i = 0; i < index.size(); i++)
        offset += strides[i] * index[i];

    return (offset);
};

template <typename T>
T get_flatten_offset(const std::vector<T>& lengths, const std::vector<T>& index)
{
    T offset = 0;

    assert(lengths.size() == index.size() && !lengths.empty());

    int len  = lengths.size();
    T stride = 1;

    // for len==1, the loop is not executed
    for(int i = len - 1; i > 0; i--)
    {
        offset += stride * index[i];

        stride *= lengths[i];
    };

    offset += stride * index[0];

    return (offset);
};

template <typename compType>
struct Reducer
{
    compType acc;
    bool withIdx;
    int idx; // meaningful only when WithIdx==true
    miopenNanPropagation_t nanOpt;
    // functors for reduction
    decltype(reduce::ReduceOpFn<compType>(MIOPEN_REDUCE_TENSOR_ADD)) opNoIdx;
    decltype(reduce::ReduceOpFn2<compType>(MIOPEN_REDUCE_TENSOR_ADD)) opWithIdx;

    Reducer(miopenNanPropagation_t n, miopenReduceTensorOp_t rop, compType zero, bool useIdx)
        : acc(zero),
          withIdx(useIdx),
          idx(0),
          nanOpt(n),
          opNoIdx(reduce::ReduceOpFn<compType>(rop)),
          opWithIdx(reduce::ReduceOpFn2<compType>(rop))
    {
    }

    inline void step(compType v, int flat_i)
    {
        if(withIdx)
            reduce::binop_with_nan_check2(nanOpt, opWithIdx, acc, v, idx, flat_i);
        else
            reduce::binop_with_nan_check(nanOpt, opNoIdx, acc, v);
    }

    inline void combine(const Reducer& other)
    {
        if(withIdx)
            reduce::binop_with_nan_check2(nanOpt, opWithIdx, acc, other.acc, idx, other.idx);
        else
            reduce::binop_with_nan_check(nanOpt, opNoIdx, acc, other.acc);
    }
};

template <typename Tgpu, typename Tref, typename compType, typename SizeT>
std::tuple<tensor<Tref>, tensor<int>> reduce_cpu_common(const miopenReduceTensorOp_t& reduceOp,
                                                        const miopenNanPropagation_t& nanOpt,
                                                        const std::vector<SizeT>& inLengths,
                                                        const std::vector<SizeT>& outLengths,
                                                        const std::vector<Tgpu>& input,
                                                        const std::vector<SizeT>& inStrides,
                                                        const std::vector<Tref>& output,
                                                        const std::vector<SizeT>& outStrides,
                                                        float alpha,
                                                        float beta,
                                                        bool parallel,
                                                        bool withIdx)
{
    using reduce::convert_type;
    using reduce::ReduceOpZeroVal;

    // Partition dims
    std::vector<int> invariantDims, toReduceDims;
    std::vector<std::size_t> invLens, redLens, invStrides_v, redStrides_v;

    for(int i = 0; i < static_cast<int>(inLengths.size()); ++i)
    {
        if(inLengths[i] == outLengths[i])
        {
            invariantDims.push_back(i);
            invLens.push_back(inLengths[i]);
            invStrides_v.push_back(inStrides[i]);
        }
        else
        {
            toReduceDims.push_back(i);
            redLens.push_back(inLengths[i]);
            redStrides_v.push_back(inStrides[i]);
        }
    }

    const bool reduceAllDims = invariantDims.empty();

    // unary ops & zero vals
    const compType zeroV = ReduceOpZeroVal<compType>(reduceOp);

    // divider = Π reduced dims (or N if reduce-all)
    std::size_t divider = 1;
    if(reduceAllDims)
        divider = std::accumulate(
            inLengths.begin(), inLengths.end(), std::size_t{1}, std::multiplies<>());
    else
        divider =
            std::accumulate(redLens.begin(), redLens.end(), std::size_t{1}, std::multiplies<>());

    auto PreUnaryOp = reduce::PreUnaryOpFn<compType>(reduceOp, divider);
    auto PosUnaryOp = reduce::PosUnaryOpFn<compType>(reduceOp, divider);

    // outputs
    auto res         = tensor<Tref>{outLengths};
    res.data         = output;
    auto res_indices = tensor<int>{outLengths};
    if(withIdx)
        std::fill(res_indices.begin(), res_indices.end(), 0);

    if(reduceAllDims)
    {
        // Flatten whole tensor
        const std::size_t N = divider; // product of all dims
        std::vector<std::size_t> lens_radix;
        reduce::build_radix(inLengths, lens_radix);

        // parallel chunking
        std::size_t hw =
            std::max(std::size_t{1}, static_cast<std::size_t>(std::thread::hardware_concurrency()));
        const std::size_t P     = std::min(N, hw * 4ul);
        const std::size_t chunk = (N + P - 1) / P;

        std::vector<Reducer<compType>> partial;
        partial.reserve(P);
        for(std::size_t p = 0; p < P; ++p)
            partial.emplace_back(nanOpt, reduceOp, zeroV, withIdx);

        auto worker = [&](int p) {
            const std::size_t begin = std::size_t(p) * chunk;
            const std::size_t end   = std::min(begin + chunk, N);

            auto& r = partial[p];
            for(std::size_t i = begin; i < end; ++i)
            {
                const auto off =
                    reduce::linear_to_offset_by_lens_strides(i, inLengths, lens_radix, inStrides);
                auto v = convert_type<compType>(input[off]);
                PreUnaryOp(v);
                r.step(v, static_cast<int>(i)); // flat index across whole tensor
            }
        };

        if(parallel)
        {
            miopen::par_for(static_cast<int>(P), worker);
        }
        else
        {
            for(int p = 0; p < P; ++p)
            {
                worker(p);
            }
        }

        // combine
        Reducer<compType> R(nanOpt, reduceOp, zeroV, withIdx);
        for(std::size_t p = 0; p < P; ++p)
            R.combine(partial[p]);

        // post
        PosUnaryOp(R.acc);
        if(alpha != 1.0f)
            R.acc *= convert_type<compType>(alpha);
        if(beta != 0.0f)
            R.acc += convert_type<compType>(output[0]) * convert_type<compType>(beta);

        res.data[0] = convert_type<Tref>(R.acc);
        if(withIdx)
            res_indices.data[0] = R.idx;
    }
    else
    {
        // Build radices for invariant and reduced subspaces
        std::vector<std::size_t> invRad, redRad;
        reduce::build_radix(invLens, invRad);
        reduce::build_radix(redLens, redRad);

        const std::size_t INV =
            std::accumulate(invLens.begin(), invLens.end(), std::size_t{1}, std::multiplies<>());
        const std::size_t TR = divider;

        std::size_t hw =
            std::max(std::size_t{1}, static_cast<std::size_t>(std::thread::hardware_concurrency()));
        const std::size_t Te    = std::min(hw * 4ul, std::max<std::size_t>(1, INV));
        const std::size_t chunk = (INV + Te - 1) / Te;

        auto worker = [&](int t) {
            const std::size_t row0 = std::size_t(t) * chunk;
            const std::size_t row1 = std::min(row0 + chunk, INV);

            for(std::size_t r = row0; r < row1; ++r)
            {
                // decode invariant multi-index; compute base offsets
                std::size_t tmp          = r;
                std::size_t base_in_off  = 0;
                std::size_t base_out_off = 0;
                for(std::size_t k = 0; k < invLens.size(); ++k)
                {
                    const std::size_t idx = (tmp / invRad[k]) % invLens[k];
                    base_in_off += idx * invStrides_v[k];
                    base_out_off += idx * outStrides[invariantDims[k]];
                }

                Reducer<compType> R(nanOpt, reduceOp, zeroV, withIdx);

                // iterate reduced subspace
                for(std::size_t i = 0; i < TR; ++i)
                {
                    std::size_t tmp2    = i;
                    std::size_t red_off = 0;
                    for(std::size_t k = 0; k < redLens.size(); ++k)
                    {
                        const std::size_t idx = (tmp2 / redRad[k]) % redLens[k];
                        red_off += idx * redStrides_v[k];
                    }

                    auto v = convert_type<compType>(input[base_in_off + red_off]);
                    PreUnaryOp(v);
                    R.step(v, static_cast<int>(i)); // flat index inside reduced subspace
                }

                PosUnaryOp(R.acc);
                if(alpha != 1.0f)
                    R.acc *= convert_type<compType>(alpha);
                if(beta != 0.0f)
                    R.acc +=
                        convert_type<compType>(output[base_out_off]) * convert_type<compType>(beta);

                res.data[base_out_off] = convert_type<Tref>(R.acc);
                if(withIdx)
                    res_indices.data[base_out_off] = R.idx;
            }
        };

        if(parallel)
        {
            miopen::par_for(static_cast<int>(Te), worker);
        }
        else
        {
            for(int te = 0; te < Te; ++te)
            {
                worker(te);
            }
        }
    }

    return {res, res_indices};
}

#endif
