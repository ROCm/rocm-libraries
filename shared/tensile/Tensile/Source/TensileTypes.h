/*******************************************************************************
 *
 * Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
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

#ifndef TENSILETYPES_H
#define TENSILETYPES_H
#include "tensile_bfloat16.h"
#include "tensile_float8_bfloat8.h"
#include <algorithm>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cstddef>

#include <hip/hip_runtime.h>
#define TensileStatus hipError_t
#define tensileStatusSuccess hipSuccess
// FIXME - steal hip error encodings since rocBLAS expects hip error codes to be
// returned...
#define tensileStatusFailure hipErrorUnknown
#define tensileStatusAssertFailure hipErrorRuntimeOther
#define TensileHalf _Float16
inline std::ostream& operator<<(std::ostream& os, const _Float16& dt)
{
    os << (float)(dt);
    return os;
}

template <typename t>
struct tensile_complex
{
    t x;
    t y;

    __host__ __device__ tensile_complex() = default;

    constexpr __host__ __device__ tensile_complex(t real, t imag = 0)
        : x{real}
        , y{imag}
    {
    }

    template <typename u, typename std::enable_if<std::is_arithmetic<u>{}>::type* = nullptr>
    tensile_complex<t>& operator=(const u a)
    {
        x = a;
        y = 0;
        return (*this);
    }
};

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const tensile_complex<T>& data)
{
    if(data.y >= 0)
    {
        os << data.x << "+" << data.y << "i";
    }
    else
    {
        os << data.x << data.y << "i";
    }

    return os;
}

template <typename T>
constexpr __host__ __device__ tensile_complex<T> operator+(tensile_complex<T> const& data)
{
    return data;
}

template <typename T>
constexpr __host__ __device__ tensile_complex<T> operator-(tensile_complex<T> const& data)
{
    return tensile_complex<T>{-data.x, -data.y};
}

template <typename T>
constexpr __host__ __device__ tensile_complex<T> operator+(tensile_complex<T> const& a,
                                                           tensile_complex<T> const& b)
{
    return tensile_complex<T>{a.x + b.x, a.y + b.y};
}

template <typename T>
constexpr __host__ __device__ tensile_complex<T> operator-(tensile_complex<T> const& a,
                                                           tensile_complex<T> const& b)
{
    return tensile_complex<T>{a.x - b.x, a.y - b.y};
}

template <typename T>
constexpr __host__ __device__ tensile_complex<T> operator*(tensile_complex<T> const& a,
                                                           tensile_complex<T> const& b)
{
    return tensile_complex<T>{a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}

template <typename T>
constexpr __host__ __device__ tensile_complex<T> operator/(tensile_complex<T> const& a,
                                                           tensile_complex<T> const& b)
{
    return tensile_complex<T>{(a.x * b.x + a.y * b.y) / (b.x * b.x + b.y * b.y),
                              (a.y * b.x - a.x * b.y) / (b.x * b.x + b.y * b.y)};
}

template <typename T>
constexpr __host__ __device__ tensile_complex<T>& operator+=(tensile_complex<T>&       a,
                                                             tensile_complex<T> const& b)
{
    return (a = a + b);
}

template <typename T>
constexpr __host__ __device__ tensile_complex<T>& operator-=(tensile_complex<T>&       a,
                                                             tensile_complex<T> const& b)
{
    return (a = a - b);
}

template <typename T>
constexpr __host__ __device__ tensile_complex<T>& operator*=(tensile_complex<T>&       a,
                                                             tensile_complex<T> const& b)
{
    return (a = a * b);
}

template <typename T>
constexpr __host__ __device__ tensile_complex<T>& operator/=(tensile_complex<T>&       a,
                                                             tensile_complex<T> const& b)
{
    return (a = a / b);
}
template <typename T>
constexpr __host__ __device__ bool operator==(tensile_complex<T> const& a,
                                              tensile_complex<T> const& b)
{
    return (a.x == b.x) && (a.y == b.y);
}

template <typename T>
constexpr __host__ __device__ bool operator!=(tensile_complex<T> const& a,
                                              tensile_complex<T> const& b)
{
    return (a.x != b.x) || (a.y != b.y);
}

using tensile_float_complex  = tensile_complex<float>;
using tensile_double_complex = tensile_complex<double>;

#define TensileComplexFloat tensile_float_complex
#define TensileComplexDouble tensile_double_complex

#define TensileInt8x4 uint32_t
#define TensileInt32 int32_t

/*******************************************************************************
 * tensileSetup
 ******************************************************************************/
TensileStatus tensileSetup();

/*******************************************************************************
 * tensileTeardown
 ******************************************************************************/
TensileStatus tensileTeardown();

/*******************************************************************************
 * tensileCheckStatus
 ******************************************************************************/
#define tensileStatusCheck(RET)                                           \
    {                                                                     \
        TensileStatus tensileCheckStatusTmp = RET;                        \
        if(tensileCheckStatusTmp != tensileStatusSuccess)                 \
        {                                                                 \
            fprintf(stderr,                                               \
                    "ERROR:  TensileStatusFailure %i on line %u of %s\n", \
                    tensileCheckStatusTmp,                                \
                    __LINE__,                                             \
                    __FILE__);                                            \
            abort();                                                      \
        }                                                                 \
    }

// class ProblemDims
// -  stores all dimensions of the problems (sizes and strides)
// TensileCreateLibrary.cpp will create a typedef for each specific problem, ie
// ProblemDims_Cijk_Ailk_Bljk_SB.
template <int FirstStride,
          int LastStrideD,
          int LastStrideC,
          int LastStrideA,
          int LastStrideB,
          int NumSizes>
class ProblemDims
{
public:
    using SizeType = unsigned int;

    // Constructor accepts variable number of sizes:
    template <typename... Ts>
    explicit ProblemDims(Ts... args)
    {
        init<NumSizes>(args...);
    };

    const SizeType strideD(int idx) const
    {
        return _dims[idx];
    }

    const SizeType strideC(int idx) const
    {
        return _dims[LastStrideD - FirstStride + idx];
    }

    const SizeType strideA(int idx) const
    {
        return _dims[LastStrideD - FirstStride + LastStrideC - FirstStride + idx];
    }

    const SizeType strideB(int idx) const
    {
        return _dims[LastStrideD - FirstStride + LastStrideC - FirstStride + LastStrideA
                     - FirstStride + idx];
    }

    const SizeType sizes(int idx = 0) const
    {
        return _dims[LastStrideD - FirstStride + LastStrideC - FirstStride + LastStrideA
                     - FirstStride + LastStrideB - FirstStride + idx];
    }

    static int numSizes()
    {
        return NumSizes;
    };

    std::ostream& print(std::ostream& os) const
    {
        for(int i = 0; i < NumSizes; i++)
        {
            if(i != 0)
            {
                os << ", ";
            };
            os << _dims[i];
        };
        return os;
    };

private:
    template <SizeType I, typename T>
    void init(T v)
    {
        _dims[NumSizes - I] = v;
    }

    template <SizeType I, typename T, typename... Ts>
    void init(T v, Ts... args)
    {
        _dims[NumSizes - I] = v;
        init<I - 1>(args...);
    }

private:
    // order: Dstride, Cstride, Astride, Bstrides, sizes
    SizeType _dims[LastStrideD - FirstStride + LastStrideC - FirstStride + LastStrideA - FirstStride
                   + LastStrideB - FirstStride + NumSizes];
};

// Base template for ProblemKey
// -  stores the sizes and strides
// -  supports hash generation and comparison for lookup
// TensileCreateLibrary.cpp will create a typedef for each specific problem, ie
// ProblemKey_Cijk_Ailk_Bljk_SB.
// Some templates below use a parm called ProblemKeyType which can be any of
// these generated types.
template <int NumSizes>
class ProblemKey
{
public:
    using SizeType = unsigned int;

    // Constructor accepts variable number of sizes:
    template <typename... Ts>
    explicit ProblemKey(Ts... args)
        : _db(0)
    {
        init<NumSizes - 1>(args...);
    }

    template <int FirstStride,
              int LastStrideD,
              int LastStrideC,
              int LastStrideA,
              int LastStrideB,
              int NumDimSizes>
    explicit ProblemKey(const ProblemDims<FirstStride,
                                          LastStrideD,
                                          LastStrideC,
                                          LastStrideA,
                                          LastStrideB,
                                          NumDimSizes>& pdims)
        : _db(0)
        , _equalStrides(true)
    {
        for(int i = 0; i < NumSizes; i++)
        {
            _sizes[i] = pdims.sizes(i);
        }

        _equalStrides
            = ((pdims.strideD(0) == pdims.strideC(0)) && (pdims.strideD(1) == pdims.strideC(1)));
    }

    bool operator<(const ProblemKey<NumSizes>& p) const
    {
        if(p._equalStrides != this->_equalStrides)
            return true;
        for(int i = 0; i < NumSizes; i++)
        {
            if(p._sizes[i] < this->_sizes[i])
                return true;
            else if(p._sizes[i] > this->_sizes[i])
                return false;
            // if equal, continue to next index
        }
        return false; // get here if all indices are equal
    };

    bool operator==(const ProblemKey<NumSizes>& p) const
    {
        if(p._equalStrides != this->_equalStrides)
            return false;
        for(int i = 0; i < NumSizes; i++)
        {
            if(p._sizes[i] != this->_sizes[i])
                return false;
        }
        return true;
    };

    size_t hash() const
    {
        size_t h = 0;
        for(int i = 0; i < NumSizes; i++)
        {
            h ^= _sizes[i] + 0x9b9773e99e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }

    const SizeType sizes(int i) const
    {
        return _sizes[i];
    };

    static int numSizes()
    {
        return NumSizes;
    };

    std::ostream& print(std::ostream& os) const
    {
        for(int i = 0; i < NumSizes; i++)
        {
            if(i != 0)
            {
                os << ", ";
            };
            os << _sizes[i];
        };
        return os;
    };

private:
    template <int I>
    void init(const SizeType& v)
    {
        _sizes[NumSizes - I - 1] = v;
    }

    template <int I, typename... Ts>
    void init(const SizeType& v, Ts... args)
    {
        _sizes[NumSizes - I - 1] = v;
        init<I - 1>(args...);
    }

private:
    // Data members:
    SizeType _sizes[NumSizes];
    bool     _equalStrides;
    uint32_t _db;
};

//-------------
// Distance Functions between two problem sizes - used to find nearest neighbor
// or closest solution Assumes p1 and p2 have same number of sizes
//-------------

template <class ProblemKeyType>
struct RatioDistance
{
    double operator()(const ProblemKeyType& p1, const ProblemKeyType& p2) const
    {
        double distance = 1.0;
        for(int i = 0; i < p1.numSizes(); i++)
        {
            distance += ::fabs(::log(double(p1.sizes(i)) / double(p2.sizes(i))));
        }
        return distance;
    }
};

template <class ProblemKeyType>
struct ManhattanDistance
{
    double operator()(const ProblemKeyType& p1, const ProblemKeyType& p2) const
    {
        double distance = 0;
        for(int i = 0; i < p1.numSizes(); i++)
        {
            distance += ::fabs(double(p1.sizes(i)) - double(p2.sizes(i)));
        }
        return distance;
    }
};

template <class ProblemKeyType>
struct EuclideanDistance
{
    double operator()(const ProblemKeyType& p1, const ProblemKeyType& p2) const
    {
        double distance = 0;
        for(int i = 0; i < p1.numSizes(); i++)
        {
            if(p1.sizes(i) > p2.sizes(i))
                distance += pow(p1.sizes(i) - p2.sizes(i), 2);
            else
                distance += pow(p2.sizes(i) - p1.sizes(i), 2);
        }
        // distance = sqrt(distance);
        return distance;
    }
};

template <class ProblemKeyType>
struct RandomDistance
{
    double operator()(const ProblemKeyType& p1, const ProblemKeyType& p2) const
    {
        return double(rand());
    }
};

// ProblemType
// Stores the different dimensions (free, index, batch) and other
// problem-specific info A single ProblemType may have many ProblemDims
// Combination of a ProblemType and ProblemDim defines a Problem
class ProblemType
{
public:
    ProblemType(const std::vector<unsigned>& indicesFree,
                const std::vector<unsigned>& indicesSummation,
                const std::vector<unsigned>& indicesBatch,
                const std::vector<unsigned>& indexAssignmentsA,
                const std::vector<unsigned>& indexAssignmentsB)
        : _indicesFree(indicesFree)
        , _indicesSummation(indicesSummation)
        , _indicesBatch(indicesBatch)
        , _indexAssignmentsA(indexAssignmentsA)
        , _indexAssignmentsB(indexAssignmentsB)
    {
    }

    unsigned lastSummationIdx() const
    {
        return _indicesSummation.back();
    };
    unsigned free0Idx() const
    {
        return _indicesFree[0];
    };
    unsigned free1Idx() const
    {
        return _indicesFree[1];
    };
    bool isBatchIdx(unsigned idx) const
    {
        return std::find(_indicesBatch.begin(), _indicesBatch.end(), idx) != _indicesBatch.end();
    };
    unsigned numIndicesC() const
    {
        return _indicesFree.size() + _indicesBatch.size();
    };

    const std::vector<unsigned> indexAssignmentsA() const
    {
        return _indexAssignmentsA;
    }
    const std::vector<unsigned> indexAssignmentsB() const
    {
        return _indexAssignmentsB;
    }

private:
    const std::vector<unsigned> _indicesFree;
    const std::vector<unsigned> _indicesSummation;
    const std::vector<unsigned> _indicesBatch;
    const std::vector<unsigned> _indexAssignmentsA;
    const std::vector<unsigned> _indexAssignmentsB;
};

// These are properties of the problemDims and problemType which drive the
// 'assertions'
//   - Solutions may be optimized so they only work for a subset of
//   ProblemProperties -
//   - At runtime, the ProblemProperties are computed for the given ProblemType
//   and
//     ProblemDims, and then checked against the Solution requirements.
// Must be checked by the runtime before launching the solution
struct ProblemProperties
{

    // Constructor used in solution tables
    ProblemProperties(unsigned summationElementMultiple,
                      unsigned free0ElementMultiple,
                      unsigned free1ElementMultiple,
                      int      approxSize,
                      bool     equalStrides,
                      bool     allBatchAStridesAreZero,
                      bool     allBatchBStridesAreZero)
        : _summationElementMultiple(summationElementMultiple)
        , _free0ElementMultiple(free0ElementMultiple)
        , _free1ElementMultiple(free1ElementMultiple)
        , _approxSize(approxSize)
        , _equalStrides(equalStrides)
        , _allBatchAStridesAreZero(allBatchAStridesAreZero)
        , _allBatchBStridesAreZero(allBatchBStridesAreZero)
        , _db(0)
    {
        const char* db = std::getenv("TENSILE_DB");
        if(db)
        {
            _db = strtol(db, nullptr, 0);
        }
    }

    // Constructor used to compute assertions for a specified problem size
    template <class ProblemDimsType>
    ProblemProperties(const ProblemDimsType& pdims, const ProblemType* props)
        : _db(0)
    {
        _summationElementMultiple = 1; // problem summation element multiple
        auto sumSize              = pdims.sizes(props->lastSummationIdx());
        if((sumSize & 0x7) == 0)
            _summationElementMultiple = 8;
        else if((sumSize & 0x3) == 0)
            _summationElementMultiple = 4;
        else if((sumSize & 0x1) == 0)
            _summationElementMultiple = 2;

        auto free0Size        = pdims.sizes(props->free0Idx());
        _free0ElementMultiple = 1; // problem free0 element multiple
        if((free0Size & 0x7) == 0)
            _free0ElementMultiple = 8;
        else if((free0Size & 0x3) == 0)
            _free0ElementMultiple = 4;
        else if((free0Size & 0x1) == 0)
            _free0ElementMultiple = 2;

        auto free1Size        = pdims.sizes(props->free1Idx());
        _free1ElementMultiple = 1; // problem free1 element multiple
        if((free1Size & 0x7) == 0)
            _free1ElementMultiple = 8;
        else if((free1Size & 0x3) == 0)
            _free1ElementMultiple = 4;
        else if((free1Size & 0x1) == 0)
            _free1ElementMultiple = 2;

        bool allBelow1 = true;
        bool anyBelow1 = false;
        bool anyBelow4 = false;
        for(int si = 0; si != pdims.numSizes(); si++)
        {
            if(!props->isBatchIdx(si))
            {
                auto size = pdims.sizes(si);
                if(size > 1)
                    allBelow1 = false;
                if(size == 1)
                    anyBelow1 = true;
                else if(size < 4)
                    anyBelow4 = true;
            }
        }
        if(allBelow1)
            _approxSize = 1; // really small
        else if(anyBelow1 || anyBelow4)
            _approxSize = 2; // one dim not big enough
        else
            _approxSize = 99; // big enough

        _equalStrides = true;
        // Would need to fix for UseInitialStrides, if UseInitialStrides==1 then
        // don't subtract 1
        for(int i = 0; i < props->numIndicesC() - 1; i++)
        {
            if(pdims.strideD(i) != pdims.strideC(i))
            {
                _equalStrides = false;
                break;
            }
        }

        _allBatchAStridesAreZero = 1;
        int strideIdx            = 0;
        for(auto& idx : props->indexAssignmentsA())
        {
            bool isb = props->isBatchIdx(idx);
            // Would need to fix for UseInitialStrides
            auto stride = strideIdx == 0 ? 1 : pdims.strideA(strideIdx - 1);
            if(isb && stride != 0)
            {
                _allBatchAStridesAreZero = 0;
            }
            strideIdx++;
        }

        _allBatchBStridesAreZero = 1;
        strideIdx                = 0;
        for(auto& idx : props->indexAssignmentsB())
        {
            bool isb = props->isBatchIdx(idx);
            // Would need to fix for UseInitialStrides
            auto stride = strideIdx == 0 ? 1 : pdims.strideB(strideIdx - 1);
            if(isb && stride != 0)
            {
                _allBatchBStridesAreZero = 0;
            }
            strideIdx++;
        }
    };

    // Returns True if this AsssertionProperties meet the requirements for the
    // specified soluition (this object represents the 'Problem')
    bool validForSolution(const ProblemProperties& solutionRequirements) const
    {
        bool rv
            = (this->_summationElementMultiple >= solutionRequirements._summationElementMultiple)
              && (this->_free0ElementMultiple >= solutionRequirements._free0ElementMultiple)
              && (this->_free1ElementMultiple >= solutionRequirements._free1ElementMultiple)
              && ((this->_approxSize) >= solutionRequirements._approxSize)
              && ((this->_equalStrides) == solutionRequirements._equalStrides)
              && ((this->_allBatchAStridesAreZero) >= solutionRequirements._allBatchAStridesAreZero)
              && ((this->_allBatchBStridesAreZero)
                  >= solutionRequirements._allBatchBStridesAreZero);

#if 0
      if(this->_db & 0x10)
      {
          if(!rv)
          {
#define check_print(name, oper)                                                                \
    if(!(this->name oper solutionRequirements.name))                                           \
    {                                                                                          \
        std::cout << #name << "failed: !(" << this->name << #oper << solutionRequirements.name \
                  << ")" << std::endl;                                                         \
    }

              std::cout << "Assertion failed:" << std::endl;
              check_print(_summationElementMultiple, >=);
              check_print(_free0ElementMultiple, >=);
              check_print(_free1ElementMultiple, >=);
              check_print(_approxSize, >=);
              check_print(_equalStrides, ==);
              check_print(_allBatchAStridesAreZero, >=);
              check_print(_allBatchBStridesAreZero, >=);
#undef check_print
          }
          else
          {
              std::cout << "Satisfied asserts!" << std::endl;
          }
      }
#endif

        return rv;
    }

    unsigned _summationElementMultiple;
    unsigned _free0ElementMultiple;
    unsigned _free1ElementMultiple;
    int      _approxSize;
    bool     _equalStrides;
    bool     _allBatchAStridesAreZero; // for problems: true if all batchA strides are
        // 0. Solutions: true if all batchA strides
        // must be 0 to run the kernel. Prob,Soln: x,0:
        // ok:  0,1: FAILED_ASSERT, 1,1: OK .  P>=S
    bool     _allBatchBStridesAreZero; // true if all batchB strides are 0
    uint32_t _db;
};

#endif
