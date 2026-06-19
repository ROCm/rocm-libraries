/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "DataInitialization.hpp"
#include "BatchPointerLayout.hpp"

#if HIPBLASLT_ENABLE_MXDATAGENERATOR
#include <mxDataGen.hpp>
#include "DataInitializationHelpers.hpp"
#endif
#include "TensorDataManipulation.hpp"
#include "TimingInstrumentation.hpp"
#include "Utility.hpp"
// #include "DataInitializationTyped.hpp"

#ifdef TENSILELITE_DATAINIT_TEST_HOOKS
#include "DataInitializationTestHooks.hpp"
#endif

#include <Tensile/Utils.hpp>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <limits>
#include <list>
#include <map>
#include <tuple>

#ifdef TENSILELITE_DATAINIT_TEST_HOOKS
namespace TensileLite::testing::detail
{
    namespace
    {
        thread_local size_t optionalAltAllocationCallsRemaining = 0;
        thread_local bool   optionalAltAllocationFailureArmed    = false;
    } // namespace

    void setOptionalAltAllocationFailureCountdown(size_t callsBeforeFailure)
    {
        optionalAltAllocationCallsRemaining = callsBeforeFailure;
        optionalAltAllocationFailureArmed    = true;
    }

    void clearOptionalAltAllocationFailure()
    {
        optionalAltAllocationCallsRemaining = 0;
        optionalAltAllocationFailureArmed    = false;
    }

    bool shouldFailOptionalAltAllocation()
    {
        if(!optionalAltAllocationFailureArmed)
            return false;

        if(optionalAltAllocationCallsRemaining == 0)
            return true;

        --optionalAltAllocationCallsRemaining;
        return false;
    }
} // namespace TensileLite::testing::detail
#endif

namespace
{
    using TensileLite::Client::RingPolicyInputs;

    RingPolicyInputs makeRingPolicyInputs(TensileLite::Client::po::variables_map const& args)
    {
        // Mirror ReferenceValidator::m_printAny; print-valids alone does not enable the
        // validation driver.
        bool const printAny = args["print-tensor-a"].as<bool>() || args["print-tensor-b"].as<bool>()
                              || args["print-tensor-c"].as<bool>()
                              || args["print-tensor-d"].as<bool>()
                              || args["print-tensor-ref"].as<bool>()
                              || args["print-tensor-bias"].as<bool>()
                              || args["print-tensor-amaxd"].as<bool>();

        return {args["num-benchmarks"].as<int>(),
                args["num-enqueues-per-sync"].as<int>(),
                args["max-enqueues-per-sync"].as<int>(),
                args["num-syncs-per-benchmark"].as<int>(),
                args["min-flops-per-sync"].as<size_t>(),
                args["num-elements-to-validate"].as<int>(),
                printAny};
    }
} // namespace

namespace TensileLite
{
    namespace Client
    {
        template <typename K, typename T, std::size_t MaxNumEntries = 128>
        class LRUCache
        {
            using Entries    = std::list<K>;
            using EntryTrack = std::pair<T, typename Entries::iterator>;
            using EntryMap   = std::map<K, EntryTrack>;

        public:
            template <typename... Args>
            std::pair<typename EntryMap::iterator, bool> emplace(const K& key, Args&&... args)
            {
                if(!entryMap.count(key))
                {
                    entries.push_back(key);
                    auto&& ret = entryMap.emplace(
                        key, std::make_pair(T(std::forward<Args>(args)...), --entries.end()));
                    while(entries.size() > MaxNumEntries)
                    {
                        auto& front = entries.front();
                        entryMap.erase(front);
                        entries.pop_front();
                    }
                    return ret;
                }
                else
                {
                    auto& track = entryMap.at(key);
                    track.first = T(std::forward<Args>(args)...);
                    entries.splice(entries.end(), entries, track.second);
                }
                return {entryMap.find(key), true};
            }

            size_t count(const K& key) const
            {
                return entryMap.count(key);
            }

            const T& at(const K& key) const
            {
                auto& track = entryMap.at(key);
                entries.splice(entries.end(), entries, track.second);
                return track.first;
            }

            T& at(const K& key)
            {
                auto& track = entryMap.at(key);
                entries.splice(entries.end(), entries, track.second);
                return track.first;
            }

            const K& back() const
            {
                return entries.back();
            }

        private:
            EntryMap entryMap;
            Entries  entries;
        };

        using BitWidth        = uint8_t;
        using Size            = uint64_t;
        using SwizzleCacheKey = std::tuple<BitWidth, Size, Size>;
        using SwizzleCacheVal = ::Tensor::Manipulation::Tensor;
        using SwizzleCache    = LRUCache<SwizzleCacheKey, SwizzleCacheVal>;
        static thread_local SwizzleCache g_swizzleCache;

        BitWidth toBitWidth(rocisa::DataType datatype)
        {
            switch(datatype)
            {
            case rocisa::DataType::Double:
                return 64;
            case rocisa::DataType::XFloat32:
            case rocisa::DataType::Float:
                return 32;
            case rocisa::DataType::Half:
            case rocisa::DataType::BFloat16:
                return 16;
            case rocisa::DataType::Int8:
            case rocisa::DataType::Float8_fnuz:
            case rocisa::DataType::BFloat8_fnuz:
            case rocisa::DataType::Float8BFloat8_fnuz:
            case rocisa::DataType::BFloat8Float8_fnuz:
            case rocisa::DataType::Float8:
            case rocisa::DataType::BFloat8:
            case rocisa::DataType::Float8BFloat8:
            case rocisa::DataType::BFloat8Float8:
            case rocisa::DataType::E8:
            case rocisa::DataType::E5M3:
                return 8;
            default:
                throw std::runtime_error("unsupported datatype");
            }
        }

        std::string ToString(InitMode mode)
        {
            switch(mode)
            {
            case InitMode::Zero:
                return "Zero";
            case InitMode::One:
                return "One";
            case InitMode::Two:
                return "Two";
            case InitMode::Random:
                return "Random";
            case InitMode::NaN:
                return "NaN";
            case InitMode::Inf:
                return "Inf";
            case InitMode::BadInput:
                return "BadInput";
            case InitMode::BadOutput:
                return "BadOutput";
            case InitMode::SerialIdx:
                return "SerialIdx";
            case InitMode::SerialDim0:
                return "SerialDim0";
            case InitMode::SerialDim1:
                return "SerialDim1";
            case InitMode::Identity:
                return "Identity";
            case InitMode::TrigSin:
                return "TrigSin";
            case InitMode::TrigCos:
                return "TrigCos";
            case InitMode::TrigAbsSin:
                return "TrigAbsSin";
            case InitMode::TrigAbsCos:
                return "TrigAbsCos";
            case InitMode::RandomNarrow:
                return "RandomNarrow";
            case InitMode::NegOne:
                return "NegOne";
            case InitMode::Max:
                return "Max";
            case InitMode::DenormMin:
                return "DenormMin";
            case InitMode::DenormMax:
                return "DenormMax";
            case InitMode::RandomNegPosLimited:
                return "RandomNegPosLimited";
            case InitMode::Free:
                return "Free";
            case InitMode::TrigIndSin:
                return "TrigIndSin";
            case InitMode::TrigIndCos:
                return "TrigIndCos";
            case InitMode::TrigIndAbsSin:
                return "TrigIndAbsSin";
            case InitMode::TrigIndAbsCos:
                return "TrigIndAbsCos";
            case InitMode::UniformLowPrecision:
                return "UniformLowPrecision";

            case InitMode::Count:
                break;
            }

            throw std::runtime_error(
                concatenate("Invalid InitMode value: ", static_cast<int>(mode)));
        }

        std::ostream& operator<<(std::ostream& stream, InitMode const& mode)
        {
            return stream << ToString(mode);
        }

        std::istream& operator>>(std::istream& stream, InitMode& mode)
        {
            std::string strValue;
            stream >> strValue;

            if(strValue == ToString(InitMode::Zero))
                mode = InitMode::Zero;
            else if(strValue == ToString(InitMode::One))
                mode = InitMode::One;
            else if(strValue == ToString(InitMode::Two))
                mode = InitMode::Two;
            else if(strValue == ToString(InitMode::Random))
                mode = InitMode::Random;
            else if(strValue == ToString(InitMode::NaN))
                mode = InitMode::NaN;
            else if(strValue == ToString(InitMode::Inf))
                mode = InitMode::Inf;
            else if(strValue == ToString(InitMode::BadInput))
                mode = InitMode::BadInput;
            else if(strValue == ToString(InitMode::BadOutput))
                mode = InitMode::BadOutput;
            else if(strValue == ToString(InitMode::SerialIdx))
                mode = InitMode::SerialIdx;
            else if(strValue == ToString(InitMode::SerialDim0))
                mode = InitMode::SerialDim0;
            else if(strValue == ToString(InitMode::SerialDim1))
                mode = InitMode::SerialDim1;
            else if(strValue == ToString(InitMode::Identity))
                mode = InitMode::Identity;
            else if(strValue == ToString(InitMode::TrigSin))
                mode = InitMode::TrigSin;
            else if(strValue == ToString(InitMode::TrigCos))
                mode = InitMode::TrigCos;
            else if(strValue == ToString(InitMode::TrigAbsSin))
                mode = InitMode::TrigAbsSin;
            else if(strValue == ToString(InitMode::TrigAbsCos))
                mode = InitMode::TrigAbsCos;
            else if(strValue == ToString(InitMode::RandomNarrow))
                mode = InitMode::RandomNarrow;
            else if(strValue == ToString(InitMode::NegOne))
                mode = InitMode::NegOne;
            else if(strValue == ToString(InitMode::Max))
                mode = InitMode::Max;
            else if(strValue == ToString(InitMode::DenormMin))
                mode = InitMode::DenormMin;
            else if(strValue == ToString(InitMode::DenormMax))
                mode = InitMode::DenormMax;
            else if(strValue == ToString(InitMode::RandomNegPosLimited))
                mode = InitMode::RandomNegPosLimited;
            else if(strValue == ToString(InitMode::TrigIndSin))
                mode = InitMode::TrigIndSin;
            else if(strValue == ToString(InitMode::TrigIndCos))
                mode = InitMode::TrigIndCos;
            else if(strValue == ToString(InitMode::TrigIndAbsSin))
                mode = InitMode::TrigIndAbsSin;
            else if(strValue == ToString(InitMode::TrigIndAbsCos))
                mode = InitMode::TrigIndAbsCos;
            else if(strValue == ToString(InitMode::UniformLowPrecision))
                mode = InitMode::UniformLowPrecision;
            else if(std::all_of(strValue.begin(), strValue.end(), isdigit))
            {
                int value = atoi(strValue.c_str());
                if(value >= 0 && value < static_cast<int>(InitMode::Count))
                    mode = static_cast<InitMode>(value);
                else
                    throw std::runtime_error(
                        concatenate("Can't convert ", strValue, " to InitMode."));
            }
            else
            {
                throw std::runtime_error(concatenate("Can't convert ", strValue, " to InitMode."));
            }

            return stream;
        }

        std::ostream& operator<<(std::ostream& stream, BoundsCheckMode const& mode)
        {
            std::string strValue;

            if(mode == BoundsCheckMode::Disable)
                strValue = "Disable";
            else if(mode == BoundsCheckMode::NaN)
                strValue = "NaN";
            else if(mode == BoundsCheckMode::GuardPageFront)
                strValue = "GuardPageFront";
            else if(mode == BoundsCheckMode::GuardPageBack)
                strValue = "GuardPageBack";
            else if(mode == BoundsCheckMode::GuardPageAll)
                strValue = "GuardPageAll";
            else
                throw std::runtime_error(
                    concatenate("Invalid BoundsCheckMode value: ", static_cast<int>(mode)));

            return stream << strValue;
        }

        std::istream& operator>>(std::istream& stream, BoundsCheckMode& mode)
        {
            std::string strValue;
            stream >> strValue;

            if(strValue == "Disable")
                mode = BoundsCheckMode::Disable;
            else if(strValue == "NaN")
                mode = BoundsCheckMode::NaN;
            else if(strValue == "GuardPageFront")
                mode = BoundsCheckMode::GuardPageFront;
            else if(strValue == "GuardPageBack")
                mode = BoundsCheckMode::GuardPageBack;
            else if(strValue == "GuardPageAll")
                mode = BoundsCheckMode::GuardPageAll;
            else if(std::all_of(strValue.begin(), strValue.end(), isdigit))
            {
                int value = atoi(strValue.c_str());
                if(value >= 0 && value < static_cast<int>(BoundsCheckMode::MaxMode))
                    mode = static_cast<BoundsCheckMode>(value);
                else
                    throw std::runtime_error(
                        concatenate("Can't convert ", strValue, " to BoundsCheckMode."));
            }
            else
            {
                throw std::runtime_error(
                    concatenate("Can't convert ", strValue, " to BoundsCheckMode."));
            }

            return stream;
        }

        void calculateKforSwizzling(rocisa::DataType datatype,
                                    size_t&          MiK,
                                    size_t&          MiKv,
                                    size_t&          PackK)
        {
            switch(datatype)
            {
            case rocisa::DataType::Float:
                MiK  = 4;
                MiKv = 1;
                break;
            case rocisa::DataType::Double:
                MiK  = 4;
                MiKv = 1;
                break;
            case rocisa::DataType::XFloat32:
                MiK  = 8;
                MiKv = 2;
                break;
            case rocisa::DataType::Half:
            case rocisa::DataType::BFloat16:
                MiK  = 16;
                MiKv = 4;
                break;
            case rocisa::DataType::Int8:
            case rocisa::DataType::Float8_fnuz:
            case rocisa::DataType::BFloat8_fnuz:
            case rocisa::DataType::Float8BFloat8_fnuz:
            case rocisa::DataType::BFloat8Float8_fnuz:
            case rocisa::DataType::Float8:
            case rocisa::DataType::BFloat8:
            case rocisa::DataType::Float8BFloat8:
            case rocisa::DataType::BFloat8Float8:
            case rocisa::DataType::E8:
            case rocisa::DataType::E5M3:
                MiK  = 32;
                MiKv = 8;
                break;
            default:
                throw std::runtime_error("unsupported datatype for swizzling");
            }

            PackK = 16 / MiKv / rocisa::GetElementSize(datatype);
        }

        template <typename T>
        std::shared_ptr<T> allocNewGPUBuffer(const char* title, size_t size)
        {
            static const int sizew = 10;
            T*               ptr   = nullptr;
            HIP_CHECK_EXC(hipMalloc(&ptr, size));
            auto p = std::shared_ptr<T>(ptr, hipFree);
            if(Debug::Instance().printTensorInfo())
                std::cout << "info: allocate " << title << " " << std::setw(sizew) << size
                          << " bytes at " << static_cast<void*>(ptr) << "\n";
            return p;
        }

        template <typename T>
        std::shared_ptr<T> tryAllocNewGPUBuffer(const char* title, size_t size)
        {
            static const int sizew = 10;

#ifdef TENSILELITE_DATAINIT_TEST_HOOKS
            if(TensileLite::testing::detail::shouldFailOptionalAltAllocation())
                return nullptr;
#endif

            T* ptr = nullptr;
            if(hipMalloc(&ptr, size) != hipSuccess)
                return nullptr;

            auto p = std::shared_ptr<T>(ptr, hipFree);
            if(Debug::Instance().printTensorInfo())
                std::cout << "info: allocate " << title << " " << std::setw(sizew) << size
                          << " bytes at " << static_cast<void*>(ptr) << "\n";
            return p;
        }

        template <typename T>
        void pruneSparseArray(PruneSparseMode         mode,
                              T*                      array,
                              TensorDescriptor const& tensor,
                              size_t                  pruneDim)
        {
            auto const& sizes        = tensor.sizes();
            auto        count        = CoordCount(sizes.begin(), sizes.end());
            size_t      pruneDimSize = sizes[pruneDim];
            size_t      loop_count   = count / pruneDimSize;
            if(pruneDimSize % 4 != 0)
                throw std::runtime_error("prune dimension size must be multiple of 4.");
            switch(mode)
            {
            case PruneSparseMode::PruneXX00:
            case PruneSparseMode::PruneX0X0:
            case PruneSparseMode::Prune0XX0:
            case PruneSparseMode::PruneX00X:
            case PruneSparseMode::Prune0X0X:
            case PruneSparseMode::Prune00XX:
            case PruneSparseMode::PruneRandom:
                break;
            default:
                throw std::runtime_error("prune mode is not allowed.");
                break;
            }

            constexpr std::array<uint8_t, static_cast<uint32_t>(PruneSparseMode::MaxPruneMode)>
                pruneMask = [] {
                    std::array<uint8_t, static_cast<uint32_t>(PruneSparseMode::MaxPruneMode)> m{};
                    m[static_cast<uint32_t>(PruneSparseMode::PruneXX00)] = 0x3;
                    m[static_cast<uint32_t>(PruneSparseMode::PruneX0X0)] = 0x5;
                    m[static_cast<uint32_t>(PruneSparseMode::Prune0XX0)] = 0x6;
                    m[static_cast<uint32_t>(PruneSparseMode::PruneX00X)] = 0x9;
                    m[static_cast<uint32_t>(PruneSparseMode::Prune0X0X)] = 0xA;
                    m[static_cast<uint32_t>(PruneSparseMode::Prune00XX)] = 0xC;
                    return m;
                }();

#pragma omp parallel
            {
                std::random_device                      rd;
                std::mt19937                            rng(rd());
                std::uniform_int_distribution<uint32_t> dist(
                    1, static_cast<uint32_t>(PruneSparseMode::MaxPruneMode) - 1);

#pragma omp for schedule(static)
                for(size_t loop = 0; loop < loop_count; loop++)
                {
                    std::vector<size_t> coord(tensor.dimensions(), 0);
                    CoordNumberedExclude(
                        loop, coord.begin(), coord.end(), sizes.begin(), sizes.end(), pruneDim);
                    for(size_t pruneDimIdx = 0; pruneDimIdx < pruneDimSize;
                        pruneDimIdx += 4) //traverse along pruneDim
                    {
                        uint32_t umode = static_cast<uint32_t>(mode);
                        if(umode == static_cast<uint32_t>(PruneSparseMode::PruneRandom))
                            umode = dist(rng);

                        uint32_t mask_ = pruneMask[umode];

                        coord[pruneDim] = pruneDimIdx;
                        uint32_t bit    = (mask_) & 0x1u;
                        if(!bit)
                            array[tensor.index(coord)] = T{};

                        coord[pruneDim] = pruneDimIdx + 1;
                        bit             = (mask_ >> 1) & 0x1u;
                        if(!bit)
                            array[tensor.index(coord)] = T{};

                        coord[pruneDim] = pruneDimIdx + 2;
                        bit             = (mask_ >> 2) & 0x1u;
                        if(!bit)
                            array[tensor.index(coord)] = T{};

                        coord[pruneDim] = pruneDimIdx + 3;
                        bit             = (mask_ >> 3) & 0x1u;
                        if(!bit)
                            array[tensor.index(coord)] = T{};
                    }
                }
            }
        }

        template <typename T>
        void compressSparseArray(T*                      dstCompressed,
                                 unsigned char*          dstMeta,
                                 T const*                src,
                                 TensorDescriptor const& tensor,
                                 TensorDescriptor const& tensorC,
                                 TensorDescriptor const& tensorMeta,
                                 size_t                  dim,
                                 bool                    metadataLayout)
        {
            auto const& sizes      = tensor.sizes();
            auto const& sizesC     = tensorC.sizes();
            auto        sizesMeta  = tensorMeta.sizes();
            auto        count      = CoordCount(sizes.begin(), sizes.end());
            size_t      dimSize    = sizes[dim];
            size_t      loop_count = count / dimSize;

            if(dimSize % 4 != 0)
                throw std::runtime_error("compressed dimension size must be multiple of 4.");

            std::memset((void*)dstCompressed, 0, tensorC.totalAllocatedBytes());
            std::memset((void*)dstMeta, 0, tensorMeta.totalAllocatedBytes());

#pragma omp parallel
            {
#pragma omp for schedule(static)
                for(size_t loop = 0; loop < loop_count; loop++)
                {
                    std::vector<size_t> coord(tensor.dimensions());
                    std::vector<size_t> coordC(tensorC.dimensions());
                    std::vector<size_t> coordMeta(tensorMeta.dimensions());
                    std::vector<size_t> _sizesMeta(tensorMeta.dimensions());
                    CoordNumberedExclude(
                        loop, coord.begin(), coord.end(), sizes.begin(), sizes.end(), dim);
                    CoordNumberedExclude(
                        loop, coordC.begin(), coordC.end(), sizesC.begin(), sizesC.end(), dim);
                    //metadata is always a tranpose matrix until we use metadataLayout now.
                    for(int i = 0; i < tensorMeta.dimensions(); i++)
                    {
                        _sizesMeta[i] = sizesMeta[i];
                    }

                    CoordNumberedExclude(loop,
                                         coordMeta.begin(),
                                         coordMeta.end(),
                                         _sizesMeta.begin(),
                                         _sizesMeta.end(),
                                         metadataLayout);
                    coordMeta[metadataLayout] = 0;

                    for(size_t compressDimIdx = 0; compressDimIdx < dimSize;
                        compressDimIdx += 4) //traverse along compressdim
                    {
                        uint32_t metaData = 0;
                        uint32_t metaIdx[2];

                        size_t dstDimCoord = compressDimIdx / 4 * 2;

                        coord[dim]  = compressDimIdx;
                        coordC[dim] = dstDimCoord;

                        T srcData[4];
                        srcData[0] = src[tensor.index(coord)];
                        coord[dim] = compressDimIdx + 1;
                        srcData[1] = src[tensor.index(coord)];
                        coord[dim] = compressDimIdx + 2;
                        srcData[2] = src[tensor.index(coord)];
                        coord[dim] = compressDimIdx + 3;
                        srcData[3] = src[tensor.index(coord)];

                        int nnz = (srcData[0] != T{}) + (srcData[1] != T{}) + (srcData[2] != T{})
                                  + (srcData[3] != T{});
                        if(nnz > 2)
                            throw std::runtime_error("Sparse matrix must contain 2 zero "
                                                     "elements of each 4 elements.");
                        //init metadata = 10
                        metaIdx[0] = 0;
                        metaIdx[1] = 1;

                        if(srcData[2] != T{})
                        {
                            if(srcData[1] != T{})
                            {
                                metaIdx[0] = 1;
                            }
                            metaIdx[1] = 2; //metadata = 20 or 21
                        }
                        if(srcData[3] != T{})
                        {

                            if(srcData[metaIdx[1]] != T{})
                            {
                                metaIdx[0] = metaIdx[1];
                            }
                            metaIdx[1] = 3; //metadata = 32 or 31 or 30
                        }

                        dstCompressed[tensorC.index(coordC)] = srcData[metaIdx[0]];
                        coordC[dim]                          = dstDimCoord + 1;
                        dstCompressed[tensorC.index(coordC)] = srcData[metaIdx[1]];
                        metaData                             = metaIdx[0] | (metaIdx[1] << 2);
                        //meta Data coord
                        size_t shift4bit = (compressDimIdx / 4 % 2) * 4;
                        coordMeta[metadataLayout]     = compressDimIdx / 8;
                        //calculate flatten index of dstMeta
                        size_t flattenIdx = CoordFlattenIndex(
                            coordMeta.begin(), coordMeta.end(), _sizesMeta.begin(), _sizesMeta.end());
                        // store metaData to dstMeta
                        dstMeta[flattenIdx] |= metaData << shift4bit;
                    }
                }
            }
        }

        template <>
        void compressSparseArray<Int8x4>(Int8x4*                 dstCompressed,
                                         unsigned char*          dstMeta,
                                         Int8x4 const*           src,
                                         TensorDescriptor const& tensor,
                                         TensorDescriptor const& tensorC,
                                         TensorDescriptor const& tensorMeta,
                                         size_t                  dim,
                                         bool                    metadataLayout)
        {
            throw std::runtime_error("SparseMatrix doesn't support Int8x4.");
        }

        template <typename T>
        void initCPUSparseInputTemplate(PruneSparseMode         mode,
                                        T*                      dstPruned,
                                        T*                      dstCompressed,
                                        unsigned char*          dstMeta,
                                        TensorDescriptor const& tensor,
                                        TensorDescriptor const& tensorC,
                                        TensorDescriptor const& tensorMeta,
                                        size_t                  dim,
                                        bool                    metadataLayout)
        {
            pruneSparseArray(mode, dstPruned, tensor, dim);
            compressSparseArray(
                dstCompressed, dstMeta, dstPruned, tensor, tensorC, tensorMeta, dim, metadataLayout);
        }

        void initCPUSparseInput(PruneSparseMode         mode,
                                void*                   dstPruned,
                                void*                   dstCompressed,
                                void*                   dstMeta,
                                TensorDescriptor const& tensor,
                                TensorDescriptor const& tensorC,
                                TensorDescriptor const& tensorMeta,
                                size_t                  dim,
                                bool                    metadataLayout)
        {

            //alloc compressed sparse buffer
            switch(tensor.dataType())
            {
            case rocisa::DataType::Half:
                initCPUSparseInputTemplate(mode,
                                           (Half*)(dstPruned),
                                           (Half*)(dstCompressed),
                                           (unsigned char*)(dstMeta),
                                           tensor,
                                           tensorC,
                                           tensorMeta,
                                           dim,
                                           metadataLayout);
                break;
            case rocisa::DataType::BFloat16:
                initCPUSparseInputTemplate(mode,
                                           (BFloat16*)(dstPruned),
                                           (BFloat16*)(dstCompressed),
                                           (unsigned char*)(dstMeta),
                                           tensor,
                                           tensorC,
                                           tensorMeta,
                                           dim,
                                           metadataLayout);
                break;
            case rocisa::DataType::Int8:
                initCPUSparseInputTemplate(mode,
                                           (int8_t*)(dstPruned),
                                           (int8_t*)(dstCompressed),
                                           (unsigned char*)(dstMeta),
                                           tensor,
                                           tensorC,
                                           tensorMeta,
                                           dim,
                                           metadataLayout);
                break;
            case rocisa::DataType::Float8:
                initCPUSparseInputTemplate(mode,
                                           (Float8*)(dstPruned),
                                           (Float8*)(dstCompressed),
                                           (unsigned char*)(dstMeta),
                                           tensor,
                                           tensorC,
                                           tensorMeta,
                                           dim,
                                           metadataLayout);
                break;
            case rocisa::DataType::BFloat8:
                initCPUSparseInputTemplate(mode,
                                           (BFloat8*)(dstPruned),
                                           (BFloat8*)(dstCompressed),
                                           (unsigned char*)(dstMeta),
                                           tensor,
                                           tensorC,
                                           tensorMeta,
                                           dim,
                                           metadataLayout);
                break;
            case rocisa::DataType::Float8_fnuz:
                initCPUSparseInputTemplate(mode,
                                           (Float8_fnuz*)(dstPruned),
                                           (Float8_fnuz*)(dstCompressed),
                                           (unsigned char*)(dstMeta),
                                           tensor,
                                           tensorC,
                                           tensorMeta,
                                           dim,
                                           metadataLayout);
                break;
            case rocisa::DataType::BFloat8_fnuz:
                initCPUSparseInputTemplate(mode,
                                           (BFloat8_fnuz*)(dstPruned),
                                           (BFloat8_fnuz*)(dstCompressed),
                                           (unsigned char*)(dstMeta),
                                           tensor,
                                           tensorC,
                                           tensorMeta,
                                           dim,
                                           metadataLayout);
                break;
            default:
                throw std::runtime_error("SparseMatrix doesn't support");
            }
        }

        void uploadBatchPointerLayout(void*                      base,
                                      void**                     array,
                                      BatchPointerLayout const&  layout,
                                      uint8_t**                  pinnedStaging,
                                      size_t                     pinnedStagingCapacity,
                                      hipStream_t                stream = nullptr)
        {
            size_t const count = layout.count();
            if(count > pinnedStagingCapacity)
                throw std::runtime_error("Batch pointer staging capacity is too small.");

            auto* baseBytes = static_cast<uint8_t*>(base);
            for(size_t idx = 0; idx < count; ++idx)
            {
                pinnedStaging[idx] = baseBytes + layout.offsets[idx];
            }

            if(stream)
                HIP_CHECK_EXC(hipMemcpyAsync(
                    array, pinnedStaging, count * sizeof(void*), hipMemcpyHostToDevice, stream));
            else
                HIP_CHECK_EXC(
                    hipMemcpy(array, pinnedStaging, count * sizeof(void*), hipMemcpyHostToDevice));
        }

        void* copyBadInputBuffers(const TensorDescriptor& descriptor,
                                  void*                   dst,
                                  void*                   src,
                                  void*                   bad,
                                  size_t                  totalElements,
                                  hipMemcpyKind           kind,
                                  hipStream_t             stream = nullptr)
        {
            // First, fill entire buffer with NaN/Inf sentinels from "bad" buffer
            auto bytes = multiplyElementSize(
                totalElements, DataTypeInfo::Get(descriptor.dataType()).elementSize);
            if(stream)
                HIP_CHECK_EXC(hipMemcpyAsync(dst, bad, bytes, kind, stream));
            else
                HIP_CHECK_EXC(hipMemcpy(dst, bad, bytes, kind));
            // Then, copy valid data to middle section, overwriting sentinel padding
            ptrdiff_t dPadding = totalElements - descriptor.totalAllocatedElements();
            dPadding           = multiplyElementSize(dPadding, descriptor.elementBytes());

            // Ensure dPadding/2 is properly aligned for the element type
            // Round dPadding to multiple of (2 * ceil(elementBytes)) to ensure:
            // 1. dPadding is even (so dPadding/2 is a whole number)
            // 2. dPadding/2 is aligned to element boundaries
            float elementBytes = descriptor.elementBytes();
            size_t alignmentBytes = 2 * static_cast<size_t>(std::ceil(std::max(1.0f, elementBytes)));
            dPadding = (dPadding / alignmentBytes) * alignmentBytes;

            void* dstOffset    = (void*)((uint8_t*)dst + dPadding / 2);
            TensileLite::hip::CopyTensorVoid(dstOffset, src, descriptor, kind, stream);
            return dstOffset;
        }

        void* copyNaNInputBuffers(const TensorDescriptor& descriptor,
                                  void*                   dst,
                                  void*                   src,
                                  size_t                  totalElements,
                                  hipMemcpyKind           kind,
                                  ptrdiff_t               customPadding = -1,
                                  hipStream_t             stream        = nullptr)
        {
            const ptrdiff_t dPadding = (customPadding == -1)
                                           ? totalElements - descriptor.totalAllocatedElements()
                                           : customPadding;
            const size_t    numElementsToCopy
                = (customPadding == -1) ? descriptor.totalAllocatedElements()
                                        : (descriptor.totalAllocatedElements() + customPadding);
            uint8_t* dstOffset
                = (uint8_t*)dst + multiplyElementSize(dPadding, descriptor.elementBytes());
            auto     bytes     = multiplyElementSize(numElementsToCopy, descriptor.elementBytes());
            if(stream)
                HIP_CHECK_EXC(hipMemcpyAsync(dstOffset, src, bytes, kind, stream));
            else
                HIP_CHECK_EXC(hipMemcpy(dstOffset, src, bytes, kind));
            return dstOffset;
        }

        void* copyInputBuffers(const TensorDescriptor& descriptor,
                               void*                   dst,
                               void*                   src,
                               size_t                  totalElements,
                               hipMemcpyKind           kind,
                               hipStream_t             stream = nullptr)
        {
            // If we have elements to copy, pointers must be valid
            // Null pointers with non-zero totalElements indicates a bug upstream (allocation logic)
            if(totalElements > 0 && (dst == nullptr || src == nullptr))
            {
                std::stringstream ss;
                ss << "Invalid state in copyInputBuffers: totalElements=" << totalElements
                   << " but dst=" << dst << " src=" << src
                   << " for tensor " << descriptor.getName();
                throw std::runtime_error(ss.str());
            }

            if(totalElements > 0)
            {
                auto bytes = multiplyElementSize(totalElements, descriptor.elementBytes());
                if(stream)
                    HIP_CHECK_EXC(hipMemcpyAsync(dst, src, bytes, kind, stream));
                else
                    HIP_CHECK_EXC(hipMemcpy(dst, src, bytes, kind));
            }
            return dst;
        }

        std::ostream& operator<<(std::ostream& stream, PruneSparseMode const& mode)
        {
            std::string strValue;

            if(mode == PruneSparseMode::PruneRandom)
                strValue = "PruneRandom";
            else if(mode == PruneSparseMode::PruneXX00)
                strValue = "PruneXX00";
            else if(mode == PruneSparseMode::PruneX0X0)
                strValue = "PruneX0X0";
            else if(mode == PruneSparseMode::Prune0XX0)
                strValue = "Prune0XX0";
            else if(mode == PruneSparseMode::PruneX00X)
                strValue = "PruneX00X";
            else if(mode == PruneSparseMode::Prune0X0X)
                strValue = "Prune0X0X";
            else if(mode == PruneSparseMode::Prune00XX)
                strValue = "Prune00XX";
            else
                throw std::runtime_error(
                    concatenate("Invalid PruneSparseMode value: ", static_cast<int>(mode)));

            return stream << strValue;
        }

        std::istream& operator>>(std::istream& stream, PruneSparseMode& mode)
        {
            std::string strValue;
            stream >> strValue;

            if(strValue == "PruneRandom")
                mode = PruneSparseMode::PruneRandom;
            else if(strValue == "PruneXX00")
                mode = PruneSparseMode::PruneXX00;
            else if(strValue == "PruneX0X0")
                mode = PruneSparseMode::PruneX0X0;
            else if(strValue == "Prune0XX0")
                mode = PruneSparseMode::Prune0XX0;
            else if(strValue == "PruneX00X")
                mode = PruneSparseMode::PruneX00X;
            else if(strValue == "Prune0X0X")
                mode = PruneSparseMode::Prune0X0X;
            else if(strValue == "Prune00XX")
                mode = PruneSparseMode::Prune00XX;
            else if(std::all_of(strValue.begin(), strValue.end(), isdigit))
            {
                int value = atoi(strValue.c_str());
                if(value >= 0 && value < static_cast<int>(PruneSparseMode::MaxPruneMode))
                    mode = static_cast<PruneSparseMode>(value);
                else
                    throw std::runtime_error(
                        concatenate("Can't convert ", strValue, " to PruneSparseMode."));
            }
            else
            {
                throw std::runtime_error(
                    concatenate("Can't convert ", strValue, " to PruneSparseMode."));
            }

            return stream;
        }

        size_t getSwizzledTensorNumAllocatedElements(const TensorDescriptor& desc,
                                                     size_t                  miM_N,
                                                     size_t                  miK,
                                                     size_t                  packK)
        {
            // TODO: currently [0][1] = k, (m or n) is based on TN, need to make this generic in the future
            const auto k         = desc.sizes()[0];
            const auto m_n       = desc.sizes()[1];
            const auto b         = desc.sizes()[2];
            const auto swizzleK  = miK * packK;
            const auto paddedM_N = (m_n + miM_N - 1) / miM_N * miM_N;
            const auto paddedK   = (k + swizzleK - 1) / swizzleK * swizzleK;
            return paddedM_N * paddedK * b;
        }

        size_t getSwizzledMXTensorNumAllocatedElements(const TensorDescriptor& desc,
                                                       size_t                  dimk,
                                                       bool                    unrollMajor)
        {
            const auto k    = unrollMajor ? desc.sizes()[0] : desc.sizes()[1];
            const auto m_n  = unrollMajor ? desc.sizes()[1] : desc.sizes()[0];
            const auto b    = desc.sizes()[2];
            const auto padk = (k + dimk - 1) / dimk * dimk;
            return padk * m_n * b;
        }

        double DataInitialization::GetRepresentativeBetaValue(po::variables_map const& args)
        {
            auto argValue = args["init-beta"].as<int>();

            if(argValue == 0)
                return 0.0;

            if(argValue == 1)
                return 1.0;

            return 1.5;
        }

        DataInitialization::DataInitialization(po::variables_map const&    args,
                                               ClientProblemFactory const& problemFactory)
            : m_maxBatch(0)
            , m_stridedBatched(args["strided-batched"].as<bool>())
            , m_sparse(args["sparse"].as<int>())
            , m_cEqualsD(args["c-equal-d"].as<bool>() || args["sparse"].as<int>())
            , m_elementsToValidate(args["num-elements-to-validate"].as<int>())
            , m_keepPristineCopyOnGPU(args["pristine-on-gpu"].as<bool>())
            , m_workspaceSize(problemFactory.workspaceSize())
            , m_pruneMode(args["prune-mode"].as<PruneSparseMode>())
            , m_mxScaleFormat(args["mx-scale-format"].as<int>())

        {
            if(m_mxScaleFormat > 0)
            {
                hipDeviceProp_t prop;
                int deviceIdx = args.count("device-idx") ? args["device-idx"].as<int>() : 0;
                hipGetDeviceProperties(&prop, deviceIdx);
                // gfx950 subtile kernels expect the preswizzled layout produced by
                // generateMXInput. All other architectures use the K-swizzle path.
                m_isMXPreswizzleArch
                    = (std::string(prop.gcnArchName).find("gfx950") != std::string::npos);
            }

            HIP_CHECK_EXC(hipStreamCreate(&m_copyStream));
            for(size_t i = 0; i < MAX_BUFFER_SETS; i++)
                HIP_CHECK_EXC(
                    hipEventCreateWithFlags(&m_copyDoneEvents[i], hipEventDisableTiming));

            // Determine ring policy from benchmark/validation settings.
            {
                auto const ringPolicyInputs = makeRingPolicyInputs(args);
                m_ringPolicy                = chooseRingPolicy(ringPolicyInputs);
                m_warmOutputResetRequired   = hasValidationDriver(ringPolicyInputs);
                m_ring                      = RingSlotController(m_ringPolicy.activeBufferCount);
            }

            m_rotatingBuffer
                = args["rotating-buffer-size"].as<int32_t>() * 1024 * 1024; // Change to bytes
            m_rotatingMode   = args["rotating-buffer-mode"].as<int32_t>();
            m_boundsCheck    = args["bounds-check"].as<BoundsCheckMode>();
            m_curBoundsCheck = m_boundsCheck;

            if(m_boundsCheck == BoundsCheckMode::GuardPageAll)
            {
                //GuardPageAll needs 2 runs per solution.
                //First run perform front side guard page checking.
                m_curBoundsCheck     = BoundsCheckMode::GuardPageFront;
                m_numRunsPerSolution = 2;
            }

            std::vector<std::vector<double>> activationAdditionalArgs;
            if(args.count("activation-additional-args"))
                activationAdditionalArgs
                    = args["activation-additional-args"].as<std::vector<std::vector<double>>>();

            if(problemFactory.problems().empty())
            {
                throw std::runtime_error("No problems in ProblemFactory.");
            }

            // Add switch cases here if needed. ex. GEMM, GEMM+GEMM

            // Get tensor info from problem factory.
            // TODO: Let ContractionProblemGroupedGemm use the same API as ContractionProblemGemm if possible.
            {
                auto const& p = problemFactory.problems()[0];
                if(auto ptr = dynamic_cast<ContractionProblemGroupedGemm const*>(p.get()))
                {
                    const ContractionProblemGroupedGemm& grouped = (*ptr);
                    if(problemFactory.problems().size() != 1)
                    {
                        throw std::runtime_error("Currently only supports one ContractionProblem "
                                                 "if grouped gemm is found in the ProblemFactory.");
                    }
                    m_vdata.resize(grouped.gemms[0].tensors().size());
                    m_cdata.resize(grouped.gemms[0].constants().size());
                }
                else
                {
                    m_vdata.resize(problemFactory.problems()[0]->tensors().size());
                    m_cdata.resize(problemFactory.problems()[0]->constants().size());
                }
            }

            for(auto const& p : problemFactory.problems())
            {
                if(auto ptr = dynamic_cast<ContractionProblemGemm const*>(p.get()))
                {
                    const ContractionProblemGemm& problem = (*ptr);
                    for(size_t i = 0; i < problem.tensors().size(); i++)
                    {
                        auto dataType = problem.tensors()[i].dataType();
                        if(m_vdata[i].pristine.find(dataType) == m_vdata[i].pristine.end())
                        {
                            m_vdata[i].pristine[dataType]             = PristineUnit();
                            m_vdata[i].pristine[dataType].maxElements = 0;
                        }
                        auto& pristine = m_vdata[i].pristine[dataType];
                        pristine.initDescriptor.resize(1);

                        auto numAllocatedElements = problem.tensors()[i].totalAllocatedElements();
                        auto numAllocatedBytes    = problem.tensors()[i].totalAllocatedBytes();

                        if((problem.swizzleTensorA() && i == ContractionProblemGemm::TENSOR::A)
                           || (problem.swizzleTensorB() && i == ContractionProblemGemm::TENSOR::B))
                        {
                            //TODO: support more swizzle types,
                            //      currently, if A then it means MiM = 16, if B then it means MiN = 16
                            size_t MiM_N = 16, MiK = 0, MiKv = 0, PackK = 0;
                            calculateKforSwizzling(dataType, MiK, MiKv, PackK);
                            numAllocatedElements = getSwizzledTensorNumAllocatedElements(
                                problem.tensors()[i], MiM_N, MiK, PackK);
                            numAllocatedBytes = multiplyElementSize(
                                numAllocatedElements, rocisa::GetElementSize(dataType));
                        }
                        if (i == ContractionProblemGemm::TENSOR::MXSA && problem.mxBlockA() != 0)
                        {
                            bool unrollMajor = (problem.freeIndicesA()[0].i != 0);
                            size_t MX = problem.mxBlockA();
                            size_t dimk = 128 / MX;
                            numAllocatedElements = getSwizzledMXTensorNumAllocatedElements(problem.tensors()[i], dimk, unrollMajor);
                        }
                        else if (i == ContractionProblemGemm::TENSOR::MXSB && problem.mxBlockB() != 0)
                        {
                            bool unrollMajor = (problem.freeIndicesB()[0].i != 0);
                            size_t MX = problem.mxBlockB();
                            size_t dimk = 128 / MX;
                            numAllocatedElements = getSwizzledMXTensorNumAllocatedElements(problem.tensors()[i], dimk, unrollMajor);
                        }

                        pristine.maxElements = std::max(pristine.maxElements, numAllocatedElements);

                        if(m_vdata[i].name.empty())
                        {
                            m_vdata[i].name = problem.tensors()[i].getName();
                        }
                        else if(m_vdata[i].name != problem.tensors()[i].getName())
                        {
                            std::string s = "Input tensor name " + problem.tensors()[i].getName()
                                            + " not match the pristine name " + m_vdata[i].name
                                            + " at index " + std::to_string(i) + ".";
                            throw std::runtime_error(s.c_str());
                        }
                    }
                    auto constants = problem.constants();
                    for(size_t i = 0; i < constants.size(); i++)
                    {
                        if(m_cdata[i].name.empty())
                        {
                            m_cdata[i].name = constants[i].name;
                        }
                        else if(m_cdata[i].name != constants[i].name)
                        {
                            std::string s = "Input constant name " + constants[i].name
                                            + " not match the pristine name " + m_cdata[i].name
                                            + " at index " + std::to_string(i) + ".";
                            throw std::runtime_error(s.c_str());
                        }
                    }

                    size_t numOfBatch = 1;
                    for(size_t i = 0; i < problem.batchIndices().size(); i++)
                        numOfBatch *= problem.batchSize(i);
                    m_maxBatch = std::max(m_maxBatch, numOfBatch);
                }
                else if(auto ptr = dynamic_cast<ContractionProblemGroupedGemm const*>(p.get()))
                {
                    const ContractionProblemGroupedGemm& problems = (*ptr);

                    struct gElement
                    {
                        size_t              maxElements;
                        std::vector<size_t> offsets;
                    };
                    auto gElements
                        = std::vector<std::map<rocisa::DataType, gElement>>(m_vdata.size());
                    for(auto const& problem : problems.gemms)
                    {
                        for(size_t i = 0; i < problem.tensors().size(); i++)
                        {
                            auto dataType = problem.tensors()[i].dataType();
                            if(m_vdata[i].pristine.find(dataType) == m_vdata[i].pristine.end())
                            {
                                m_vdata[i].pristine[dataType]             = PristineUnit();
                                m_vdata[i].pristine[dataType].maxElements = 0;
                            }
                            if(gElements[i].find(dataType) == gElements[i].end())
                            {
                                gElements[i][dataType].maxElements = 0;
                            }
                            auto& pristine = m_vdata[i].pristine[dataType];
                            pristine.initDescriptor.resize(problems.gemms.size());
                            gElements[i][dataType].maxElements
                                += problem.tensors()[i].totalAllocatedElements();
                            gElements[i][dataType].offsets.push_back(
                                problem.tensors()[i].totalAllocatedElements());
                            if(m_vdata[i].name.empty())
                            {
                                m_vdata[i].name = problem.tensors()[i].getName();
                            }
                            else if(m_vdata[i].name != problem.tensors()[i].getName())
                            {
                                std::string s = "Input tensor name "
                                                + problem.tensors()[i].getName()
                                                + " not match the pristine name " + m_vdata[i].name
                                                + " at index " + std::to_string(i) + ".";
                                throw std::runtime_error(s.c_str());
                            }
                        }
                        auto constants = problem.constants();
                        for(size_t i = 0; i < constants.size(); i++)
                        {
                            if(m_cdata[i].name.empty())
                            {
                                m_cdata[i].name = constants[i].name;
                            }
                            else if(m_cdata[i].name != constants[i].name)
                            {
                                std::string s = "Input constant name " + constants[i].name
                                                + " not match the pristine name " + m_cdata[i].name
                                                + " at index " + std::to_string(i) + ".";
                                throw std::runtime_error(s.c_str());
                            }
                        }

                        size_t numOfBatch = 1;
                        for(size_t i = 0; i < problem.batchIndices().size(); i++)
                            numOfBatch *= problem.batchSize(i);
                        m_maxBatch = std::max(m_maxBatch, numOfBatch);
                    }

                    // Update maxElements
                    for(size_t i = 0; i < gElements.size(); i++)
                    {
                        for(auto it : gElements[i])
                        {
                            auto& pristine = m_vdata[i].pristine[it.first];
                            pristine.maxElements
                                = std::max(pristine.maxElements, it.second.maxElements);
                            if(pristine.groupedGemmOffsets.empty())
                            {
                                pristine.groupedGemmOffsets = it.second.offsets;
                            }
                            else
                            {
                                if(pristine.groupedGemmOffsets.size() != it.second.offsets.size())
                                {
                                    throw std::runtime_error(
                                        "Unable to update groupedGemmOffsets.");
                                }
                                for(size_t j = 0; j < it.second.offsets.size(); j++)
                                {
                                    pristine.groupedGemmOffsets[j] = std::max(
                                        pristine.groupedGemmOffsets[j], it.second.offsets[j]);
                                }
                            }
                        }
                    }
                }
            }

            // Init tensors
            for(size_t i = 0; i < m_vdata.size(); i++)
            {
                std::string initName = "init-" + m_vdata[i].name;
                std::string typeName = m_vdata[i].name + "-type";
                if(args.count(initName))
                {
                    m_vdata[i].init = args[initName].as<InitMode>();
                }
                else if(m_vdata[i].name == "Synchronizer")
                {
                    m_vdata[i].init = InitMode::Zero;
                }
                else
                {
                    m_vdata[i].init = InitMode::Zero;
                }

                for(auto p = m_vdata[i].pristine.begin(); p != m_vdata[i].pristine.end();)
                {
                    // Remove pristine with maxElements = 0
                    if(p->second.maxElements == 0)
                    {
                        p = m_vdata[i].pristine.erase(p);
                        continue;
                    }

                    if(m_curBoundsCheck == BoundsCheckMode::NaN)
                    {
                        p->second.maxElements += 1024;
                    }
                    else if(m_curBoundsCheck == BoundsCheckMode::GuardPageFront
                            || m_curBoundsCheck == BoundsCheckMode::GuardPageBack)
                    {
                        float  dataTypeSize = DataTypeInfo::Get(p->first).elementSize;
                        size_t roundUpSize  = divideElementSize(pageSize, dataTypeSize);
                        p->second.maxElements
                            = RoundUpToMultiple<size_t>(p->second.maxElements, roundUpSize);
                        // No bias page guard
                    }
                    ++p;
                }
                std::cout << "Tensor name " << m_vdata[i].name << " init mode "
                          << ToString(m_vdata[i].init) << std::endl;
            }

            // Rotating buffer sizes must match post-bounds-check pristine.maxElements (e.g. guard
            // page round-up). vec_rm was previously built before that adjustment, undersizing pools.
            if(m_rotatingBuffer)
            {
                m_rm.reset();
                bool isRMInitPost = false;
                for(auto const& p : problemFactory.problems())
                {
                    if(auto ptr = dynamic_cast<ContractionProblemGemm const*>(p.get()))
                    {
                        std::vector<size_t>           vec_rm;
                        const ContractionProblemGemm& problem = *ptr;
                        for(size_t i = 0; i < problem.tensors().size(); i++)
                        {
                            if(i > ContractionProblemGemm::TENSOR::METADATA)
                                continue;
                            auto dataType = problem.tensors()[i].dataType();
                            auto  it      = m_vdata[i].pristine.find(dataType);
                            if(i == ContractionProblemGemm::TENSOR::C && problem.beta() == 0.0)
                            {
                                vec_rm.push_back(0);
                                continue;
                            }
                            if(it == m_vdata[i].pristine.end() || it->second.maxElements == 0)
                            {
                                vec_rm.push_back(0);
                                continue;
                            }
                            size_t const bytes = multiplyElementSize(
                                it->second.maxElements, DataTypeInfo::Get(dataType).elementSize);
                            vec_rm.push_back(bytes);
                        }
                        if(!isRMInitPost)
                        {
                            m_rm          = std::make_shared<RotatingMemory>(vec_rm.size());
                            isRMInitPost = true;
                        }
                        m_rm->addRotatingSize(vec_rm);
                    }
                    else if(auto ptr = dynamic_cast<ContractionProblemGroupedGemm const*>(p.get()))
                    {
                        const ContractionProblemGroupedGemm& grouped = *ptr;
                        std::vector<size_t>                    vec_rm;
                        for(auto const& problem : grouped.gemms)
                        {
                            std::vector<size_t> tmp_rm;
                            for(size_t i = 0; i < problem.tensors().size(); i++)
                            {
                                if(i > ContractionProblemGemm::TENSOR::METADATA)
                                    continue;
                                auto dataType = problem.tensors()[i].dataType();
                                auto  it      = m_vdata[i].pristine.find(dataType);
                                if(i == ContractionProblemGemm::TENSOR::C && problem.beta() == 0.0)
                                {
                                    tmp_rm.push_back(0);
                                    continue;
                                }
                                if(it == m_vdata[i].pristine.end() || it->second.maxElements == 0)
                                {
                                    tmp_rm.push_back(0);
                                    continue;
                                }
                                size_t const bytes = multiplyElementSize(
                                    it->second.maxElements, DataTypeInfo::Get(dataType).elementSize);
                                tmp_rm.push_back(bytes);
                            }
                            if(vec_rm.empty())
                            {
                                vec_rm = std::move(tmp_rm);
                            }
                            else
                            {
                                if(vec_rm.size() != tmp_rm.size())
                                {
                                    throw std::runtime_error("Unable to update vec_rm.");
                                }
                                for(size_t j = 0; j < tmp_rm.size(); j++)
                                {
                                    vec_rm[j] += tmp_rm[j];
                                }
                            }
                        }
                        if(!isRMInitPost)
                        {
                            m_rm          = std::make_shared<RotatingMemory>(vec_rm.size());
                            isRMInitPost = true;
                        }
                        m_rm->addRotatingSize(vec_rm);
                    }
                }
            }

            // Init contants
            for(size_t i = 0; i < m_cdata.size(); i++)
            {
                std::string initName = "init-" + m_cdata[i].name;
                m_cdata[i].dataType  = rocisa::DataType::None;
                // FIXME: Currently hardcoded
                if(m_cdata[i].name.find("activation") != std::string::npos)
                {
                    double value = 0.0;
                    if(activationAdditionalArgs.empty())
                    {
                        value = getValueWithUpperLowerBoundFP<double>(2.0, -2.0);
                    }
                    else
                    {
                        std::string            name   = m_cdata[i].name;
                        std::string            prefix = "activation-";
                        std::string::size_type pos    = name.find(prefix);

                        size_t index = -1;
                        if(pos != std::string::npos)
                        {
                            name.erase(pos, prefix.length());
                            index = greekToIndex(name);
                        }
                        // FIXME: Valgrind error: Invalid read of size 8
                        const auto& actArgs = activationAdditionalArgs[0];
                        value = (index >= actArgs.size()) ? actArgs[actArgs.size() - 1]
                                                          : actArgs[index];
                    }
                    m_cdata[i].freeValue = value;
                    m_cdata[i].init      = InitMode::Free;
                }
                else if(args.count(initName))
                {
                    m_cdata[i].init = args[initName].as<InitMode>();
                }
                else
                {
                    m_cdata[i].init = InitMode::Zero;
                }
                std::cout << "constant name " << m_cdata[i].name << " init mode "
                          << ToString(m_cdata[i].init) << std::endl;
            }

            // Need refactor, gemm a, b, c, d only
            m_problemDependentData = 0;
            for(size_t i = 0; i < 4; i++)
            {
                m_problemDependentData
                    = m_problemDependentData || IsProblemDependent(m_vdata[i].init);
            }
            m_problemDependentData
                |= (m_sparse
                    | (args["bias-type-args"].as<std::vector<rocisa::DataType>>().size() > 1));

            // Force problem-dependent initialization for MX FP4 to enable mxDataGenerator
            if(args.count("mx-a-block") && args["mx-a-block"].as<int>() > 0)
                m_problemDependentData = true;
            if(args.count("mx-b-block") && args["mx-b-block"].as<int>() > 0)
                m_problemDependentData = true;

            allocNewCPUInputs();
            allocNewGPUInputs();

            for(auto& it : m_vdata)
            {
                for(auto& p : it.pristine)
                {
                    auto  dataTypeSize = DataTypeInfo::Get(p.first).elementSize;
                    auto& pUnit        = p.second;
                    // Init and copy valid from cpu to gpu, only copies when != dependent data
                    if(!m_problemDependentData)
                    {

                        initArray(p.first, it.init, pUnit.cpuInput.valid.get(), pUnit.maxElements);
                        HIP_CHECK_EXC(
                            hipMemcpy(pUnit.gpuInput.valid.get(),
                                      pUnit.cpuInput.valid.get(),
                                      multiplyElementSize(pUnit.maxElements, dataTypeSize),
                                      hipMemcpyHostToDevice));
                    }
                    // Init and copy bad from cpu to gpu
                    if(pUnit.gpuInput.bad && pUnit.cpuInput.bad)
                    {
                        initArray(p.first,
                                  InitMode::BadOutput,
                                  pUnit.cpuInput.bad.get(),
                                  pUnit.maxElements);
                        HIP_CHECK_EXC(
                            hipMemcpy(pUnit.gpuInput.bad.get(),
                                      pUnit.cpuInput.bad.get(),
                                      multiplyElementSize(pUnit.maxElements, dataTypeSize),
                                      hipMemcpyHostToDevice));
                    }
                }
            }
        }

        void DataInitialization::allocNewCPUInputs()
        {
            for(auto& it : m_vdata)
            {
                for(auto& p : it.pristine)
                {
                    auto&  pUnit = p.second;
                    size_t size  = multiplyElementSize(pUnit.maxElements,
                                                      DataTypeInfo::Get(p.first).elementSize);
                    if(size <= 0)
                    {
                        throw std::runtime_error("Size not exists.");
                    }

                    std::stringstream ss;
                    ss << "Failed to allocate cpu input " << it.name << " type("
                       << DataTypeInfo::Get(p.first).abbrev
                       << "), element size: " << DataTypeInfo::Get(p.first).elementSize
                       << ", element length: " << pUnit.maxElements;

                    auto allocPinned = [](size_t bytes) {
                        void* raw = nullptr;
                        HIP_CHECK_EXC(hipHostMalloc(&raw, bytes, 0));
                        return std::shared_ptr<void>(raw, [](auto p) {
                            hipError_t e = hipHostFree(p);
                            if(e)
                                std::cerr << "hipHostFree failed: "
                                          << hipGetErrorString(e) << std::endl;
                        });
                    };

                    if(!pUnit.cpuInput.current)
                    {
                        auto ptr = allocPinned(size);
                        if(ptr == nullptr)
                        {
                            std::stringstream s;
                            s << "[input]" << ss.str();
                            throw std::runtime_error(s.str().c_str());
                        }
                        pUnit.cpuInput.current = ptr;
                    }
                    if(!pUnit.cpuInput.valid)
                    {
                        auto ptr = allocPinned(size);
                        if(ptr == nullptr)
                        {
                            std::stringstream s;
                            s << "[valid]" << ss.str();
                            throw std::runtime_error(s.str().c_str());
                        }
                        pUnit.cpuInput.valid = ptr;
                    }
                    if(!pUnit.cpuInput.bad && m_curBoundsCheck == BoundsCheckMode::NaN)
                    {
                        auto ptr = allocPinned(size);
                        if(ptr == nullptr)
                        {
                            std::stringstream s;
                            s << "[bad]" << ss.str();
                            throw std::runtime_error(s.str().c_str());
                        }
                        pUnit.cpuInput.bad = ptr;
                    }
                }
            }
            return;
        }

        void DataInitialization::rollbackAltGPUInputs() noexcept
        {
            for(auto& vd : m_vdata)
            {
                for(auto& [_, pUnit] : vd.pristine)
                {
                    for(size_t slot = 1; slot < MAX_BUFFER_SETS; ++slot)
                    {
                        pUnit.gpuInput.buffers[slot].reset();
                        pUnit.gpuInput.batchBufs[slot].reset();
                    }
                }
            }

            for(size_t slot = 1; slot < MAX_BUFFER_SETS; ++slot)
            {
                m_gpuPtrsRing[slot].clear();
                m_gpuBatchPtrsRing[slot].clear();
                m_cachedInputsRing[slot].reset();
            }
            m_altSlotsReady = false;
        }

        void DataInitialization::allocNewGPUInputs()
        {
            m_hasAltBuffers = m_ringPolicy.allocatesAltBuffers();

            m_pinnedBatchStagingBufferSlots = std::max<size_t>(1, m_ring.activeBufferCount());
            m_pinnedBatchStagingTensorSlots = m_vdata.size();

            auto checkedMultiply = [](size_t lhs, size_t rhs, char const* message) {
                if(lhs != 0 && rhs > (std::numeric_limits<size_t>::max() / lhs))
                    throw std::runtime_error(message);
                return lhs * rhs;
            };

            // Allocate reusable pinned staging buffer for batch pointer setup
            if(!m_pinnedBatchStaging && m_maxBatch > 0)
            {
                size_t totalPointers = checkedMultiply(
                    m_pinnedBatchStagingBufferSlots,
                    m_pinnedBatchStagingTensorSlots,
                    "[DataInitialization] pinned batch staging allocation overflow");
                totalPointers = checkedMultiply(
                    totalPointers,
                    m_maxBatch,
                    "[DataInitialization] pinned batch staging allocation overflow");
                size_t const bytes = checkedMultiply(
                    totalPointers,
                    sizeof(*m_pinnedBatchStaging),
                    "[DataInitialization] pinned batch staging allocation overflow");

                HIP_CHECK_EXC(
                    hipHostMalloc(&m_pinnedBatchStaging, bytes, 0));
            }

            auto disableAltBuffersAndRollback = [this]() noexcept {
                m_hasAltBuffers = false;
                rollbackAltGPUInputs();
            };

            struct AltAllocationRollback
            {
                decltype(disableAltBuffersAndRollback)& rollback;
                bool                                    active;

                AltAllocationRollback(
                    decltype(disableAltBuffersAndRollback)& rollbackFn, bool enabled) noexcept
                    : rollback(rollbackFn)
                    , active(enabled)
                {
                }

                ~AltAllocationRollback() noexcept
                {
                    if(active)
                        rollback();
                }

                void release() noexcept
                {
                    active = false;
                }

                void rollbackNow() noexcept
                {
                    if(active)
                    {
                        rollback();
                        active = false;
                    }
                }
            };

            AltAllocationRollback altAllocationGuard(
                disableAltBuffersAndRollback, m_hasAltBuffers);

            std::vector<std::shared_ptr<void>> guardPage;
            void*                              guardPagePtr;
            bool enableGuardPage = (m_curBoundsCheck == BoundsCheckMode::GuardPageFront
                                    || m_curBoundsCheck == BoundsCheckMode::GuardPageBack);
            std::shared_ptr<void> tmpPtr;
            if(m_rotatingBuffer > 0)
            {
                m_rm->createRotatingMemory(m_rotatingMode, m_rotatingBuffer);
            }

            size_t   offset    = 0;
            uint32_t tensorIdx = 0;
            for(auto& it : m_vdata)
            {
                for(auto& p : it.pristine)
                {
                    auto&  pUnit = p.second;
                    size_t size  = multiplyElementSize(pUnit.maxElements,
                                                      DataTypeInfo::Get(p.first).elementSize);

                    std::stringstream ss;
                    ss << "[" << tensorIdx << "]" << "Failed to allocate gpu input " << it.name
                       << " type(" << DataTypeInfo::Get(p.first).abbrev
                       << "), element size: " << DataTypeInfo::Get(p.first).elementSize
                       << ", element length: " << pUnit.maxElements;

                    if(!pUnit.gpuInput.current)
                    {
                        if(enableGuardPage)
                        {
                            HIP_CHECK_EXC(hipMalloc(&guardPagePtr, pageSize));
                            guardPage.push_back(std::shared_ptr<void>(guardPagePtr, hipFree));
                        }
                        std::shared_ptr<void> ptr;
                        if(m_rotatingBuffer)
                        {
                            auto mem = m_rm->getRotatingMemory();
                            if(tensorIdx <= ContractionProblemGemm::TENSOR::METADATA)
                                ptr = mem[0][tensorIdx].data;
                            else
                                ptr = allocNewGPUBuffer<void>(it.name.c_str(), size);
                        }
                        else
                        {
                            ptr = allocNewGPUBuffer<void>(it.name.c_str(), size);
                        }
                        if(ptr == nullptr)
                        {
                            std::stringstream s;
                            s << "[input gpu]" << ss.str();
                            throw std::runtime_error(s.str().c_str());
                        }
                        pUnit.gpuInput.current    = ptr;
                        pUnit.gpuInput.buffers[0] = ptr;
                        std::string n             = "batch" + it.name;
                        auto        batch_ptr
                            = allocNewGPUBuffer<void*>(n.c_str(), sizeof(uint8_t*) * m_maxBatch);
                        if(batch_ptr == nullptr)
                            throw std::runtime_error("out of batch gpu memory");
                        pUnit.gpuInput.batch       = batch_ptr;
                        pUnit.gpuInput.batchBufs[0] = batch_ptr;

                        // Allocate alternate buffers for multi-buffering
                        for(size_t slot = 1; m_hasAltBuffers && slot < m_ring.activeBufferCount();
                            slot++)
                        {
                            if(!pUnit.gpuInput.buffers[slot])
                            {
                                auto altSuffix = "_alt" + std::to_string(slot);
                                auto altPtr    = tryAllocNewGPUBuffer<void>(
                                    (it.name + altSuffix).c_str(), size);
                                if(!altPtr)
                                {
                                    altAllocationGuard.rollbackNow();
                                }
                                else
                                {
                                    auto altBatch = tryAllocNewGPUBuffer<void*>(
                                        (n + altSuffix).c_str(),
                                        sizeof(uint8_t*) * m_maxBatch);
                                    if(altBatch)
                                    {
                                        pUnit.gpuInput.buffers[slot]   = altPtr;
                                        pUnit.gpuInput.batchBufs[slot] = altBatch;
                                    }
                                    else
                                    {
                                        altAllocationGuard.rollbackNow();
                                    }
                                }
                            }
                        }
                    }
                    if(!pUnit.gpuInput.valid)
                    {
                        if(enableGuardPage)
                        {
                            HIP_CHECK_EXC(hipMalloc(&guardPagePtr, pageSize));
                            guardPage.push_back(std::shared_ptr<void>(guardPagePtr, hipFree));
                        }
                        auto ptr = allocNewGPUBuffer<void>(it.name.c_str(), size);
                        if(ptr == nullptr)
                        {
                            std::stringstream s;
                            s << "[valid]" << ss.str();
                            throw std::runtime_error(s.str().c_str());
                        }
                        pUnit.gpuInput.valid = ptr;
                    }
                    if(!pUnit.gpuInput.bad)
                    {
                        if(enableGuardPage)
                        {
                            HIP_CHECK_EXC(hipMalloc(&guardPagePtr, pageSize));
                            guardPage.push_back(std::shared_ptr<void>(guardPagePtr, hipFree));
                        }
                        auto ptr = allocNewGPUBuffer<void>(it.name.c_str(), size);
                        if(ptr == nullptr)
                        {
                            std::stringstream s;
                            s << "[bad]" << ss.str();
                            throw std::runtime_error(s.str().c_str());
                        }
                        pUnit.gpuInput.bad = ptr;
                    }
                }
                tensorIdx++;
            }

            if(!m_workspacePristine)
            {
                std::shared_ptr<void> ptr = nullptr;
                if(m_workspaceSize > 0)
                {
                    ptr = allocNewGPUBuffer<void>("ws", m_workspaceSize);
                    if(ptr == nullptr)
                        throw std::runtime_error(
                            "out of gpu memory while allocating workspace size");
                }
                m_workspacePristine = ptr;
            }

            altAllocationGuard.release();
        }

        DataInitialization::BatchPointerSignature
            DataInitialization::makeBatchPointerSignature(
                ContractionProblemGemm const& problem) const
        {
            BatchPointerSignature signature;
            signature.base.boundsCheck = m_curBoundsCheck;
            signature.base.tensors     = problem.tensors();
            signature.base.useBias     = problem.useBias();
            signature.base.biasSrc     = problem.biasSrc();
            signature.base.sparse      = problem.sparse();
            signature.base.swizzleA    = problem.swizzleTensorA();
            signature.base.swizzleB    = problem.swizzleTensorB();

            auto const& batchIndices = problem.batchIndices();
            signature.base.batchIndices.reserve(batchIndices.size());
            for(auto const& batchIndex : batchIndices)
            {
                signature.base.batchIndices.push_back(
                    {batchIndex.a, batchIndex.b, batchIndex.c, batchIndex.d});
            }

            return signature;
        }

        bool DataInitialization::batchPointersCurrentFor(
            ContractionProblemGemm const& problem) const
        {
            return m_batchPointerSignatureValid
                && m_batchPointerSignature == makeBatchPointerSignature(problem);
        }

        void DataInitialization::markBatchPointersCurrent(
            ContractionProblemGemm const& problem)
        {
            m_batchPointerSignature      = makeBatchPointerSignature(problem);
            m_batchPointerSignatureValid = true;
        }

        DataInitialization::PreparedProblemSignature
            DataInitialization::makePreparedProblemSignature(
                ContractionProblemGemm const& problem) const
        {
            PreparedProblemSignature signature;
            signature.base     = makeBatchPointerSignature(problem).base;
            signature.mxBlockA = problem.mxBlockA();
            signature.mxBlockB = problem.mxBlockB();
            return signature;
        }

        bool DataInitialization::gpuInputsPreparedFor(ContractionProblemGemm const& problem) const
        {
            return m_gpuInit && m_preparedProblemSignatureValid
                && m_preparedProblemSignature == makePreparedProblemSignature(problem);
        }

        void DataInitialization::markGpuInputsPrepared(
            ContractionProblemGemm const& problem)
        {
            m_preparedProblemSignature      = makePreparedProblemSignature(problem);
            m_preparedProblemSignatureValid = true;
        }

        void DataInitialization::initializeGPUBatchedInputs(ContractionProblemGemm const& problem,
                                                            hipStream_t                   targetStream,
                                                            size_t                        stagingBufferSlot)
        {
            auto const& batchIdxs = problem.batchIndices();
            // FIXME: batch not supported for bias
            for(size_t i = 0; i < 4 /*m_vdata.size()*/; i++)
            {
                auto const& tensor = problem.tensors()[i];
                auto        it     = m_vdata[i].pristine.find(tensor.dataType());
                if(it == m_vdata[i].pristine.end())
                    continue;
                auto&               pUnit     = it->second;
                auto const          batchIdx  = batchPointerTensorBatchIndices(
                    batchIdxs, static_cast<ContractionProblemGemm::TENSOR>(i));
                ptrdiff_t           padding   = 0;
                auto const          layout    = makeBatchPointerLayout(tensor, batchIdx);
                auto*               offsetBase = static_cast<uint8_t*>(pUnit.gpuInput.current.get());
                if(m_curBoundsCheck == BoundsCheckMode::NaN)
                {
                    padding = (pUnit.maxElements - tensor.totalAllocatedElements()) / 2;
                }
                else if(m_curBoundsCheck == BoundsCheckMode::GuardPageBack)
                {
                    padding = pUnit.maxElements - tensor.totalAllocatedElements();

                    if((problem.swizzleTensorA() && i == ContractionProblemGemm::TENSOR::A)
                       || (problem.swizzleTensorB() && i == ContractionProblemGemm::TENSOR::B))
                    {
                        //TODO: support more swizzle types,
                        //      currently, if A then it means MiM = 16, if B then it means MiN = 16
                        size_t MiM_N = 16, MiK = 0, MiKv = 0, PackK = 0;
                        calculateKforSwizzling(tensor.dataType(), MiK, MiKv, PackK);
                        padding = pUnit.maxElements
                                  - getSwizzledTensorNumAllocatedElements(tensor,
                                                                           MiM_N,
                                                                           MiK,
                                                                           PackK);
                    }
                }
                padding = multiplyElementSize(
                    padding, DataTypeInfo::Get(tensor.dataType()).elementSize);
                uploadBatchPointerLayout(static_cast<void*>(offsetBase + padding),
                                         pUnit.gpuInput.batch.get(),
                                         layout,
                                         pinnedBatchStagingSlice(stagingBufferSlot, i),
                                         m_maxBatch,
                                         targetStream);

                if(problem.useBias() && problem.biasSrc() == i)
                {
                    auto const& biasTensor = problem.tensors()[ContractionProblemGemm::TENSOR::BIAS];
                    auto& pUnitBias = m_vdata[ContractionProblemGemm::TENSOR::BIAS]
                                          .pristine[problem.bias().dataType()];
                    if(m_curBoundsCheck == BoundsCheckMode::NaN)
                    {
                        padding = (pUnitBias.maxElements
                                   - biasTensor.totalAllocatedElements())
                                  / 2;
                    }
                    else if(m_curBoundsCheck == BoundsCheckMode::GuardPageBack)
                    {
                        padding = pUnitBias.maxElements
                                  - biasTensor.totalAllocatedElements();
                    }
                    padding = multiplyElementSize(
                        padding, DataTypeInfo::Get(biasTensor.dataType()).elementSize);
                    auto const biasLayout = makeBatchPointerLayout(biasTensor, batchIdx);
                    uploadBatchPointerLayout(static_cast<void*>(
                                                 static_cast<uint8_t*>(pUnitBias.gpuInput.current.get())
                                                 + padding),
                                        pUnitBias.gpuInput.batch.get(),
                                        biasLayout,
                                        pinnedBatchStagingSlice(
                                            stagingBufferSlot,
                                            ContractionProblemGemm::TENSOR::BIAS),
                                        m_maxBatch,
                                        targetStream);
                }

                if((problem.sparse() == 1 && i == ContractionProblemGemm::TENSOR::A)
                   || (problem.sparse() == 2 && i == ContractionProblemGemm::TENSOR::B))
                {
                    auto caculate_padding = [](BoundsCheckMode mode, auto& p, auto& t) {
                        ptrdiff_t padding = 0;
                        if(mode == BoundsCheckMode::NaN)
                        {
                            padding = (p.maxElements - t.totalAllocatedElements()) / 2;
                        }
                        else if(mode == BoundsCheckMode::GuardPageBack)
                        {
                            padding = p.maxElements - t.totalAllocatedElements();
                        }
                        padding = multiplyElementSize(padding,
                                                      DataTypeInfo::Get(t.dataType()).elementSize);
                        return padding;
                    };

                    auto const& metadataTensor = problem.tensors()[ContractionProblemGemm::TENSOR::METADATA];
                    auto& pUnitM = m_vdata[ContractionProblemGemm::TENSOR::METADATA]
                                       .pristine[problem.metadata().dataType()];

                    padding = caculate_padding(
                        m_curBoundsCheck,
                        pUnitM,
                        metadataTensor);
                    auto const metadataLayout = makeBatchPointerLayout(metadataTensor, batchIdx);
                    uploadBatchPointerLayout(static_cast<void*>(
                                                 static_cast<uint8_t*>(pUnitM.gpuInput.current.get())
                                                 + padding),
                                        pUnitM.gpuInput.batch.get(),
                                        metadataLayout,
                                        pinnedBatchStagingSlice(
                                            stagingBufferSlot,
                                            ContractionProblemGemm::TENSOR::METADATA),
                                        m_maxBatch,
                                        targetStream);

                    auto const& compressedTensor
                        = problem.tensors()[ContractionProblemGemm::TENSOR::COMPRESSED];
                    auto& pUnitCp = m_vdata[ContractionProblemGemm::TENSOR::COMPRESSED]
                                        .pristine[problem.compressed().dataType()];
                    padding = caculate_padding(
                        m_curBoundsCheck,
                        pUnitCp,
                        compressedTensor);
                    auto const compressedLayout = makeBatchPointerLayout(compressedTensor, batchIdx);
                    uploadBatchPointerLayout(
                        static_cast<void*>(static_cast<uint8_t*>(pUnitCp.gpuInput.current.get())
                                           + padding),
                        pUnitCp.gpuInput.batch.get(),
                        compressedLayout,
                        pinnedBatchStagingSlice(stagingBufferSlot,
                                                ContractionProblemGemm::TENSOR::COMPRESSED),
                        m_maxBatch,
                        targetStream);
                }
            }
        }

        void DataInitialization::initializeCPUInputs(ContractionProblemGroupedGemm const& problem)
        {
            for(size_t i = 0; i < m_vdata.size(); i++)
            {
                if(m_problemDependentData)
                {
                    if(i == ContractionProblemGemm::TENSOR::COMPRESSED
                       or i == ContractionProblemGemm::TENSOR::METADATA)
                        continue;
                    // Should this m_cEqualsD set in ContractionProblem or boost args?
                    for(auto& p : m_vdata[i].pristine)
                    {
                        uint64_t gemmInitOffset = 0;
                        for(size_t j = 0; j < problem.gemms.size(); j++)
                        {
                            auto& tensors = problem.gemms[j].tensors();
                            bool const primaryStale = p.second.initDescriptor[j] != tensors[i];
                            if(primaryStale)
                            {
                                p.second.initDescriptor[j] = tensors[i];
                                initArray(p.first,
                                          m_vdata[i].init,
                                          (void*)((int8_t*)p.second.cpuInput.valid.get()
                                                  + gemmInitOffset),
                                          tensors[i]);
                            }

                            // FIXME: Should we init unused part to 0?
                            if((problem.gemms[j].sparse() == 1
                                && i == ContractionProblemGemm::TENSOR::A)
                               || (problem.gemms[j].sparse() == 2
                                   && i == ContractionProblemGemm::TENSOR::B))
                            {
                                const TensorDescriptor& t = problem.gemms[j].sparse() == 2
                                                                ? problem.gemms[j].b()
                                                                : problem.gemms[j].a();
                                int                     tDim;
                                if(problem.gemms[j].sparse() == 2)
                                    tDim = problem.gemms[j].boundIndices()[0].b;
                                else
                                    tDim = problem.gemms[j].boundIndices()[0].a;

                                const TensorDescriptor& tM = problem.gemms[j].metadata();
                                const TensorDescriptor& tC = problem.gemms[j].compressed();
                                auto& pUnitM = m_vdata[ContractionProblemGemm::TENSOR::METADATA]
                                                   .pristine[tM.dataType()];
                                auto& pUnitCp
                                    = m_vdata[ContractionProblemGemm::TENSOR::COMPRESSED]
                                          .pristine[tC.dataType()];
                                if(pUnitM.initDescriptor.size() <= j)
                                    pUnitM.initDescriptor.resize(problem.gemms.size());
                                if(pUnitCp.initDescriptor.size() <= j)
                                    pUnitCp.initDescriptor.resize(problem.gemms.size());

                                bool const sparseSideStale
                                    = pUnitM.initDescriptor[j] != tM
                                      || pUnitCp.initDescriptor[j] != tC;
                                if(primaryStale || sparseSideStale)
                                {
                                    pUnitM.initDescriptor[j] = tM;
                                    pUnitCp.initDescriptor[j] = tC;
                                    initCPUSparseInput(
                                        m_pruneMode,
                                        (char*)p.second.cpuInput.valid.get() + gemmInitOffset,
                                        (char*)pUnitCp.cpuInput.valid.get() + gemmInitOffset,
                                        (char*)pUnitM.cpuInput.valid.get() + gemmInitOffset,
                                        t,
                                        tC,
                                        tM,
                                        tDim,
                                        problem.gemms[j].metadataLayout());
                                }
                            }
                            gemmInitOffset += multiplyElementSize(p.second.groupedGemmOffsets[j],
                                                                  tensors[i].elementBytes());
                        }
                    }
                }
            }
        }

        void DataInitialization::initializeCPUInputs(ContractionProblemGemm const& problem)
        {
            // Only the gfx950 subtile MX kernels need the mxDataGenerator (DGen) seeding
            // of A/B and pre-swizzled E8 scales. Architectures that read canonical scales
            // (e.g. gfx1250) must use the same plain initArray path develop uses, so the
            // bytes the kernel sees are identical to the bytes the reference reads. We
            // gate on m_mxScaleFormat > 0 because that is the user-visible signal that
            // they opted into the subtile / pre-swizzle layout.
            bool useMXGenerator = isMXProblemExceptF6(problem) && m_mxScaleFormat > 0;
            if(useMXGenerator)
                initializeMXData(problem);

            auto& tensors = problem.tensors();
            for(size_t i = 0; i < m_vdata.size(); i++)
            {
                if(i == ContractionProblemGemm::TENSOR::COMPRESSED
                   or i == ContractionProblemGemm::TENSOR::METADATA)
                    continue;

                if(useMXGenerator && (i == ContractionProblemGemm::TENSOR::A
                                      || i == ContractionProblemGemm::TENSOR::B
                                      || i == ContractionProblemGemm::TENSOR::MXSA
                                      || i == ContractionProblemGemm::TENSOR::MXSB))
                    continue;

                if(m_problemDependentData)
                {
                    // Should this m_cEqualsD set in ContractionProblem or boost args?
                    for(auto& p : m_vdata[i].pristine)
                    {
                        // Only update when the descriptor changed
                        if(p.second.initDescriptor[0] != tensors[i])
                        {
                            p.second.initDescriptor[0] = tensors[i];
                            initArray(p.first,
                                      m_vdata[i].init,
                                      p.second.cpuInput.valid.get(),
                                      tensors[i]);
                            if((problem.sparse() == 1 && i == ContractionProblemGemm::TENSOR::A)
                               || (problem.sparse() == 2 && i == ContractionProblemGemm::TENSOR::B))
                            {
                                const TensorDescriptor& t
                                    = problem.sparse() == 2 ? problem.b() : problem.a();
                                int              tDim;
                                rocisa::DataType tDataType;
                                if(problem.sparse() == 2)
                                {
                                    tDim      = problem.boundIndices()[0].b;
                                    tDataType = problem.b().dataType();
                                }
                                else
                                {
                                    tDim      = problem.boundIndices()[0].a;
                                    tDataType = problem.a().dataType();
                                }

                                const TensorDescriptor& tM = problem.metadata();
                                const TensorDescriptor& tC = problem.compressed();
                                auto& pUnitM = m_vdata[ContractionProblemGemm::TENSOR::METADATA]
                                                   .pristine[problem.metadata().dataType()];
                                auto& pUnitCp = m_vdata[ContractionProblemGemm::TENSOR::COMPRESSED]
                                                    .pristine[p.first];
                                pUnitM.initDescriptor[0]
                                    = tensors[ContractionProblemGemm::TENSOR::METADATA];
                                pUnitCp.initDescriptor[0]
                                    = tensors[ContractionProblemGemm::TENSOR::COMPRESSED];

                                initCPUSparseInput(m_pruneMode,
                                                   p.second.cpuInput.valid.get(),
                                                   pUnitCp.cpuInput.valid.get(),
                                                   pUnitM.cpuInput.valid.get(),
                                                   t,
                                                   tC,
                                                   tM,
                                                   tDim,
                                                   problem.metadataLayout());
                            }
                        }
                    }
                }
            }
        }

#if HIPBLASLT_ENABLE_MXDATAGENERATOR

        using namespace detail;

        static std::string_view initModeToMXMethod(InitMode mode)
        {
            switch(mode)
            {
            case InitMode::Zero:
                return "Zeros";
            case InitMode::One:
                return "Ones";
            case InitMode::Identity:
                return "Identity";
            case InitMode::SerialIdx:
            case InitMode::SerialDim0:
            case InitMode::SerialDim1:
                return "Sequential";
            default:
                return "Bounded";
            }
        }

        // generateMXInput emits scales packed for the unpadded data K, but setMXScaleA/B
        // pad ceil(K/mxBlock) up to a multiple of 8. When those differ (e.g. K=384 →
        // 12 padded to 16) the kernel and CPU reference read every (m>0, k_block) at the
        // wrong byte. Only the K-fast layouts (bound dim at index 0 → TN A / NT B) need
        // this: K-slow layouts keep K-blocks as the slow axis and the unfilled padding
        // tail is already zero from the pre-memset. Walk the free axis backward so the
        // expansion can happen in place.
        static void restrideMXScaleBufferKFast(uint8_t* buffer,
                                               size_t   compactFreeDim,
                                               size_t   compactKBlocks,
                                               size_t   paddedKBlocks,
                                               size_t   elemBytes)
        {
            if(compactKBlocks == paddedKBlocks || compactFreeDim == 0)
                return;
            const size_t compactRow = compactKBlocks * elemBytes;
            const size_t paddedRow  = paddedKBlocks * elemBytes;
            const size_t padTail    = paddedRow - compactRow;
            for(size_t f = compactFreeDim; f-- > 1;)
            {
                std::memmove(buffer + f * paddedRow, buffer + f * compactRow, compactRow);
                std::memset(buffer + f * paddedRow + compactRow, 0x00, padTail);
            }
            std::memset(buffer + compactRow, 0x00, padTail);
        }

        void DataInitialization::initializeMXData(ContractionProblemGemm const& problem)
        {
            // Initializes A, B, MXSA, MXSB so the default-init loop in initializeCPUInputs
            // can safely skip them. For MX-FP4 / MX-FP8 / MX-BFloat8 sides we drive
            // mxDataGenerator (so the values are coordinated with their E8 scales); for any
            // non-FP4/FP8 side (e.g. MX-B6 or non-MX mixed-mode) we fall back to the same
            // initArray path the default loop would have taken, to avoid leaving the
            // malloc'd buffers uninitialized
            auto const& tensors = problem.tensors();

            auto initTensorFromDefault = [&](int i) {
                for(auto& p : m_vdata[i].pristine)
                {
                    if(p.second.initDescriptor[0] != tensors[i])
                    {
                        p.second.initDescriptor[0] = tensors[i];
                        initArray(p.first,
                                  m_vdata[i].init,
                                  p.second.cpuInput.valid.get(),
                                  tensors[i]);
                    }
                }
            };

            // Reset preswizzle flags; they will be set below if gpuInput.valid is populated.
            m_mxPreswizzledA = false;
            m_mxPreswizzledB = false;

            // Compute preSwizzle parameters from the solution's matrix instruction to rearrange
            // the scale tensor into the GPU kernel's expected memory layout
            std::vector<size_t> preSwizzleA, preTileA, preSwizzleB, preTileB;

            if(m_mxScaleFormat > 0 && m_currentSolution != nullptr)
            {
                auto const&      mi            = m_currentSolution->sizeMapping.matrixInstruction;
                size_t           MiK           = static_cast<size_t>(mi[2]);
                constexpr size_t swizzleTileMN = 32; // 2 SIMDs * 16 lanes per wave for MN access
                constexpr size_t tileK         = 256 / swizzleTileMN; // scale blocks per wave in K

                if(MiK > 0)
                {
                    if(problem.mxBlockA() > 0 && MiK % problem.mxBlockA() == 0)
                    {
                        // Scale tensor dimensions from setMXScaleA are already padded
                        // (K/mxBlock to multiple of 8, M to multiple of 32)
                        auto const& mxsaSizes = problem.mxsa().sizes();
                        size_t scaleRowsA = mxsaSizes[0];
                        size_t scaleColsA = mxsaSizes[1];
                        if(scaleRowsA % tileK == 0 && scaleColsA % swizzleTileMN == 0)
                        {
                            size_t subTileK = MiK / problem.mxBlockA();
                            preSwizzleA     = {swizzleTileMN, tileK, subTileK};
                            preTileA        = {tileK, swizzleTileMN};
                        }
                    }

                    if(problem.mxBlockB() > 0 && MiK % problem.mxBlockB() == 0)
                    {
                        // Scale tensor dimensions from setMXScaleB are already padded
                        // (K/mxBlock to multiple of 8, N to multiple of 32)
                        auto const& mxsbSizes = problem.mxsb().sizes();
                        size_t scaleRowsB = mxsbSizes[0];
                        size_t scaleColsB = mxsbSizes[1];
                        if(scaleRowsB % tileK == 0 && scaleColsB % swizzleTileMN == 0)
                        {
                            size_t subTileK = MiK / problem.mxBlockB();
                            preSwizzleB     = {swizzleTileMN, tileK, subTileK};
                            preTileB        = {tileK, swizzleTileMN};
                        }
                    }
                }
            }

            if(isMXTensor(problem.a(), problem.mxBlockA()))
            {
                auto const& tensorA = problem.a();
                auto        rows    = tensorA.sizes()[0];
                auto        cols    = tensorA.sizes()[1];
                auto        stride  = tensorA.strides()[1];
                size_t      batchCount = tensorA.sizes().size() > 2 ? tensorA.sizes()[2] : 1;

                auto& pristineA
                    = m_vdata[ContractionProblemGemm::TENSOR::A].pristine[tensorA.dataType()];
                auto& pristineE8A
                    = m_vdata[ContractionProblemGemm::TENSOR::MXSA].pristine[problem.mxsa().dataType()];

                // FP4: 2 elements packed per byte (packing=2); FP8: 1 element per byte
                // (packing=1). Compute byte stride generically via DataTypeInfo so we
                // never hard-code /2 again the next time a new dtype shows up
                size_t dataBatchStrideBytes = 0;
                size_t scaleBatchStrideBytes = 0;
                if(batchCount > 1)
                {
                    auto const  dataInfo         = DataTypeInfo::Get(tensorA.dataType());
                    dataBatchStrideBytes
                        = multiplyElementSize(tensorA.strides()[2], static_cast<float>(dataInfo.elementSize));
                    auto const& mxsaTensor = problem.mxsa();
                    scaleBatchStrideBytes = mxsaTensor.strides()[mxsaTensor.sizes().size() - 1];
                }

                auto initA = m_vdata[ContractionProblemGemm::TENSOR::A].init;

                // Zero the scale buffer; padding beyond the valid region stays 0x00
                std::memset(pristineE8A.cpuInput.valid.get(),
                            0x00,
                            problem.mxsa().totalAllocatedElements());

                // cpuInput.valid always holds canonical (non-preswizzled) scale so the CPU
                // reference reads it with correct linear strides.
                auto const& mxsaTensor   = problem.mxsa();
                auto        boundIdxA    = problem.boundIndices()[0].a;
                auto        freeIdxA     = problem.freeIndicesA()[0].i;
                size_t      compactKA    = (tensorA.sizes()[boundIdxA] + problem.mxBlockA() - 1)
                                           / problem.mxBlockA();
                size_t      paddedKA     = mxsaTensor.sizes()[boundIdxA];
                size_t      compactFreeA = tensorA.sizes()[freeIdxA];
                size_t      scaleElemA   = DataTypeInfo::Get(mxsaTensor.dataType()).elementSize;
                bool        kFastA       = (boundIdxA == 0);
                for(size_t b = 0; b < batchCount; b++)
                {
                    auto* dataPtr  = static_cast<uint8_t*>(pristineA.cpuInput.valid.get())
                                     + b * dataBatchStrideBytes;
                    auto* scalePtr = static_cast<uint8_t*>(pristineE8A.cpuInput.valid.get())
                                     + b * scaleBatchStrideBytes;
                    generateMXInput(hipMxDataTypeForDataGenerator(tensorA.dataType()),
                                    hipMxScaleTypeForDataGenerator(problem.mxTypeA()),
                                    dataPtr,
                                    scalePtr,
                                    rows,
                                    cols,
                                    stride,
                                    problem.transA(),
                                    {},
                                    {},
                                    problem.mxBlockA(),
                                    1,
                                    true,
                                    initModeToMXMethod(initA),
                                    -1.0f,
                                    1.0f);
                    if(kFastA)
                        restrideMXScaleBufferKFast(
                            scalePtr, compactFreeA, compactKA, paddedKA, scaleElemA);
                }

                pristineA.initDescriptor[0] = tensorA;
                pristineE8A.initDescriptor[0] = problem.mxsa();

                // For preswizzle-arch (gfx950): when the preswizzle condition fires,
                // generate the preswizzled scale and upload it directly to gpuInput.valid.
                // copySwizzledToGPUBuffer will use gpuInput.valid as-is instead of
                // applying the gfx1250 K-swizzle.
                if(m_isMXPreswizzleArch && !preSwizzleA.empty() && pristineE8A.gpuInput.valid)
                {
                    size_t gpuScaleBytes = problem.mxsa().totalAllocatedElements()
                                          * DataTypeInfo::Get(problem.mxsa().dataType()).elementSize;
                    std::vector<uint8_t> gpuScaleBuf(gpuScaleBytes, 0);
                    for(size_t b = 0; b < batchCount; b++)
                    {
                        auto* dataPtr  = static_cast<uint8_t*>(pristineA.cpuInput.valid.get())
                                         + b * dataBatchStrideBytes;
                        auto* scalePtr = gpuScaleBuf.data() + b * scaleBatchStrideBytes;
                        generateMXInput(hipMxDataTypeForDataGenerator(tensorA.dataType()),
                                        hipMxScaleTypeForDataGenerator(problem.mxTypeA()),
                                        dataPtr,
                                        scalePtr,
                                        rows,
                                        cols,
                                        stride,
                                        problem.transA(),
                                        preSwizzleA,
                                        preTileA,
                                        problem.mxBlockA(),
                                        1,
                                        true,
                                        initModeToMXMethod(initA),
                                        -1.0f,
                                        1.0f);
                    }
                    HIP_CHECK_EXC(hipMemcpy(pristineE8A.gpuInput.valid.get(),
                                            gpuScaleBuf.data(),
                                            gpuScaleBytes,
                                            hipMemcpyHostToDevice));
                    m_mxPreswizzledA = true;
                }
            }
            else
            {
                // A is not FP4/FP8 (or mxBlockA == 0). The default-init loop will skip A and
                // MXSA because useMXGenerator is true, so seed them here with the same
                // initArray path the default loop would have used.
                initTensorFromDefault(ContractionProblemGemm::TENSOR::A);
                if(problem.mxBlockA() > 0)
                    initTensorFromDefault(ContractionProblemGemm::TENSOR::MXSA);
            }

            if(isMXTensor(problem.b(), problem.mxBlockB()))
            {
                auto const& tensorB = problem.b();
                auto        rows    = tensorB.sizes()[0];
                auto        cols    = tensorB.sizes()[1];
                auto        stride  = tensorB.strides()[1];
                size_t      batchCount = tensorB.sizes().size() > 2 ? tensorB.sizes()[2] : 1;

                auto& pristineB
                    = m_vdata[ContractionProblemGemm::TENSOR::B].pristine[tensorB.dataType()];
                auto& pristineE8B
                    = m_vdata[ContractionProblemGemm::TENSOR::MXSB].pristine[problem.mxsb().dataType()];

                // FP4: 2 elements packed per byte (packing=2); FP8: 1 element per byte
                // (packing=1). Generic byte-stride via DataTypeInfo (see A side above).
                size_t dataBatchStrideBytes = 0;
                size_t scaleBatchStrideBytes = 0;
                if(batchCount > 1)
                {
                    auto const  dataInfo         = DataTypeInfo::Get(tensorB.dataType());
                    dataBatchStrideBytes
                        = multiplyElementSize(tensorB.strides()[2], static_cast<float>(dataInfo.elementSize));
                    auto const& mxsbTensor = problem.mxsb();
                    scaleBatchStrideBytes = mxsbTensor.strides()[mxsbTensor.sizes().size() - 1];
                }

                auto initB = m_vdata[ContractionProblemGemm::TENSOR::B].init;

                // Zero the scale buffer; padding beyond the valid region stays 0x00
                std::memset(pristineE8B.cpuInput.valid.get(),
                            0x00,
                            problem.mxsb().totalAllocatedElements());

                // cpuInput.valid holds canonical scale for the CPU reference.
                auto const& mxsbTensorRef = problem.mxsb();
                auto        boundIdxB    = problem.boundIndices()[0].b;
                auto        freeIdxB     = problem.freeIndicesB()[0].i;
                size_t      compactKB    = (tensorB.sizes()[boundIdxB] + problem.mxBlockB() - 1)
                                           / problem.mxBlockB();
                size_t      paddedKB     = mxsbTensorRef.sizes()[boundIdxB];
                size_t      compactFreeB = tensorB.sizes()[freeIdxB];
                size_t      scaleElemB   = DataTypeInfo::Get(mxsbTensorRef.dataType()).elementSize;
                bool        kFastB       = (boundIdxB == 0);
                for(size_t b = 0; b < batchCount; b++)
                {
                    auto* dataPtr  = static_cast<uint8_t*>(pristineB.cpuInput.valid.get())
                                     + b * dataBatchStrideBytes;
                    auto* scalePtr = static_cast<uint8_t*>(pristineE8B.cpuInput.valid.get())
                                     + b * scaleBatchStrideBytes;
                    generateMXInput(hipMxDataTypeForDataGenerator(tensorB.dataType()),
                                    hipMxScaleTypeForDataGenerator(problem.mxTypeB()),
                                    dataPtr,
                                    scalePtr,
                                    rows,
                                    cols,
                                    stride,
                                    problem.transB(),
                                    {},
                                    {},
                                    problem.mxBlockB(),
                                    1,
                                    false,
                                    initModeToMXMethod(initB),
                                    -1.0f,
                                    1.0f);
                    if(kFastB)
                        restrideMXScaleBufferKFast(
                            scalePtr, compactFreeB, compactKB, paddedKB, scaleElemB);
                }

                pristineB.initDescriptor[0] = tensorB;
                pristineE8B.initDescriptor[0] = problem.mxsb();

                // For preswizzle-arch (gfx950): upload preswizzled scale directly to gpuInput.valid.
                if(m_isMXPreswizzleArch && !preSwizzleB.empty() && pristineE8B.gpuInput.valid)
                {
                    size_t gpuScaleBytes = problem.mxsb().totalAllocatedElements()
                                          * DataTypeInfo::Get(problem.mxsb().dataType()).elementSize;
                    std::vector<uint8_t> gpuScaleBuf(gpuScaleBytes, 0);
                    for(size_t b = 0; b < batchCount; b++)
                    {
                        auto* dataPtr  = static_cast<uint8_t*>(pristineB.cpuInput.valid.get())
                                         + b * dataBatchStrideBytes;
                        auto* scalePtr = gpuScaleBuf.data() + b * scaleBatchStrideBytes;
                        generateMXInput(hipMxDataTypeForDataGenerator(tensorB.dataType()),
                                        hipMxScaleTypeForDataGenerator(problem.mxTypeB()),
                                        dataPtr,
                                        scalePtr,
                                        rows,
                                        cols,
                                        stride,
                                        problem.transB(),
                                        preSwizzleB,
                                        preTileB,
                                        problem.mxBlockB(),
                                        1,
                                        false,
                                        initModeToMXMethod(initB),
                                        -1.0f,
                                        1.0f);
                    }
                    HIP_CHECK_EXC(hipMemcpy(pristineE8B.gpuInput.valid.get(),
                                            gpuScaleBuf.data(),
                                            gpuScaleBytes,
                                            hipMemcpyHostToDevice));
                    m_mxPreswizzledB = true;
                }
            }
            else
            {
                // B is not FP4/FP8 (or mxBlockB == 0). Same fallback rationale as the A side.
                initTensorFromDefault(ContractionProblemGemm::TENSOR::B);
                if(problem.mxBlockB() > 0)
                    initTensorFromDefault(ContractionProblemGemm::TENSOR::MXSB);
            }
        }
#else  // HIPBLASLT_ENABLE_MXDATAGENERATOR
        void DataInitialization::initializeMXData(ContractionProblemGemm const& /*problem*/)
        {
            // The MX data generator is disabled at build time. Reaching this
            // path means a problem requiring MX FP4 or MX FP8 initialization was issued
            // against a build that doesn't include mxDataGenerator support.
            throw std::runtime_error(
                "MX data initialization requires HIPBLASLT_ENABLE_MXDATAGENERATOR=ON at build time");
        }
#endif // HIPBLASLT_ENABLE_MXDATAGENERATOR

        bool DataInitialization::cpuInputsNeedRefresh(
            ContractionProblemGemm const& problem) const
        {
            if(!m_problemDependentData)
                return false;

            auto const& tensors = problem.tensors();
            if(tensors.size() != m_vdata.size())
                return true;

            for(size_t i = 0; i < m_vdata.size(); i++)
            {
                if((i == ContractionProblemGemm::TENSOR::METADATA
                    || i == ContractionProblemGemm::TENSOR::COMPRESSED)
                   && problem.sparse() == 0)
                {
                    continue;
                }

                auto const& desc = tensors[i];
                auto const& pristine = m_vdata[i].pristine;
                if(pristine.empty())
                    continue;
                auto        it       = pristine.find(desc.dataType());
                if(it == pristine.end())
                    return true;

                auto const& p = it->second;
                if(p.initDescriptor.size() != 1 || p.initDescriptor[0] != desc)
                    return true;
            }

            return false;
        }

        bool DataInitialization::cpuInputsNeedRefresh(
            ContractionProblemGroupedGemm const& problem) const
        {
            if(!m_problemDependentData)
                return false;

            if(problem.gemms.empty())
                return true;

            for(auto const& gemm : problem.gemms)
            {
                if(gemm.tensors().size() != m_vdata.size())
                    return true;
            }

            for(size_t i = 0; i < m_vdata.size(); i++)
            {
                auto const& pristine = m_vdata[i].pristine;
                if(pristine.empty())
                    continue;
                for(size_t j = 0; j < problem.gemms.size(); j++)
                {
                    if((i == ContractionProblemGemm::TENSOR::METADATA
                        || i == ContractionProblemGemm::TENSOR::COMPRESSED)
                       && problem.gemms[j].sparse() == 0)
                    {
                        continue;
                    }

                    auto const& desc = problem.gemms[j].tensors()[i];
                    auto        it    = pristine.find(desc.dataType());
                    if(it == pristine.end())
                        return true;

                    auto const& p = it->second;
                    if(p.initDescriptor.size() <= j || p.initDescriptor[j] != desc)
                        return true;
                }
            }

            return false;
        }

        void DataInitialization::ensureCPUInputsCurrent(
            ContractionProblemGemm const& problem)
        {
            if(cpuInputsNeedRefresh(problem))
            {
                initializeCPUInputs(problem);
                assert(!cpuInputsNeedRefresh(problem));
            }
        }

        void DataInitialization::ensureCPUInputsCurrent(
            ContractionProblemGroupedGemm const& problem)
        {
            if(cpuInputsNeedRefresh(problem))
            {
                initializeCPUInputs(problem);
                assert(!cpuInputsNeedRefresh(problem));
            }
        }

        bool DataInitialization::shouldRefreshMXForSolution(
            ContractionSolution const*     solution,
            ContractionProblemGemm const& problem) const
        {
            return solution != nullptr && m_mxScaleFormat > 0 && isMXProblemExceptF6(problem)
                && gpuInputsPreparedFor(problem);
        }

        void DataInitialization::initializeConstantInputs(ContractionProblemGemm const& problem)
        {
            // Update constants if needed
            for(size_t i = 0; i < problem.constants().size(); i++)
            {
                auto& prop = m_cdata[i];
                if(prop.dataType != problem.constants()[i].dataType)
                {
                    prop.dataType = problem.constants()[i].dataType;
                    switch(prop.dataType)
                    {
                    case rocisa::DataType::Float:
                        prop.value = getValue<float>(prop.init, prop.freeValue);
                        break;
                    case rocisa::DataType::Double:
                        prop.value = getValue<double>(prop.init, prop.freeValue);
                        break;
                    case rocisa::DataType::Half:
                        prop.value = getValue<Half>(prop.init, prop.freeValue);
                        break;
                    case rocisa::DataType::Int32:
                        prop.value = getValue<int32_t>(prop.init, prop.freeValue);
                        break;
                    case rocisa::DataType::BFloat16:
                        prop.value = getValue<BFloat16>(prop.init, prop.freeValue);
                        break;
                    case rocisa::DataType::Int8:
                        prop.value = getValue<int8_t>(prop.init, prop.freeValue);
                        break;
                    case rocisa::DataType::ComplexFloat:
                        prop.value = getValue<std::complex<float>>(prop.init, prop.freeValue);
                        break;
                    case rocisa::DataType::ComplexDouble:
                        prop.value = getValue<std::complex<double>>(prop.init, prop.freeValue);
                        break;
                    case rocisa::DataType::Int8x4:
                        prop.value = getValue<Int8x4>(prop.init, prop.freeValue);
                        break;
                    case rocisa::DataType::Float8:
                        prop.value = getValue<Float8>(prop.init, prop.freeValue);
                        break;
                    case rocisa::DataType::BFloat8:
                        prop.value = getValue<BFloat8>(prop.init, prop.freeValue);
                        break;
                    case rocisa::DataType::Float8_fnuz:
                        prop.value = getValue<Float8_fnuz>(prop.init, prop.freeValue);
                        break;
                    case rocisa::DataType::BFloat8_fnuz:
                        prop.value = getValue<BFloat8_fnuz>(prop.init, prop.freeValue);
                        break;
#ifndef _WIN32
#ifdef TENSILE_USE_FP6
                    case rocisa::DataType::Float6:
                        prop.value = getValue<Float6x32>(prop.init, prop.freeValue);
                        break;
#endif // #ifdef TENSILE_USE_FP6
#ifdef TENSILE_USE_BF6
                    case rocisa::DataType::BFloat6:
                        prop.value = getValue<BFloat6x32>(prop.init, prop.freeValue);
                        break;
#endif // #ifdef TENSILE_USE_BF6
#ifdef TENSILE_USE_FP4
                    case rocisa::DataType::Float4:
                        prop.value = getValue<Float4x2>(prop.init, prop.freeValue);
                        break;
#endif // #ifdef TENSILE_USE_FP4
#endif // !_WIN32
                    case rocisa::DataType::E8:
                        prop.value = getValue<E8>(prop.init, prop.freeValue);
                        break;
                    case rocisa::DataType::E5M3:
                    case rocisa::DataType::Int64:
                    case rocisa::DataType::XFloat32:
                    case rocisa::DataType::Count:
                    case rocisa::DataType::Float8BFloat8:
                    case rocisa::DataType::BFloat8Float8:
                    case rocisa::DataType::Float8BFloat8_fnuz:
                    case rocisa::DataType::BFloat8Float8_fnuz:
#ifdef _WIN32
                    case rocisa::DataType::Float6:
                    case rocisa::DataType::BFloat6:
                    case rocisa::DataType::Float4:
#endif // _WIN32
                    ;
                    }
                }
                if(Debug::Instance().printTensorInfo() && prop.dataType != rocisa::DataType::None)
                    std::cout << "Constant " << m_cdata[i].name << ". Type "
                              << DataTypeInfo::Get(prop.dataType).abbrev << std::endl;
            }
            return;
        }

        void DataInitialization::copyInputs(std::vector<void*>&               ptrs,
                                            std::vector<void**>&              batchPtrs,
                                            std::vector<size_t>&              maxElements,
                                            std::vector<std::vector<size_t>>& offsets,
                                            ContractionProblemGemm const&     problem,
                                            hipMemcpyKind                     kind,
                                            hipStream_t                       targetStream)
        {
            ptrs.clear();
            batchPtrs.clear();
            maxElements.clear();
            if(m_curBoundsCheck == BoundsCheckMode::NaN)
            {
                for(size_t i = 0; i < m_vdata.size(); i++)
                {
                    void* ptr  = nullptr;
                    auto& desc = problem.tensors()[i];
                    auto  it   = m_vdata[i].pristine.find(desc.dataType());
                    if(it != m_vdata[i].pristine.end())
                    {
                        auto& p = it->second;
                        if(kind == hipMemcpyHostToHost)
                            ptr = copyBadInputBuffers(desc,
                                                      p.cpuInput.current.get(),
                                                      p.cpuInput.valid.get(),
                                                      p.cpuInput.bad.get(),
                                                      p.maxElements,
                                                      kind);
                        else if(kind == hipMemcpyHostToDevice)
                            ptr = copyBadInputBuffers(desc,
                                                      p.gpuInput.current.get(),
                                                      p.cpuInput.valid.get(),
                                                      p.cpuInput.bad.get(),
                                                      p.maxElements,
                                                      kind);
                        else if(kind == hipMemcpyDeviceToDevice)
                            ptr = copyBadInputBuffers(desc,
                                                      p.gpuInput.current.get(),
                                                      p.gpuInput.valid.get(),
                                                      p.gpuInput.bad.get(),
                                                      p.maxElements,
                                                      kind,
                                                      targetStream);
                        ptrs.push_back(ptr);
                        batchPtrs.push_back(p.getInputByKind(kind).batch.get());
                        maxElements.push_back(p.maxElements);
                        offsets.push_back(p.groupedGemmOffsets);
                    }
                    else
                    {
                        ptrs.push_back(nullptr);
                        batchPtrs.push_back(nullptr);
                        maxElements.push_back(0);
                        offsets.push_back(std::vector<size_t>());
                    }
                }
            }
            else if(m_curBoundsCheck == BoundsCheckMode::GuardPageBack)
            {
                for(size_t i = 0; i < m_vdata.size(); i++)
                {
                    void* ptr  = nullptr;
                    auto& desc = problem.tensors()[i];
                    auto  it   = m_vdata[i].pristine.find(desc.dataType());
                    if(it != m_vdata[i].pristine.end())
                    {
                        auto&     p = it->second;
                        ptrdiff_t swizzlePadding{-1};

                        if(problem.swizzleTensorA() && i == ContractionProblemGemm::TENSOR::A
                           || (problem.swizzleTensorB() && i == ContractionProblemGemm::TENSOR::B))
                        {
                            //TODO: support more swizzle types,
                            //      currently, if A then it means MiM = 16, if B then it means MiN = 16
                            size_t MiM_N = 16, MiK = 0, MiKv = 0, PackK = 0;
                            calculateKforSwizzling(desc.dataType(), MiK, MiKv, PackK);
                            swizzlePadding
                                = getSwizzledTensorNumAllocatedElements(desc, MiM_N, MiK, PackK)
                                  - desc.totalAllocatedElements();
                        }

                        if(kind == hipMemcpyHostToHost)
                            ptr = copyNaNInputBuffers(desc,
                                                      p.cpuInput.current.get(),
                                                      p.cpuInput.valid.get(),
                                                      p.maxElements,
                                                      kind,
                                                      swizzlePadding);
                        else if(kind == hipMemcpyHostToDevice)
                            ptr = copyNaNInputBuffers(desc,
                                                      p.gpuInput.current.get(),
                                                      p.cpuInput.valid.get(),
                                                      p.maxElements,
                                                      kind,
                                                      swizzlePadding);
                        else if(kind == hipMemcpyDeviceToDevice)
                            ptr = copyNaNInputBuffers(desc,
                                                      p.gpuInput.current.get(),
                                                      p.gpuInput.valid.get(),
                                                      p.maxElements,
                                                      kind,
                                                      swizzlePadding,
                                                      targetStream);
                        ptrs.push_back(ptr);
                        batchPtrs.push_back(p.getInputByKind(kind).batch.get());
                        maxElements.push_back(p.maxElements);
                        offsets.push_back(p.groupedGemmOffsets);
                    }
                    else
                    {
                        ptrs.push_back(nullptr);
                        batchPtrs.push_back(nullptr);
                        maxElements.push_back(0);
                        offsets.push_back(std::vector<size_t>());
                    }
                }
            }
            else
            {
                for(size_t i = 0; i < m_vdata.size(); i++)
                {
                    void* ptr  = nullptr;
                    auto& desc = problem.tensors()[i];
                    auto  it   = m_vdata[i].pristine.find(desc.dataType());
                    if(it != m_vdata[i].pristine.end())
                    {
                        auto& p = it->second;
                        if(kind == hipMemcpyHostToHost)
                            ptr = copyInputBuffers(desc,
                                                   p.cpuInput.current.get(),
                                                   p.cpuInput.valid.get(),
                                                   p.maxElements,
                                                   kind);
                        else if(kind == hipMemcpyHostToDevice)
                            ptr = copyInputBuffers(desc,
                                                   p.gpuInput.current.get(),
                                                   p.cpuInput.valid.get(),
                                                   p.maxElements,
                                                   kind);
                        else if(kind == hipMemcpyDeviceToDevice)
                            ptr = copyInputBuffers(desc,
                                                   p.gpuInput.current.get(),
                                                   p.gpuInput.valid.get(),
                                                   p.maxElements,
                                                   kind,
                                                   targetStream);
                        if(ptr == nullptr)
                        {
                            throw std::runtime_error("output ptr is null when copy input");
                        }
                        ptrs.push_back(ptr);
                        batchPtrs.push_back(p.getInputByKind(kind).batch.get());
                        maxElements.push_back(p.maxElements);
                        offsets.push_back(p.groupedGemmOffsets);
                    }
                    else
                    {
                        ptrs.push_back(nullptr);
                        batchPtrs.push_back(nullptr);
                        maxElements.push_back(0);
                        offsets.push_back(std::vector<size_t>());
                    }
                }
            }
        }

        void DataInitialization::resetOutput(std::vector<void*>&               ptrs,
                                             std::vector<void**>&              batchPtrs,
                                             std::vector<size_t>&              maxElements,
                                             std::vector<std::vector<size_t>>& offsets,
                                             ContractionProblemGemm const&     problem,
                                             hipMemcpyKind                     kind,
                                             hipStream_t                       targetStream)
        {
            hipStream_t copyStream = targetStream ? targetStream : m_copyStream;
            bool        useAsync   = (kind == hipMemcpyDeviceToDevice) && copyStream;
            {
                for(size_t i = 0; i < m_vdata.size(); i++)
                {
                    void* ptr  = nullptr;
                    auto& desc = problem.tensors()[i];
                    if(!desc.isOutput()) // Need init first
                        continue;
                    auto it = m_vdata[i].pristine.find(desc.dataType());
                    if(it != m_vdata[i].pristine.end())
                    {
                        auto& p = it->second;
                        // For output tensors with NaN bounds checking, initialize buffer with NaN sentinels
                        if(m_curBoundsCheck == BoundsCheckMode::NaN)
                        {
                            if(kind == hipMemcpyHostToHost)
                                ptr = copyBadInputBuffers(desc,
                                                          p.cpuInput.current.get(),
                                                          p.cpuInput.valid.get(),
                                                          p.cpuInput.bad.get(),
                                                          p.maxElements,
                                                          kind);
                            else if(kind == hipMemcpyHostToDevice)
                                ptr = copyBadInputBuffers(desc,
                                                          p.gpuInput.current.get(),
                                                          p.cpuInput.valid.get(),
                                                          p.cpuInput.bad.get(),
                                                          p.maxElements,
                                                          kind);
                            else if(kind == hipMemcpyDeviceToDevice)
                                ptr = copyBadInputBuffers(desc,
                                                          p.gpuInput.current.get(),
                                                          p.gpuInput.valid.get(),
                                                          p.gpuInput.bad.get(),
                                                          p.maxElements,
                                                          kind,
                                                          useAsync ? copyStream : nullptr);
                        }
                        else
                        {
                            if(kind == hipMemcpyHostToHost)
                                ptr = copyInputBuffers(desc,
                                                       p.cpuInput.current.get(),
                                                       p.cpuInput.valid.get(),
                                                       p.maxElements,
                                                       kind);
                            else if(kind == hipMemcpyHostToDevice)
                                ptr = copyInputBuffers(desc,
                                                       p.gpuInput.current.get(),
                                                       p.cpuInput.valid.get(),
                                                       p.maxElements,
                                                       kind);
                            else if(kind == hipMemcpyDeviceToDevice)
                            {
                                if(useAsync)
                                {
                                    HIP_CHECK_EXC(hipMemcpyAsync(p.gpuInput.current.get(),
                                                                 p.gpuInput.valid.get(),
                                                                 multiplyElementSize(p.maxElements,
                                                                                     desc.elementBytes()),
                                                                 kind,
                                                                 copyStream));
                                    ptr = p.gpuInput.current.get();
                                }
                                else
                                    ptr = copyInputBuffers(desc,
                                                           p.gpuInput.current.get(),
                                                           p.gpuInput.valid.get(),
                                                           p.maxElements,
                                                           kind);
                            }
                        }
                        if(ptr == nullptr)
                        {
                            throw std::runtime_error("output ptr is null when copy input");
                        }
                        ptrs[i]        = ptr;
                        batchPtrs[i]   = p.getInputByKind(kind).batch.get();
                        maxElements[i] = p.maxElements;
                        offsets[i]     = p.groupedGemmOffsets;
                    }
                    else
                    {
                        ptrs[i]        = nullptr;
                        batchPtrs[i]   = nullptr;
                        maxElements[i] = 0;
                        offsets[i].clear();
                    }
                }
            }
            if(useAsync && !targetStream)
                HIP_CHECK_EXC(hipStreamSynchronize(copyStream));
        }

        void DataInitialization::copyValidToGPUBuffer(
            ContractionProblemGemm const& problem,
            bool                          callerOwnsCopySync)
        {
            for(size_t i = 0; i < m_vdata.size(); i++)
            {
                bool needSwizzle
                    = (problem.swizzleTensorA() && i == ContractionProblemGemm::TENSOR::A)
                      || (problem.swizzleTensorB() && i == ContractionProblemGemm::TENSOR::B);
                bool needMXSwizzle
                    = (problem.mxBlockA() && (i == ContractionProblemGemm::TENSOR::MXSA))
                      || (problem.mxBlockB() && (i == ContractionProblemGemm::TENSOR::MXSB));
                //Copy swizzle tensor would be in copySwizzledToGPUBuffer
                if(needSwizzle || needMXSwizzle)
                    continue;
                auto& desc = problem.tensors()[i];
                auto  it   = m_vdata[i].pristine.find(desc.dataType());
                if(it == m_vdata[i].pristine.end())
                    continue;
                auto& p = m_vdata[i].pristine[desc.dataType()];
                if(p.gpuInput.valid.get() == nullptr || p.cpuInput.valid.get() == nullptr)
                    continue;
                if(m_copyStream)
                {
                    HIP_CHECK_EXC(hipMemcpyAsync(p.gpuInput.valid.get(),
                                                 p.cpuInput.valid.get(),
                                                 multiplyElementSize(p.maxElements,
                                                                     desc.elementBytes()),
                                                 hipMemcpyHostToDevice,
                                                 m_copyStream));
                }
                else
                {
                    HIP_CHECK_EXC(hipMemcpy(p.gpuInput.valid.get(),
                                            p.cpuInput.valid.get(),
                                            multiplyElementSize(p.maxElements,
                                                                desc.elementBytes()),
                                            hipMemcpyHostToDevice));
                }
            }
            if(m_copyStream && !callerOwnsCopySync)
                HIP_CHECK_EXC(hipStreamSynchronize(m_copyStream));
        }

        void DataInitialization::copySwizzledToGPUBuffer(
            ContractionProblemGemm const& problem,
            hipStream_t                   targetStream,
            std::vector<SwizzleUpload>*   swizzleStaging)
        {
            using ManipTensor = ::Tensor::Manipulation::Tensor;

            hipStream_t copyStream = targetStream ? targetStream : m_copyStream;
            bool const  useAsync   = copyStream != nullptr;
            bool const  callerOwnsCopySync = targetStream != nullptr;

            if(targetStream && swizzleStaging == nullptr)
                throw std::logic_error("Async swizzle uploads require staging storage.");

            std::vector<SwizzleUpload> localSwizzleStaging;
            auto& uploadStaging = swizzleStaging ? *swizzleStaging : localSwizzleStaging;

            auto stageTensor = [&](ManipTensor const& source) -> SwizzleUpload& {
                auto& upload         = uploadStaging.emplace_back();
                upload.totalElements = source.getDesc().flattenSize();
                upload.bytes.resize(source.getNumBytes());
                if(!upload.bytes.empty())
                    memcpy(upload.bytes.data(), source.as<void>(), upload.bytes.size());
                return upload;
            };

            for(size_t i = 0; i < m_vdata.size(); i++)
            {
                auto& desc = problem.tensors()[i];
                auto  it   = m_vdata[i].pristine.find(desc.dataType());
                if(it == m_vdata[i].pristine.end())
                    continue;
                auto& p = m_vdata[i].pristine[desc.dataType()];
                if(p.gpuInput.valid.get() == nullptr || p.cpuInput.valid.get() == nullptr)
                    continue;

                bool needSwizzle
                    = (problem.swizzleTensorA() && i == ContractionProblemGemm::TENSOR::A)
                      || (problem.swizzleTensorB() && i == ContractionProblemGemm::TENSOR::B);

                bool needMXSwizzle = false;
                bool unrollMajor = false;
                size_t MX = 0;
                if (i == ContractionProblemGemm::TENSOR::MXSA && problem.mxBlockA())
                {
                    needMXSwizzle = true;
                    unrollMajor = (problem.freeIndicesA()[0].i != 0);
                    MX = problem.mxBlockA();
                }
                else if (i == ContractionProblemGemm::TENSOR::MXSB && problem.mxBlockB())
                {
                    needMXSwizzle = true;
                    unrollMajor = (problem.freeIndicesB()[0].i != 0);
                    MX = problem.mxBlockB();
                }

                void* ptr{};

                if(needSwizzle)
                {
                    // currently, if A then it means MiM = 16, if B then it means MiN = 16
                    size_t MiM_N = 16, MiK = 0, MiKv = 0, PackK = 0;
                    calculateKforSwizzling(desc.dataType(), MiK, MiKv, PackK);
                    auto                          unrolledSize = desc.sizes()[0];
                    auto                          tiledSize    = desc.sizes()[1];
                    ::Tensor::Manipulation::Shape paddedShape{
                        ((tiledSize / MiM_N) + !!(tiledSize % MiM_N)) * MiM_N,
                        (unrolledSize / (MiK * PackK) + !!(unrolledSize % (MiK * PackK))) * MiK
                            * PackK};
                    auto swizzleKey
                        = std::make_tuple(toBitWidth(desc.dataType()), unrolledSize, tiledSize);

                    if(g_swizzleCache.count(swizzleKey))
                    {
                        if(swizzleKey != g_swizzleCache.back())
                        {
                            auto& permuted = g_swizzleCache.at(swizzleKey);
                            if(useAsync)
                            {
                                auto& staged = stageTensor(permuted);
                                ptr          = copyInputBuffers(desc,
                                                   p.gpuInput.valid.get(),
                                                   staged.bytes.data(),
                                                   staged.totalElements,
                                                   hipMemcpyHostToDevice,
                                                   copyStream);
                            }
                            else
                            {
                                ptr = copyInputBuffers(desc,
                                                       p.gpuInput.valid.get(),
                                                       permuted.as<void>(),
                                                       permuted.getDesc().flattenSize(),
                                                       hipMemcpyHostToDevice);
                            }
                        }
                        else
                        {
                            ptr = p.gpuInput.valid.get();
                        }
                    }
                    else
                    {
                        ManipTensor tmpTensor({tiledSize, unrolledSize}, desc.elementBytes());

                        memcpy(
                            tmpTensor.as<void>(), p.cpuInput.valid.get(), tmpTensor.getNumBytes());
                        //Temporary hack
                        uint64_t padVal{};
                        auto     paddedTensor = ::Tensor::Manipulation::pad(
                            tmpTensor, paddedShape, &padVal, tmpTensor.getElementSize());
                        paddedTensor.reshape({paddedShape[0] / MiM_N,
                                              MiM_N,
                                              paddedShape[1] / (MiK * PackK),
                                              MiK / MiKv,
                                              MiKv * PackK});
                        ManipTensor permuted = permute(paddedTensor, {0, 2, 3, 1, 4});
                        g_swizzleCache.emplace(swizzleKey, std::move(permuted));
                        auto& cachedPermuted = g_swizzleCache.at(swizzleKey);
                        if(useAsync)
                        {
                            auto& staged = stageTensor(cachedPermuted);
                            ptr          = copyInputBuffers(desc,
                                                       p.gpuInput.valid.get(),
                                                       staged.bytes.data(),
                                                       staged.totalElements,
                                                       hipMemcpyHostToDevice,
                                                       copyStream);
                        }
                        else
                        {
                            ptr = copyInputBuffers(desc,
                                                   p.gpuInput.valid.get(),
                                                   cachedPermuted.as<void>(),
                                                   cachedPermuted.getDesc().flattenSize(),
                                                   hipMemcpyHostToDevice);
                        }
                    }
                }
                else if (needMXSwizzle)
                {
                    bool isMXSA = (i == ContractionProblemGemm::TENSOR::MXSA);
                    bool isMXSB = (i == ContractionProblemGemm::TENSOR::MXSB);
                    bool preswizzledAlready = (isMXSA && m_mxPreswizzledA)
                                             || (isMXSB && m_mxPreswizzledB);

                    // The picked solution dictates the in-device MX scale layout via
                    // problemType.mxScaleFormat (mirrors the MXScaleFormat solution
                    // parameter): 0=NoSwizzle, 1=HostPreSwizzle, 2=InMemorySwizzle.
                    // Sentinel -1 means "no solution selected yet" (e.g. the first
                    // prepareGPUInputs call per problem, before solution iteration);
                    // in that case the path below uses the arch-driven default
                    // (gfx950 host preswizzle, otherwise K-swizzle).
                    int kernelMxScaleFormat = -1;
                    if (m_currentSolution != nullptr)
                        kernelMxScaleFormat = m_currentSolution->problemType.mxScaleFormat;

                    if (kernelMxScaleFormat == 0)
                    {
                        // NoSwizzle: kernel reads scales in canonical row/column
                        // layout (buffer_load_* path). Upload cpuInput.valid as-is,
                        // no K-swizzle, no padding permute.
                        ptr = copyInputBuffers(desc,
                                               p.gpuInput.valid.get(),
                                               p.cpuInput.valid.get(),
                                               p.maxElements,
                                               hipMemcpyHostToDevice,
                                               copyStream);
                    }
                    else if (m_isMXPreswizzleArch && preswizzledAlready)
                    {
                        // gfx950 subtile: preswizzle was applied by initializeMXDataForFP4 and
                        // gpuInput.valid was already populated — use it as-is.
                        ptr = p.gpuInput.valid.get();
                    }
                    else if (m_isMXPreswizzleArch)
                    {
                        // gfx950: preswizzle didn't fire (scale dims not divisible by tileK,
                        // e.g. small K). Kernel expects canonical layout — copy cpuInput.valid
                        // directly without K-swizzle.
                        ptr = copyInputBuffers(desc,
                                               p.gpuInput.valid.get(),
                                               p.cpuInput.valid.get(),
                                               p.maxElements,
                                               hipMemcpyHostToDevice,
                                               copyStream);
                    }
                    else
                    {
                        // gfx1250 and other arches: apply K-dimension swizzle.
                        // gfx950 is excluded by the branches above.
                        // Batch dim (if present) goes at the front; pad/reshape/permute
                        // operate natively on N-D so all batches are processed at once.
                        size_t batch = desc.sizes().size() > 2 ? desc.sizes()[2] : 1;

                        if (unrollMajor)
                        {
                            auto unrolledSize = desc.sizes()[0];
                            auto tiledSize    = desc.sizes()[1];
                            size_t dimk       = 128 / MX;
                            ManipTensor tmpTensor({batch, tiledSize, unrolledSize},
                                                  desc.elementBytes());
                            ::Tensor::Manipulation::Shape paddedShape{
                                batch, tiledSize, (unrolledSize + dimk - 1) / dimk * dimk};

                            memcpy(tmpTensor.as<void>(), p.cpuInput.valid.get(), tmpTensor.getNumBytes());
                            //Temporary hack
                            uint64_t padVal{};
                            auto     paddedTensor = ::Tensor::Manipulation::pad(
                                tmpTensor, paddedShape, &padVal, tmpTensor.getElementSize());
                            paddedTensor.reshape({batch,
                                                  paddedShape[1],
                                                  paddedShape[2] / dimk,
                                                  dimk});
                            ManipTensor permuted = permute(paddedTensor, {0, 2, 1, 3});
                            if(useAsync)
                            {
                                auto& staged = stageTensor(permuted);
                                ptr          = copyInputBuffers(desc,
                                                   p.gpuInput.valid.get(),
                                                   staged.bytes.data(),
                                                   staged.totalElements,
                                                   hipMemcpyHostToDevice,
                                                   copyStream);
                            }
                            else
                            {
                                ptr = copyInputBuffers(desc,
                                                       p.gpuInput.valid.get(),
                                                       permuted.as<void>(),
                                                       permuted.getDesc().flattenSize(),
                                                       hipMemcpyHostToDevice);
                            }
                        }
                        else
                        {
                            auto unrolledSize = desc.sizes()[1];
                            auto tiledSize    = desc.sizes()[0];
                            size_t dimk       = 128 / MX;
                            ManipTensor tmpTensor({batch, unrolledSize, tiledSize},
                                                  desc.elementBytes());
                            ::Tensor::Manipulation::Shape paddedShape{
                                batch, (unrolledSize + dimk - 1) / dimk * dimk, tiledSize};

                            memcpy(tmpTensor.as<void>(), p.cpuInput.valid.get(), tmpTensor.getNumBytes());
                            //Temporary hack
                            uint64_t padVal{};
                            auto     paddedTensor = ::Tensor::Manipulation::pad(
                                tmpTensor, paddedShape, &padVal, tmpTensor.getElementSize());
                            paddedTensor.reshape({batch,
                                                  paddedShape[1] / dimk,
                                                  dimk,
                                                  paddedShape[2]});
                            ManipTensor permuted = permute(paddedTensor, {0, 1, 3, 2});
                            if(useAsync)
                            {
                                auto& staged = stageTensor(permuted);
                                ptr          = copyInputBuffers(desc,
                                                   p.gpuInput.valid.get(),
                                                   staged.bytes.data(),
                                                   staged.totalElements,
                                                   hipMemcpyHostToDevice,
                                                   copyStream);
                            }
                            else
                            {
                                ptr = copyInputBuffers(desc,
                                                       p.gpuInput.valid.get(),
                                                       permuted.as<void>(),
                                                       permuted.getDesc().flattenSize(),
                                                       hipMemcpyHostToDevice);
                            }
                        }
                    }
                }
                else
                {
                    ptr = copyInputBuffers(desc,
                                           p.gpuInput.valid.get(),
                                           p.cpuInput.valid.get(),
                                           p.maxElements,
                                           hipMemcpyHostToDevice,
                                           copyStream);
                }

                if(ptr == nullptr)
                    std::__throw_runtime_error("error");
            }
            if(useAsync && !callerOwnsCopySync)
                HIP_CHECK_EXC(hipStreamSynchronize(copyStream));
        }

        template <typename T>
        void DataInitialization::setContractionInputs(std::vector<T*>&     ptrs,
                                                      std::vector<void**>& batchPtrs,
                                                      void*                ws,
                                                      std::vector<ConstDataInitProperties>& cdata,
                                                      std::vector<size_t> maxElements,
                                                      bool                isGPU,
                                                      ContractionInputs*  inputs)
        {
            inputs->a             = (void*)ptrs[ContractionProblemGemm::TENSOR::A];
            inputs->b             = (void*)ptrs[ContractionProblemGemm::TENSOR::B];
            inputs->c             = (void*)ptrs[ContractionProblemGemm::TENSOR::C];
            inputs->d             = (void*)ptrs[ContractionProblemGemm::TENSOR::D];
            inputs->e             = (void*)ptrs[ContractionProblemGemm::TENSOR::E];
            inputs->bias          = (void*)ptrs[ContractionProblemGemm::TENSOR::BIAS];
            inputs->scaleA        = (void*)ptrs[ContractionProblemGemm::TENSOR::SCALEA];
            inputs->scaleB        = (void*)ptrs[ContractionProblemGemm::TENSOR::SCALEB];
            inputs->scaleC        = (void*)ptrs[ContractionProblemGemm::TENSOR::SCALEC];
            inputs->scaleD        = (void*)ptrs[ContractionProblemGemm::TENSOR::SCALED];
            inputs->scaleAlphaVec = (void*)ptrs[ContractionProblemGemm::TENSOR::SCALEALPHAVEC];
            inputs->mxsa          = (void*)ptrs[ContractionProblemGemm::TENSOR::MXSA];
            inputs->mxsb          = (void*)ptrs[ContractionProblemGemm::TENSOR::MXSB];
            inputs->metadata      = (unsigned char*)ptrs[ContractionProblemGemm::TENSOR::METADATA];
            inputs->Synchronizer  = (void*)ptrs[ContractionProblemGemm::TENSOR::Synchronizer];
            inputs->amaxD         = (void*)ptrs[ContractionProblemGemm::TENSOR::AMAXD];
            inputs->compressed    = (void*)ptrs[ContractionProblemGemm::TENSOR::COMPRESSED];

            inputs->batchA    = (void**)batchPtrs[ContractionProblemGemm::TENSOR::A];
            inputs->batchB    = (void**)batchPtrs[ContractionProblemGemm::TENSOR::B];
            inputs->batchC    = (void**)batchPtrs[ContractionProblemGemm::TENSOR::C];
            inputs->batchD    = (void**)batchPtrs[ContractionProblemGemm::TENSOR::D];
            inputs->batchBias = (void**)batchPtrs[ContractionProblemGemm::TENSOR::BIAS];

            inputs->gpu = isGPU;

            inputs->ws             = (void*)ws;
            inputs->alpha          = cdata[ContractionProblemGemm::CONST::ALPHA].value;
            inputs->beta           = cdata[ContractionProblemGemm::CONST::BETA].value;
            inputs->activationArgs = {cdata[ContractionProblemGemm::CONST::ACTALPHA].value,
                                      cdata[ContractionProblemGemm::CONST::ACTBETA].value};

            inputs->maxElements = maxElements;
        }

        void DataInitialization::setContractionGroupedInputs(
            std::vector<void*>&                     ptrs,
            std::vector<void**>&                    batchPtrs,
            void*                                   ws,
            std::vector<ConstDataInitProperties>&   cdata,
            bool                                    isGPU,
            ContractionProblemGemm const&           problem,
            std::vector<std::vector<size_t>> const& offsets,
            ContractionGroupedInputs*               inputs)
        {

            std::vector<uint8_t*> u8Ptr;
            for(auto p : ptrs)
            {
                u8Ptr.push_back((uint8_t*)p);
            }

            inputs->ws = ws;

            for(int idx = 0; idx < offsets[0].size(); idx++)
            {
                ContractionInputs   unit;
                std::vector<size_t> maxElements;
                for(size_t j = 0; j < offsets.size(); j++)
                {

                    if(offsets[j].size() != 0)
                    {
                        maxElements.push_back(offsets[j][idx]);
                    }
                    else
                    {
                        maxElements.push_back(0);
                    }
                }
                setContractionInputs(u8Ptr, batchPtrs, ws, cdata, maxElements, isGPU, &unit);
                inputs->grouped.push_back(unit);

                u8Ptr[ContractionProblemGemm::TENSOR::A] += multiplyElementSize(
                    offsets[ContractionProblemGemm::TENSOR::A][idx], problem.a().elementBytes());
                u8Ptr[ContractionProblemGemm::TENSOR::B] += multiplyElementSize(
                    offsets[ContractionProblemGemm::TENSOR::B][idx], problem.b().elementBytes());
                u8Ptr[ContractionProblemGemm::TENSOR::C] += multiplyElementSize(
                    offsets[ContractionProblemGemm::TENSOR::C][idx], problem.c().elementBytes());
                u8Ptr[ContractionProblemGemm::TENSOR::D] += multiplyElementSize(
                    offsets[ContractionProblemGemm::TENSOR::D][idx], problem.d().elementBytes());
                if(u8Ptr[ContractionProblemGemm::TENSOR::E] != nullptr)
                {
                    u8Ptr[ContractionProblemGemm::TENSOR::E] += multiplyElementSize(
                        offsets[ContractionProblemGemm::TENSOR::E][idx],
                        problem.tensors()[ContractionProblemGemm::TENSOR::E].elementBytes());
                }
                if(u8Ptr[ContractionProblemGemm::TENSOR::BIAS] != nullptr)
                {
                    u8Ptr[ContractionProblemGemm::TENSOR::BIAS] += multiplyElementSize(
                        offsets[ContractionProblemGemm::TENSOR::BIAS][idx],
                        problem.tensors()[ContractionProblemGemm::TENSOR::BIAS].elementBytes());
                }
                if(u8Ptr[ContractionProblemGemm::TENSOR::SCALEA] != nullptr)
                {
                    u8Ptr[ContractionProblemGemm::TENSOR::SCALEA] += multiplyElementSize(
                        offsets[ContractionProblemGemm::TENSOR::SCALEA][idx],
                        problem.tensors()[ContractionProblemGemm::TENSOR::SCALEA].elementBytes());
                }
                if(u8Ptr[ContractionProblemGemm::TENSOR::SCALEB] != nullptr)
                {
                    u8Ptr[ContractionProblemGemm::TENSOR::SCALEB] += multiplyElementSize(
                        offsets[ContractionProblemGemm::TENSOR::SCALEB][idx],
                        problem.tensors()[ContractionProblemGemm::TENSOR::SCALEB].elementBytes());
                }
                if(u8Ptr[ContractionProblemGemm::TENSOR::SCALEC] != nullptr)
                {
                    u8Ptr[ContractionProblemGemm::TENSOR::SCALEC] += multiplyElementSize(
                        offsets[ContractionProblemGemm::TENSOR::SCALEC][idx],
                        problem.tensors()[ContractionProblemGemm::TENSOR::SCALEC].elementBytes());
                }
                if(u8Ptr[ContractionProblemGemm::TENSOR::SCALED] != nullptr)
                {
                    u8Ptr[ContractionProblemGemm::TENSOR::SCALED] += multiplyElementSize(
                        offsets[ContractionProblemGemm::TENSOR::SCALED][idx],
                        problem.tensors()[ContractionProblemGemm::TENSOR::SCALED].elementBytes());
                }
                if(u8Ptr[ContractionProblemGemm::TENSOR::SCALEALPHAVEC] != nullptr)
                {
                    u8Ptr[ContractionProblemGemm::TENSOR::SCALEALPHAVEC] += multiplyElementSize(
                        offsets[ContractionProblemGemm::TENSOR::SCALEALPHAVEC][idx],
                        problem.tensors()[ContractionProblemGemm::TENSOR::SCALEALPHAVEC]
                            .elementBytes());
                }
                if(u8Ptr[ContractionProblemGemm::TENSOR::Synchronizer] != nullptr)
                {
                    u8Ptr[ContractionProblemGemm::TENSOR::Synchronizer] += multiplyElementSize(
                        offsets[ContractionProblemGemm::TENSOR::Synchronizer][idx],
                        problem.tensors()[ContractionProblemGemm::TENSOR::Synchronizer]
                            .elementBytes());
                }
            }
        }

        // Build a ProblemInputs from explicit pointer/size vectors (GPU path).
        std::shared_ptr<ProblemInputs>
            DataInitialization::buildGPUProblemInputs(
                std::vector<void*>&                    ptrs,
                std::vector<void**>&                   batchPtrs,
                std::vector<size_t>&                   maxElements,
                std::vector<std::vector<size_t>> const& offsets,
                ContractionProblemGemm const&          problem)
        {
            using std::static_pointer_cast;
            std::shared_ptr<ProblemInputs> result;
            if(offsets.empty() || offsets[0].empty())
            {
                auto inputs = new ContractionInputs();
                setContractionInputs(ptrs,
                                     batchPtrs,
                                     m_workspacePristine.get(),
                                     m_cdata,
                                     maxElements,
                                     /*isGPU=*/true,
                                     inputs);
                result = static_pointer_cast<ProblemInputs>(
                    std::shared_ptr<ContractionInputs>(inputs));
            }
            else
            {
                auto inputs = new ContractionGroupedInputs();
                auto dummyBatchPtrs
                    = std::vector<void**>(ContractionProblemGemm::TENSOR::TENSOR_COUNT, nullptr);
                setContractionGroupedInputs(ptrs,
                                            dummyBatchPtrs,
                                            m_workspacePristine.get(),
                                            m_cdata,
                                            /*isGPU=*/true,
                                            problem,
                                            offsets,
                                            inputs);
                result = static_pointer_cast<ProblemInputs>(
                    std::shared_ptr<ContractionGroupedInputs>(inputs));
            }
            return result;
        }

        // For GEMM only
        std::shared_ptr<ProblemInputs>
        DataInitialization::ConvertToProblemInputs(ContractionProblemGemm const& problem,
                                                   bool                          isGPU)
        {
            if(isGPU)
                return buildGPUProblemInputs(
                    m_gpuPtrs, m_gpuBatchPtrs, m_maxElements, m_groupedOffsets, problem);

            // CPU path — uses m_cpuPtrs with dummy batch pointers
            using std::static_pointer_cast;
            std::shared_ptr<ProblemInputs> result;
            if(m_groupedOffsets[0].empty())
            {
                auto inputs = new ContractionInputs();
                auto dummyBatchPtrs = std::vector<void**>(
                    ContractionProblemGemm::TENSOR::TENSOR_COUNT, nullptr);
                setContractionInputs(m_cpuPtrs,
                                     dummyBatchPtrs,
                                     m_workspacePristine.get(),
                                     m_cdata,
                                     m_maxElements,
                                     isGPU,
                                     inputs);
                result = static_pointer_cast<ProblemInputs>(
                    std::shared_ptr<ContractionInputs>(inputs));
            }
            else
            {
                auto inputs = new ContractionGroupedInputs();
                auto dummyBatchPtrs
                    = std::vector<void**>(ContractionProblemGemm::TENSOR::TENSOR_COUNT, nullptr);
                setContractionGroupedInputs(m_cpuPtrs,
                                            dummyBatchPtrs,
                                            m_workspacePristine.get(),
                                            m_cdata,
                                            isGPU,
                                            problem,
                                            m_groupedOffsets,
                                            inputs);
                result = static_pointer_cast<ProblemInputs>(
                    std::shared_ptr<ContractionGroupedInputs>(inputs));
            }
            return result;
        }

        void DataInitialization::refreshRotatingMode1Inputs(ContractionProblemGemm const& problem)
        {
            if(m_rotatingMode != 1 || m_rotatingBuffer == 0)
                return;

            auto mem = m_rm->getRotatingMemory();
            for(size_t j = 1; j < mem.size(); j++)
                for(size_t i = 0; i < m_vdata.size(); i++)
                {
                    auto& desc = problem.tensors()[i];
                    auto  it   = m_vdata[i].pristine.find(desc.dataType());
                    if(it != m_vdata[i].pristine.end())
                    {
                        auto& p = it->second;
                        if(i <= ContractionProblemGemm::TENSOR::METADATA)
                            HIP_CHECK_EXC(hipMemcpy(mem[j][i].data.get(),
                                                    p.gpuInput.current.get(),
                                                    mem[j][i].size,
                                                    hipMemcpyDeviceToDevice));
                    }
                }
        }

        std::shared_ptr<ProblemInputs> DataInitialization::populateTensorSlot(
            ContractionProblemGemm const&    problem,
            std::vector<void*>&              ptrs,
            std::vector<void**>&             batchPtrs,
            std::vector<size_t>&             maxElements,
            std::vector<std::vector<size_t>>& offsets,
            hipMemcpyKind                    copyKind,
            hipStream_t                      targetStream,
            std::vector<SwizzleUpload>*      swizzleStaging,
            bool                             refreshRotatingMode1)
        {
            bool needSwizzle   = problem.swizzleTensorA() || problem.swizzleTensorB();
            bool needMXSwizzle = (problem.mxBlockA() != 0) || (problem.mxBlockB() != 0);

            {
                ScopedTimer t("async_reset_probdep");
                if(m_cpuPtrs.empty() && m_problemDependentData)
                {
                    ScopedTimer t2("async_reset_cpuinit");
                    initializeCPUInputs(problem);
                }
                if(m_problemDependentData)
                {
                    ScopedTimer t2("async_reset_copyvalid");
                    copyValidToGPUBuffer(problem, targetStream != nullptr);
                }
                if(needSwizzle || needMXSwizzle)
                {
                    ScopedTimer t2("async_reset_swizzle");
                    copySwizzledToGPUBuffer(problem, targetStream, swizzleStaging);
                }
            }

            // copyInputs appends grouped offsets, so clear the destination before
            // rebuilding this slot.
            offsets.clear();

            {
                ScopedTimer t("async_reset_copyinputs");
                copyInputs(ptrs,
                           batchPtrs,
                           maxElements,
                           offsets,
                           problem,
                           copyKind,
                           targetStream);
            }

            if(refreshRotatingMode1 && m_rotatingMode == 1 && m_rotatingBuffer > 0)
                refreshRotatingMode1Inputs(problem);

            return buildGPUProblemInputs(ptrs, batchPtrs, maxElements, offsets, problem);
        }

        size_t getRotatingSize(ContractionProblemGemm const& problem,
                               ContractionInputs const&      inputs)
        {
            size_t rotatingSize = 0;
            if(inputs.a != nullptr)
            {
                rotatingSize
                    += problem.tensors()[ContractionProblemGemm::TENSOR::A].totalAllocatedBytes();
            }
            if(inputs.b != nullptr)
            {
                rotatingSize
                    += problem.tensors()[ContractionProblemGemm::TENSOR::B].totalAllocatedBytes();
            }
            if(inputs.c != nullptr && problem.beta())
            {
                rotatingSize
                    += problem.tensors()[ContractionProblemGemm::TENSOR::C].totalAllocatedBytes();
            }
            if(inputs.d != nullptr)
            {
                rotatingSize
                    += problem.tensors()[ContractionProblemGemm::TENSOR::D].totalAllocatedBytes();
            }
            if(inputs.e != nullptr)
            {
                rotatingSize
                    += problem.tensors()[ContractionProblemGemm::TENSOR::E].totalAllocatedBytes();
            }
            if(inputs.scaleA != nullptr)
            {
                rotatingSize += problem.tensors()[ContractionProblemGemm::TENSOR::SCALEA]
                                    .totalAllocatedBytes();
            }
            if(inputs.scaleB != nullptr)
            {
                rotatingSize += problem.tensors()[ContractionProblemGemm::TENSOR::SCALEB]
                                    .totalAllocatedBytes();
            }
            if(inputs.bias != nullptr)
            {
                rotatingSize += problem.tensors()[ContractionProblemGemm::TENSOR::BIAS]
                                    .totalAllocatedBytes();
            }
            if(inputs.scaleAlphaVec != nullptr)
            {
                rotatingSize += problem.tensors()[ContractionProblemGemm::TENSOR::SCALEALPHAVEC]
                                    .totalAllocatedBytes();
            }
            if(inputs.metadata != nullptr)
            {
                rotatingSize += problem.tensors()[ContractionProblemGemm::TENSOR::METADATA]
                                    .totalAllocatedBytes();
            }
            return rotatingSize;
        }

        void* copyRotatingInput(
            const void* src, void* dst, int64_t length, int64_t& dstOffset, hipStream_t stream)
        {
            if(src == nullptr)
                return nullptr;
            void* dstPos = (void*)((uint8_t*)dst + dstOffset);
            HIP_CHECK_EXC(hipMemcpyAsync(dstPos, src, length, hipMemcpyDeviceToDevice, stream));
            dstOffset += length;
            return dstPos;
        }

        ContractionInputs createRotatingInput(ContractionProblemGemm const& problem,
                                              ContractionInputs const&      inputs,
                                              void*                         rotatingPtr,
                                              int64_t&                      offset,
                                              hipStream_t                   stream)
        {
            ContractionInputs newInputs = inputs;
            newInputs.a                 = copyRotatingInput(
                newInputs.a,
                rotatingPtr,
                problem.tensors()[ContractionProblemGemm::TENSOR::A].totalAllocatedBytes(),
                offset,
                stream);
            newInputs.b = copyRotatingInput(
                newInputs.b,
                rotatingPtr,
                problem.tensors()[ContractionProblemGemm::TENSOR::B].totalAllocatedBytes(),
                offset,
                stream);
            if(problem.beta())
                newInputs.c = copyRotatingInput(
                    newInputs.c,
                    rotatingPtr,
                    problem.tensors()[ContractionProblemGemm::TENSOR::C].totalAllocatedBytes(),
                    offset,
                    stream);
            newInputs.d = copyRotatingInput(
                newInputs.d,
                rotatingPtr,
                problem.tensors()[ContractionProblemGemm::TENSOR::D].totalAllocatedBytes(),
                offset,
                stream);
            newInputs.e = copyRotatingInput(
                newInputs.e,
                rotatingPtr,
                problem.tensors()[ContractionProblemGemm::TENSOR::E].totalAllocatedBytes(),
                offset,
                stream);
            newInputs.scaleA = copyRotatingInput(
                newInputs.scaleA,
                rotatingPtr,
                problem.tensors()[ContractionProblemGemm::TENSOR::SCALEA].totalAllocatedBytes(),
                offset,
                stream);
            newInputs.scaleB = copyRotatingInput(
                newInputs.scaleB,
                rotatingPtr,
                problem.tensors()[ContractionProblemGemm::TENSOR::SCALEB].totalAllocatedBytes(),
                offset,
                stream);
            newInputs.bias = copyRotatingInput(
                newInputs.bias,
                rotatingPtr,
                problem.tensors()[ContractionProblemGemm::TENSOR::BIAS].totalAllocatedBytes(),
                offset,
                stream);
            newInputs.scaleAlphaVec
                = copyRotatingInput(newInputs.scaleAlphaVec,
                                    rotatingPtr,
                                    problem.tensors()[ContractionProblemGemm::TENSOR::SCALEALPHAVEC]
                                        .totalAllocatedElements(),
                                    offset,
                                    stream);
            newInputs.metadata = (unsigned char*)copyRotatingInput(
                newInputs.metadata,
                rotatingPtr,
                problem.tensors()[ContractionProblemGemm::TENSOR::METADATA].totalAllocatedBytes(),
                offset,
                stream);
            return newInputs;
        }

        std::vector<std::shared_ptr<ProblemInputs>>
            DataInitialization::prepareRotatingGPUOutput(int32_t maxRotatingBufferNum,
                                                         ContractionProblem const*      problem,
                                                         std::shared_ptr<ProblemInputs> inputs,
                                                         hipStream_t                    stream)
        {
            using std::static_pointer_cast;
            std::vector<std::shared_ptr<ProblemInputs>> inputArr;
            inputArr.push_back(inputs);
            if(m_rotatingBuffer == 0)
                return inputArr;

            if(auto gemmProblem = dynamic_cast<ContractionProblemGemm const*>(problem))
            {
                auto    castInputs   = static_pointer_cast<ContractionInputs>(inputs);
                size_t  rotatingSize = getRotatingSize(*gemmProblem, *castInputs);
                int32_t rotatingNum
                    = std::min(maxRotatingBufferNum, static_cast<int32_t>(ceil((float)m_rotatingBuffer / rotatingSize)))
                      - 1; // Minus the original buffer.

                // <= 0 means don't rotating
                rotatingNum = std::max(0, rotatingNum);

                int32_t totalRotatingSizeNeeded = rotatingNum * rotatingSize;
                std::cout << "Rotating buffer set to: " << m_rotatingBuffer
                          << ". Rotating num: " << rotatingNum
                          << ". rotatingSize: " << rotatingSize << std::endl;
                if(m_rotatingMode == 0)
                {
                    auto rotatingAllocatedSize
                        = m_rm->getDataSize() - m_rm->getDataLargestUnitSize();
                    if(totalRotatingSizeNeeded > rotatingAllocatedSize)
                    {
                        std::cout << "Rotating buffer size: " << rotatingAllocatedSize
                                  << " is not enough for rotating buffer size: " << rotatingSize
                                  << " * " << rotatingNum << " = " << totalRotatingSizeNeeded
                                  << std::endl;
                        throw std::runtime_error("Insufficient rotating buffer size.");
                    }
                    uint8_t* ptr = (uint8_t*)m_rm->getData().get() + m_rm->getDataLargestUnitSize();
                    int64_t  offset = 0;
                    for(size_t i = 0; i < rotatingNum; i++)
                    {
                        auto newInputs = createRotatingInput(
                            *gemmProblem, *castInputs, (void*)ptr, offset, stream);
                        inputArr.push_back(static_pointer_cast<ProblemInputs>(
                            std::make_shared<ContractionInputs>(newInputs)));
                    }
                }
                else
                {
                    auto    mem    = m_rm->getRotatingMemory();
                    int64_t offset = 0;
                    for(size_t i = 0; i < rotatingNum; i++)
                    {
                        ContractionInputs newInputs = *castInputs;
                        newInputs.a                 = mem[i + 1][0].data.get();
                        newInputs.b                 = mem[i + 1][1].data.get();
                        newInputs.c                 = mem[i + 1][2].data.get();
                        newInputs.d                 = mem[i + 1][3].data.get();
                        newInputs.e                 = mem[i + 1][4].data.get();
                        newInputs.bias              = mem[i + 1][5].data.get();
                        newInputs.scaleAlphaVec     = mem[i + 1][6].data.get();
                        newInputs.metadata          = (unsigned char*)mem[i + 1][7].data.get();
                        inputArr.push_back(static_pointer_cast<ProblemInputs>(
                            std::make_shared<ContractionInputs>(newInputs)));
                    }
                }
            }
            else if(auto groupedProblem
                    = dynamic_cast<ContractionProblemGroupedGemm const*>(problem))
            {
                auto   castInputs   = static_pointer_cast<ContractionGroupedInputs>(inputs);
                size_t rotatingSize = 0;
                for(size_t i = 0; i < castInputs->grouped.size(); i++)
                {
                    rotatingSize
                        += getRotatingSize(groupedProblem->gemms[i], castInputs->grouped[i]);
                }
                int32_t rotatingNum
                    = std::min(maxRotatingBufferNum, static_cast<int32_t>(ceil((float)m_rotatingBuffer / rotatingSize)))
                      - 1; // Minus the original buffer.

                // <= 0 means don't rotating
                rotatingNum = std::max(0, rotatingNum);

                int32_t totalRotatingSizeNeeded = rotatingNum * rotatingSize;
                std::cout << "Rotating buffer set to: " << m_rotatingBuffer
                          << ". Rotating num: " << rotatingNum
                          << ". rotatingSize: " << rotatingSize << std::endl;
                if(m_rotatingMode == 0)
                {
                    auto rotatingAllocatedSize
                        = m_rm->getDataSize() - m_rm->getDataLargestUnitSize();
                    if(totalRotatingSizeNeeded > rotatingAllocatedSize)
                    {
                        std::cout << "Rotating buffer size: " << rotatingAllocatedSize
                                  << " is not enough for rotating buffer size: " << rotatingSize
                                  << " * " << rotatingNum << " = " << totalRotatingSizeNeeded
                                  << std::endl;
                        throw std::runtime_error("Insufficient rotating buffer size.");
                    }
                    uint8_t* ptr = (uint8_t*)m_rm->getData().get() + m_rm->getDataLargestUnitSize();
                    int64_t  offset = 0;
                    for(size_t j = 0; j < rotatingNum; j++)
                    {
                        ContractionGroupedInputs newInputs;
                        newInputs.ws = castInputs->ws;
                        for(size_t i = 0; i < castInputs->grouped.size(); i++)
                        {
                            auto newSingleInput = createRotatingInput(groupedProblem->gemms[i],
                                                                      castInputs->grouped[i],
                                                                      (void*)ptr,
                                                                      offset,
                                                                      stream);
                            newInputs.grouped.push_back(newSingleInput);
                        }
                        inputArr.push_back(static_pointer_cast<ProblemInputs>(
                            std::make_shared<ContractionGroupedInputs>(newInputs)));
                    }
                }
                else
                {
                    ContractionGroupedInputs newInputs;
                    newInputs.ws = castInputs->ws;
                    std::vector<size_t> offsets(ContractionProblemGemm::TENSOR::METADATA, 0);
                    auto                mem = m_rm->getRotatingMemory();
                    for(size_t i = 0; i < castInputs->grouped.size(); i++)
                    {
                        auto&             problem        = groupedProblem->gemms[i];
                        ContractionInputs newSingleInput = castInputs->grouped[i];
                        // clang-format off
                        newSingleInput.a             = (void*)((uint8_t*)mem[i + 1][0].data.get() + offsets[0]); offsets[0] += problem.tensors()[ContractionProblemGemm::TENSOR::A].totalAllocatedBytes();
                        newSingleInput.b             = (void*)((uint8_t*)mem[i + 1][1].data.get() + offsets[1]); offsets[1] += problem.tensors()[ContractionProblemGemm::TENSOR::B].totalAllocatedBytes();
                        newSingleInput.c             = (void*)((uint8_t*)mem[i + 1][2].data.get() + offsets[2]); offsets[2] += problem.tensors()[ContractionProblemGemm::TENSOR::C].totalAllocatedBytes();
                        newSingleInput.d             = (void*)((uint8_t*)mem[i + 1][3].data.get() + offsets[3]); offsets[3] += problem.tensors()[ContractionProblemGemm::TENSOR::D].totalAllocatedBytes();
                        newSingleInput.e             = (void*)((uint8_t*)mem[i + 1][4].data.get() + offsets[4]); offsets[4] += problem.tensors()[ContractionProblemGemm::TENSOR::E].totalAllocatedBytes();
                        newSingleInput.bias          = (void*)((uint8_t*)mem[i + 1][5].data.get() + offsets[5]); offsets[5] += problem.tensors()[ContractionProblemGemm::TENSOR::BIAS].totalAllocatedBytes();
                        newSingleInput.scaleAlphaVec = (void*)((uint8_t*)mem[i + 1][6].data.get() + offsets[6]); offsets[6] += problem.tensors()[ContractionProblemGemm::TENSOR::SCALEALPHAVEC].totalAllocatedBytes();
                        newSingleInput.metadata      = (unsigned char*)mem[i + 1][7].data.get() + offsets[7];    offsets[7] += problem.tensors()[ContractionProblemGemm::TENSOR::METADATA].totalAllocatedBytes();
                        // clang-format on
                        newInputs.grouped.push_back(newSingleInput);
                    }
                    inputArr.push_back(static_pointer_cast<ProblemInputs>(
                        std::make_shared<ContractionGroupedInputs>(newInputs)));
                }
            }
            return inputArr;
        }

        std::shared_ptr<ProblemInputs>
        DataInitialization::prepareGPUInputsInternal(
                ContractionProblemGemm const& problem,
                hipStream_t                   targetStream,
                bool                          cpuInputsAlreadyCurrent)
        {
            hipMemcpyKind kind;

            bool needSwizzle = problem.swizzleTensorA() || problem.swizzleTensorB();
            bool needMXSwizzle = (problem.mxBlockA() != 0) || (problem.mxBlockB() != 0);

            if(!cpuInputsAlreadyCurrent)
                ensureCPUInputsCurrent(problem);

            if(m_keepPristineCopyOnGPU && !m_problemDependentData)
            {
                kind = hipMemcpyDeviceToDevice;
            }
            else
            {
                kind = hipMemcpyHostToDevice;
            }

            if(!batchPointersCurrentFor(problem))
            {
                ScopedTimer t("gpu_batch_pointer_init");
                initializeGPUBatchedInputs(problem, targetStream, /*stagingBufferSlot=*/0);
                markBatchPointersCurrent(problem);
            }

            if(m_gpuInit && m_curBoundsCheck == BoundsCheckMode::Disable
               && !m_problemDependentData && !needSwizzle && !needMXSwizzle)
            {
                if(m_elementsToValidate)
                {
                    ScopedTimer t("async_reset_resetoutput");
                    resetOutput(m_gpuPtrs,
                                m_gpuBatchPtrs,
                                m_maxElements,
                                m_groupedOffsets,
                                problem,
                                kind,
                                targetStream);
                }
                markGpuInputsPrepared(problem);
                return m_cachedGPUInputs;
            }
            // buildGPUProblemInputs() copies m_cdata into the cached inputs, so
            // refresh constants before the helper snapshots ProblemInputs.
            initializeConstantInputs(problem);

            m_cachedGPUInputs = populateTensorSlot(problem,
                                                   m_gpuPtrs,
                                                   m_gpuBatchPtrs,
                                                   m_maxElements,
                                                   m_groupedOffsets,
                                                   hipMemcpyDeviceToDevice,
                                                   targetStream,
                                                   nullptr,
                                                   /*refreshRotatingMode1=*/true);
            m_gpuInit = true;

            // Store active slot state in ring[0] only on initial
            // preparation (targetStream == nullptr), not when called
            // from beginAsyncReset where m_gpuPtrs has been moved away.
            if(!targetStream)
            {
                m_gpuPtrsRing[0]      = m_gpuPtrs;
                m_gpuBatchPtrsRing[0] = m_gpuBatchPtrs;
                m_cachedInputsRing[0] = m_cachedGPUInputs;

                initializeAltBufferSets(problem, cpuInputsAlreadyCurrent);
            }
            markGpuInputsPrepared(problem);
            return m_cachedGPUInputs;
        }

        void DataInitialization::fillSlot(
            size_t                        slotIdx,
            ContractionProblemGemm const& problem,
            hipStream_t                   targetStream,
            bool                          cpuInputsAlreadyCurrent)
        {
            // RAII: point gpuInput.current/batch at target slot, restore on exit
            SlotGuard guard(m_vdata, slotIdx);

            hipMemcpyKind kind = (m_keepPristineCopyOnGPU && !m_problemDependentData)
                                     ? hipMemcpyDeviceToDevice
                                     : hipMemcpyHostToDevice;
            std::vector<SwizzleUpload>* swizzleStaging = nullptr;
            if(targetStream)
            {
                swizzleStaging = &m_swizzleUploadStaging[slotIdx];
                swizzleStaging->clear();
            }

            // Local copies — populateTensorSlot rebuilds the vectors in place,
            // so reusing m_groupedOffsets across repeated fillSlot calls would
            // cause unbounded growth.
            auto localMaxElements = m_maxElements;
            auto localOffsets     = m_groupedOffsets;

            // Full path: initialize all tensors into target slot
            if(!cpuInputsAlreadyCurrent)
                ensureCPUInputsCurrent(problem);
            // Enqueue the slot's batch-pointer upload before we snapshot the
            // ProblemInputs for this ring entry.
            initializeGPUBatchedInputs(problem, targetStream, slotIdx);

            m_cachedInputsRing[slotIdx] = populateTensorSlot(problem,
                                                             m_gpuPtrsRing[slotIdx],
                                                             m_gpuBatchPtrsRing[slotIdx],
                                                             localMaxElements,
                                                             localOffsets,
                                                             kind,
                                                             targetStream,
                                                             swizzleStaging,
                                                             /*refreshRotatingMode1=*/false);
        }

        void DataInitialization::initializeAltBufferSets(
            ContractionProblemGemm const& problem,
            bool                          cpuInputsAlreadyCurrent)
        {
            // Early-out when m_gpuPtrsRing[1] is already populated: prevents
            // re-initialization if called redundantly before cancelAsyncReset
            // has cleared the ring for a new problem.
            if(!m_hasAltBuffers || !m_gpuPtrsRing[1].empty())
                return;

            for(size_t slot = 1; slot < m_ring.activeBufferCount(); slot++)
                fillSlot(slot, problem, /*targetStream=*/nullptr, cpuInputsAlreadyCurrent);

            m_altSlotsReady = true;
        }

        void DataInitialization::syncCopyStream()
        {
            if(m_copyStream)
                HIP_CHECK_EXC(hipStreamSynchronize(m_copyStream));
        }

        void DataInitialization::activateRingSlot(size_t slot)
        {
            for(auto& vd : m_vdata)
                for(auto& [dt, pu] : vd.pristine)
                {
                    pu.gpuInput.current = pu.gpuInput.buffers[slot];
                    pu.gpuInput.batch   = pu.gpuInput.batchBufs[slot];
                }
            m_gpuPtrs         = m_gpuPtrsRing[slot];
            m_gpuBatchPtrs    = m_gpuBatchPtrsRing[slot];
            m_cachedGPUInputs = m_cachedInputsRing[slot];
        }

        void DataInitialization::advanceBuffer()
        {
            auto newActiveSlot = m_ring.advance();
            assert(newActiveSlot.has_value());
            activateRingSlot(*newActiveSlot);
            // The new active slot may have an outstanding DMA on m_copyStream;
            // require waitCopyDone before the compute stream reads it.
        }

        // Insert a GPU-side dependency between m_copyStream and computeStream
        // without blocking the CPU; hipStreamWaitEvent is a device-only barrier.
        void DataInitialization::waitCopyDone(hipStream_t computeStream)
        {
            if(!m_ring.needsCopyBarrier())
                return;
            HIP_CHECK_EXC(hipStreamWaitEvent(
                computeStream, m_copyDoneEvents[m_ring.activeSlot()], 0));
            m_ring.markBarrierWaited();
        }

        void DataInitialization::resetOutputsForRingSlot(
            size_t                        targetSlot,
            ContractionProblem const*     problem)
        {
            SlotGuard guard(m_vdata, targetSlot);

            auto localMaxElements = m_maxElements;
            auto localOffsets     = m_groupedOffsets;
            if(localMaxElements.size() < m_vdata.size())
                localMaxElements.resize(m_vdata.size());
            if(localOffsets.size() < m_vdata.size())
                localOffsets.resize(m_vdata.size());

            if(auto gemmProblem = dynamic_cast<ContractionProblemGemm const*>(problem))
            {
                resetOutput(m_gpuPtrsRing[targetSlot],
                            m_gpuBatchPtrsRing[targetSlot],
                            localMaxElements,
                            localOffsets,
                            *gemmProblem,
                            hipMemcpyDeviceToDevice,
                            m_copyStream);
            }
            else if(auto groupedProblem
                    = dynamic_cast<ContractionProblemGroupedGemm const*>(problem))
            {
                assertGroupedRingFastPathInvariant(*groupedProblem);
                resetOutput(m_gpuPtrsRing[targetSlot],
                            m_gpuBatchPtrsRing[targetSlot],
                            localMaxElements,
                            localOffsets,
                            groupedProblem->gemms[0],
                            hipMemcpyDeviceToDevice,
                            m_copyStream);
            }
        }

        // Cancel any pending async resets and invalidate alt buffers
        // (e.g., when switching to a new problem whose data differs).
        void DataInitialization::cancelAsyncReset()
        {
            auto const oldActiveSlot  = m_ring.activeSlot();
            bool const hadPendingWork = m_ring.hasPendingWork();

            if(hadPendingWork)
                syncCopyStream();

            m_ring.cancel();

            // Restore the cached slot-0 aliases before clearing alternate-ring
            // storage.  We capture the old active slot up front so this still
            // happens even when no sync was needed.
            if(oldActiveSlot != 0)
            {
                activateRingSlot(0);
            }
            // Clear alt ring slots so initializeAltBufferSets re-runs for the
            // new problem.  Without this, the guard in initializeAltBufferSets
            // (!m_gpuPtrsRing[1].empty()) would short-circuit and leave stale
            // data from the previous problem's layout in the alt slots.
            for(size_t i = 1; i < MAX_BUFFER_SETS; i++)
            {
                m_gpuPtrsRing[i].clear();
                m_gpuBatchPtrsRing[i].clear();
                m_cachedInputsRing[i].reset();
                m_swizzleUploadStaging[i].clear();
            }
            m_batchPointerSignatureValid   = false;
            m_preparedProblemSignatureValid = false;
            m_altSlotsReady                = false;
        }

        // Kick off async reset of the next free buffer slot in the ring
        // on m_copyStream.  The caller must waitCopyDone() before
        // using the buffer (done in main.cpp before benchmark_runs).
        void DataInitialization::beginAsyncReset(ContractionProblem const* problem)
        {
            if(!ringEligible())
                return;
            auto targetIdx = m_ring.nextPrimeSlot();
            if(!targetIdx)
                return; // all non-active slots already have pending DMA
            size_t const targetSlot = *targetIdx;

            // Warm path: this target slot's input tensors have already
            // been prepared for the current problem, either by the
            // initial active-slot preparation or by
            // initializeAltBufferSets/fillSlot for alternate slots.
            // Skipping output reset is safe only when the run does not need
            // initialized D/output sentinels for validation; in that case
            // the next GEMM dispatch is expected to define every D element
            // that will be observed.
            // Do not treat D overwrite as a universal invariant:
            // validation-sensitive reuse must reset output tensors before
            // recording copy completion. Record a no-op event as a sync
            // marker.
            if(m_altSlotsReady)
            {
                if(m_warmOutputResetRequired)
                {
                    ScopedTimer resetTimer("async_reset_warm_resetoutput");
                    resetOutputsForRingSlot(targetSlot, problem);
                }
                HIP_CHECK_EXC(hipEventRecord(m_copyDoneEvents[targetSlot], m_copyStream));
                m_ring.markSlotPrimed();
                return;
            }

            // Fill target slot directly — no save/restore of working state.
            {
                ScopedTimer prepTimer("async_reset_prepare");
                if(auto gemmProblem = dynamic_cast<ContractionProblemGemm const*>(problem))
                    fillSlot(targetSlot, *gemmProblem, m_copyStream);
                else if(auto groupedProblem
                        = dynamic_cast<ContractionProblemGroupedGemm const*>(problem))
                {
                    assertGroupedRingFastPathInvariant(*groupedProblem);
                    fillSlot(targetSlot, groupedProblem->gemms[0], m_copyStream);
                }
            }

            HIP_CHECK_EXC(hipEventRecord(m_copyDoneEvents[targetSlot], m_copyStream));
            m_ring.markSlotPrimed();
        }

        DataInitialization::~DataInitialization()
        {
            for(size_t i = 0; i < MAX_BUFFER_SETS; i++)
            {
                if(m_copyDoneEvents[i])
                {
                    hipError_t e = hipEventDestroy(m_copyDoneEvents[i]);
                    if(e)
                        std::cerr << "~DataInitialization: hipEventDestroy failed: "
                                  << hipGetErrorString(e) << std::endl;
                }
            }
            if(m_copyStream)
            {
                hipError_t e = hipStreamSynchronize(m_copyStream);
                if(e)
                    std::cerr << "~DataInitialization: hipStreamSynchronize failed: "
                              << hipGetErrorString(e) << std::endl;
                e = hipStreamDestroy(m_copyStream);
                if(e)
                    std::cerr << "~DataInitialization: hipStreamDestroy failed: "
                              << hipGetErrorString(e) << std::endl;
            }
            if(m_pinnedBatchStaging)
            {
                hipError_t e = hipHostFree(m_pinnedBatchStaging);
                if(e)
                    std::cerr << "~DataInitialization: hipHostFree failed: "
                              << hipGetErrorString(e) << std::endl;
            }
        }
    } // namespace Client
} // namespace TensileLite
