/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "TensorDataManipulation.hpp"
#include "Utility.hpp"
// #include "DataInitializationTyped.hpp"

#include <Tensile/Utils.hpp>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <list>
#include <map>
#include <tuple>

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
            auto getPruneIndex = [](PruneSparseMode pruneMode, size_t* index1, size_t* index2) {
                if(pruneMode == PruneSparseMode::PruneRandom)
                    pruneMode = static_cast<PruneSparseMode>(
                        rand() % (static_cast<int>(PruneSparseMode::MaxPruneMode) - 1) + 1);
                switch(pruneMode)
                {
                case PruneSparseMode::PruneXX00:
                    *index1 = 2;
                    *index2 = 3;
                    break;
                case PruneSparseMode::PruneX0X0:
                    *index1 = 1;
                    *index2 = 3;
                    break;
                case PruneSparseMode::Prune0XX0:
                    *index1 = 0;
                    *index2 = 3;
                    break;
                case PruneSparseMode::PruneX00X:
                    *index1 = 1;
                    *index2 = 2;
                    break;
                case PruneSparseMode::Prune0X0X:
                    *index1 = 0;
                    *index2 = 2;
                    break;
                case PruneSparseMode::Prune00XX:
                    *index1 = 0;
                    *index2 = 1;
                    break;
                default:
                    throw std::runtime_error("prune mode is not allowed.");
                    break;
                }
            };
#pragma omp parallel for
            for(size_t loop = 0; loop < loop_count; loop++)
            {
                std::vector<size_t> coord(tensor.dimensions(), 0);
                CoordNumberedExclude(
                    loop, coord.begin(), coord.end(), sizes.begin(), sizes.end(), pruneDim);
                for(size_t pruneDimIdx = 0; pruneDimIdx < pruneDimSize;
                    pruneDimIdx += 4) //traverse along pruneDim
                {
                    size_t pruneIdx1, pruneIdx2;
                    getPruneIndex(mode, &pruneIdx1, &pruneIdx2);
                    coord[pruneDim]            = pruneDimIdx + pruneIdx1;
                    array[tensor.index(coord)] = static_cast<T>(0);
                    coord[pruneDim]            = pruneDimIdx + pruneIdx2;
                    array[tensor.index(coord)] = static_cast<T>(0);
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
                                 size_t                  dim)
        {
            auto const& sizes      = tensor.sizes();
            auto const& sizesC     = tensorC.sizes();
            auto        sizesMeta  = tensorMeta.sizes();
            auto        count      = CoordCount(sizes.begin(), sizes.end());
            size_t      dimSize    = sizes[dim];
            size_t      loop_count = count / dimSize;

            if(dimSize % 4 != 0)
                throw std::runtime_error("compressed dimension size must be multiple of 4.");

            std::memset((void*)dstCompressed,
                        0,
                        TypeInfo<T>::ElementSize * tensorC.totalAllocatedElements());
            std::memset((void*)dstMeta, 0, tensorMeta.totalLogicalElements());

#pragma omp parallel for
            for(size_t loop = 0; loop < loop_count; loop++)
            {
                std::vector<size_t> coord(tensor.dimensions());
                std::vector<size_t> coordC(tensorC.dimensions());
                std::vector<size_t> coordMeta(tensorMeta.dimensions());
                CoordNumberedExclude(
                    loop, coord.begin(), coord.end(), sizes.begin(), sizes.end(), dim);
                CoordNumberedExclude(
                    loop, coordC.begin(), coordC.end(), sizesC.begin(), sizesC.end(), dim);
                //metadata is always a tranpose matrix, so the dimension will always at 0.
                CoordNumberedExclude(loop,
                                     coordMeta.begin(),
                                     coordMeta.end(),
                                     sizesMeta.begin(),
                                     sizesMeta.end(),
                                     0);

                coordMeta[0] = 0;

                for(size_t compressDimIdx = 0; compressDimIdx < dimSize;
                    compressDimIdx += 4) //traverse along compressdim
                {
                    size_t metaData    = 0;
                    size_t metaIdx     = 0;
                    size_t dstDimCoord = compressDimIdx / 4 * 2;

                    for(size_t i = 0; i < 4; i++)
                    {
                        //src coord
                        coord[dim] = compressDimIdx + i;
                        T srcData  = src[tensor.index(coord)];
                        if(srcData != static_cast<T>(0))
                        {
                            if(metaIdx > 2)
                                throw std::runtime_error("Sparse matrix must contain 2 zero "
                                                         "elements of each 4 elements.");

                            //dst coord
                            coordC[dim]                          = dstDimCoord + metaIdx;
                            dstCompressed[tensorC.index(coordC)] = srcData;
                            metaData |= i << (2 * metaIdx);
                            metaIdx++;
                        }

                        if(i == 3 && metaIdx < 2)
                        {
                            if(metaIdx == 0) //all zeros
                            {
                                coordC[dim]                          = dstDimCoord;
                                dstCompressed[tensorC.index(coordC)] = static_cast<T>(0);
                                metaData                             = 0;
                                metaIdx++;
                            }
                            if(metaIdx == 1) //3 zeros at least
                            {
                                if(metaData != 3) //last element is zero
                                {
                                    coordC[dim]                          = dstDimCoord + metaIdx;
                                    dstCompressed[tensorC.index(coordC)] = static_cast<T>(0);
                                    metaData |= i << (2 * metaIdx);
                                }
                                else //last element is not zero
                                {
                                    coordC[dim]                          = dstDimCoord;
                                    dstCompressed[tensorC.index(coordC)] = static_cast<T>(0);
                                    coordC[dim]                          = dstDimCoord + metaIdx;
                                    dstCompressed[tensorC.index(coordC)] = srcData;
                                    metaData = (metaData << (2 * metaIdx));
                                }
                                metaIdx++;
                            }
                        }
                    }
                    //meta Data coord
                    size_t shift4bit = (compressDimIdx / 4 % 2) * 4;
                    coordMeta[0]     = compressDimIdx / 8;
                    //calculate flatten index of dstMeta
                    size_t flattenIdx = CoordFlattenIndex(
                        coordMeta.begin(), coordMeta.end(), sizesMeta.begin(), sizesMeta.end());
                    // store metaData to dstMeta
                    dstMeta[flattenIdx] |= metaData << shift4bit;
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
                                         size_t                  dim)
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
                                        size_t                  dim)
        {
            pruneSparseArray(mode, dstPruned, tensor, dim);
            compressSparseArray(
                dstCompressed, dstMeta, dstPruned, tensor, tensorC, tensorMeta, dim);
        }

        void initCPUSparseInput(PruneSparseMode         mode,
                                void*                   dstPruned,
                                void*                   dstCompressed,
                                void*                   dstMeta,
                                TensorDescriptor const& tensor,
                                TensorDescriptor const& tensorC,
                                TensorDescriptor const& tensorMeta,
                                size_t                  dim)
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
                                           dim);
                break;
            case rocisa::DataType::BFloat16:
                initCPUSparseInputTemplate(mode,
                                           (BFloat16*)(dstPruned),
                                           (BFloat16*)(dstCompressed),
                                           (unsigned char*)(dstMeta),
                                           tensor,
                                           tensorC,
                                           tensorMeta,
                                           dim);
                break;
            case rocisa::DataType::Int8:
                initCPUSparseInputTemplate(mode,
                                           (int8_t*)(dstPruned),
                                           (int8_t*)(dstCompressed),
                                           (unsigned char*)(dstMeta),
                                           tensor,
                                           tensorC,
                                           tensorMeta,
                                           dim);
                break;
            case rocisa::DataType::Float8:
                initCPUSparseInputTemplate(mode,
                                           (Float8*)(dstPruned),
                                           (Float8*)(dstCompressed),
                                           (unsigned char*)(dstMeta),
                                           tensor,
                                           tensorC,
                                           tensorMeta,
                                           dim);
                break;
            case rocisa::DataType::BFloat8:
                initCPUSparseInputTemplate(mode,
                                           (BFloat8*)(dstPruned),
                                           (BFloat8*)(dstCompressed),
                                           (unsigned char*)(dstMeta),
                                           tensor,
                                           tensorC,
                                           tensorMeta,
                                           dim);
                break;
            case rocisa::DataType::Float8_fnuz:
                initCPUSparseInputTemplate(mode,
                                           (Float8_fnuz*)(dstPruned),
                                           (Float8_fnuz*)(dstCompressed),
                                           (unsigned char*)(dstMeta),
                                           tensor,
                                           tensorC,
                                           tensorMeta,
                                           dim);
                break;
            case rocisa::DataType::BFloat8_fnuz:
                initCPUSparseInputTemplate(mode,
                                           (BFloat8_fnuz*)(dstPruned),
                                           (BFloat8_fnuz*)(dstCompressed),
                                           (unsigned char*)(dstMeta),
                                           tensor,
                                           tensorC,
                                           tensorMeta,
                                           dim);
                break;
            default:
                throw std::runtime_error("SparseMatrix doesn't support");
            }
        }

        void initGPUBatchedInput(void*                      base,
                                 void**                     array,
                                 TensorDescriptor const&    tensor,
                                 const std::vector<size_t>& batchIdx)
        {
            std::vector<size_t> batchSizes;
            std::vector<size_t> batchStrides;
            for(auto& idx : batchIdx)
            {
                batchSizes.push_back(tensor.sizes().at(idx));
                batchStrides.push_back(tensor.strides().at(idx));
            }
            std::vector<size_t> coord(batchSizes.size(), 0);

            auto      count    = CoordCount(batchSizes.begin(), batchSizes.end());
            uint8_t** cpuArray = (uint8_t**)std::malloc(count * sizeof(void*));
            for(size_t idx = 0; idx < count; idx++)
            {
                CoordNumbered(
                    idx, coord.begin(), coord.end(), batchSizes.begin(), batchSizes.end());
                cpuArray[idx] = (uint8_t*)base;
                for(size_t i = 0; i < batchSizes.size(); i++)
                {
                    cpuArray[idx] += coord[i] * batchStrides[i];
                }
            }

            HIP_CHECK_EXC(hipMemcpy(array, cpuArray, count * sizeof(void*), hipMemcpyHostToDevice));

            std::free(cpuArray);
        }

        void* copyBadInputBuffers(const TensorDescriptor& descriptor,
                                  void*                   dst,
                                  void*                   src,
                                  void*                   bad,
                                  size_t                  totalElements,
                                  hipMemcpyKind           kind)
        {
            HIP_CHECK_EXC(
                hipMemcpy(dst,
                          src,
                          DataTypeInfo::Get(descriptor.dataType()).elementSize * totalElements,
                          kind));
            ptrdiff_t dPadding = totalElements - descriptor.totalAllocatedElements();
            dPadding *= descriptor.elementBytes();
            void* dstOffset = (void*)((uint8_t*)dst + dPadding / 2);
            TensileLite::hip::CopyTensorVoid(dstOffset, src, descriptor, kind);
            return dstOffset;
        }

        void* copyNaNInputBuffers(const TensorDescriptor& descriptor,
                                  void*                   dst,
                                  void*                   src,
                                  size_t                  totalElements,
                                  hipMemcpyKind           kind,
                                  ptrdiff_t               customPadding = -1)
        {
            const ptrdiff_t dPadding = (customPadding == -1)
                                           ? totalElements - descriptor.totalAllocatedElements()
                                           : customPadding;
            const size_t    numElementsToCopy
                = (customPadding == -1) ? descriptor.totalAllocatedElements()
                                        : (descriptor.totalAllocatedElements() + customPadding);
            uint8_t* dstOffset = (uint8_t*)dst + (dPadding * descriptor.elementBytes());
            HIP_CHECK_EXC(
                hipMemcpy(dstOffset, src, descriptor.elementBytes() * numElementsToCopy, kind));
            return dstOffset;
        }

        void* copyInputBuffers(const TensorDescriptor& descriptor,
                               void*                   dst,
                               void*                   src,
                               size_t                  totalElements,
                               hipMemcpyKind           kind)
        {
            HIP_CHECK_EXC(hipMemcpy(dst, src, descriptor.elementBytes() * totalElements, kind));
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

        {
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

            bool isRMInit = false;
            for(auto const& p : problemFactory.problems())
            {
                if(auto ptr = dynamic_cast<ContractionProblemGemm const*>(p.get()))
                {
                    std::vector<size_t>           vec_rm;
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
                            numAllocatedBytes
                                = numAllocatedElements * rocisa::GetElementSize(dataType);
                        }

                        pristine.maxElements = std::max(pristine.maxElements, numAllocatedElements);

                        if(m_rotatingBuffer)
                        {
                            if(i <= ContractionProblemGemm::TENSOR::METADATA)
                            {
                                if(i == ContractionProblemGemm::TENSOR::C && problem.beta() == 0.0)
                                {
                                    vec_rm.push_back(0);
                                }
                                else
                                {
                                    vec_rm.push_back(numAllocatedBytes);
                                }
                            }
                        }
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
                    if(m_rotatingBuffer)
                    {
                        if(!isRMInit)
                        {
                            m_rm     = std::make_shared<RotatingMemory>(vec_rm.size());
                            isRMInit = true;
                        }
                        m_rm->addRotatingSize(vec_rm);
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
                    std::vector<size_t> vec_rm;
                    auto                gElements
                        = std::vector<std::map<rocisa::DataType, gElement>>(m_vdata.size());
                    for(auto const& problem : problems.gemms)
                    {
                        std::vector<size_t> tmp_rm;
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
                            if(m_rotatingBuffer)
                            {
                                if(i <= ContractionProblemGemm::TENSOR::METADATA)
                                {
                                    if(i == ContractionProblemGemm::TENSOR::C
                                       && problem.beta() == 0.0)
                                    {
                                        tmp_rm.push_back(0);
                                    }
                                    else
                                    {
                                        tmp_rm.push_back(
                                            problem.tensors()[i].totalAllocatedBytes());
                                    }
                                }
                            }
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
                        if(vec_rm.empty())
                        {
                            vec_rm = tmp_rm;
                        }
                        else
                        {
                            if(vec_rm.size() != tmp_rm.size())
                            {
                                throw std::runtime_error("Unable to update vec_rm.");
                            }
                            for(size_t i = 0; i < tmp_rm.size(); i++)
                            {
                                vec_rm[i] += tmp_rm[i];
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
                    if(m_rotatingBuffer)
                    {
                        if(!isRMInit)
                        {
                            m_rm     = std::make_shared<RotatingMemory>(vec_rm.size());
                            isRMInit = true;
                        }
                        m_rm->addRotatingSize(vec_rm);
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

                    size_t dataTypeSize = DataTypeInfo::Get(p->first).elementSize;
                    if(m_curBoundsCheck == BoundsCheckMode::NaN)
                    {
                        p->second.maxElements += 1024;
                    }
                    else if(m_curBoundsCheck == BoundsCheckMode::GuardPageFront
                            || m_curBoundsCheck == BoundsCheckMode::GuardPageBack)
                    {
                        size_t roundUpSize = pageSize / dataTypeSize;
                        p->second.maxElements
                            = RoundUpToMultiple<size_t>(p->second.maxElements, roundUpSize);
                        // No bias page guard
                    }
                    ++p;
                }
                std::cout << "Tensor name " << m_vdata[i].name << " init mode "
                          << ToString(m_vdata[i].init) << std::endl;
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
                        HIP_CHECK_EXC(hipMemcpy(pUnit.gpuInput.valid.get(),
                                                pUnit.cpuInput.valid.get(),
                                                dataTypeSize * pUnit.maxElements,
                                                hipMemcpyHostToDevice));
                    }
                    // Init and copy bad from cpu to gpu
                    if(pUnit.gpuInput.bad && pUnit.cpuInput.bad)
                    {
                        initArray(p.first,
                                  InitMode::BadOutput,
                                  pUnit.cpuInput.bad.get(),
                                  pUnit.maxElements);
                        HIP_CHECK_EXC(hipMemcpy(pUnit.gpuInput.bad.get(),
                                                pUnit.cpuInput.bad.get(),
                                                dataTypeSize * pUnit.maxElements,
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
                    size_t size  = DataTypeInfo::Get(p.first).elementSize * pUnit.maxElements;
                    if(size <= 0)
                    {
                        throw std::runtime_error("Size not exists.");
                    }

                    std::stringstream ss;
                    ss << "Failed to allocate cpu input " << it.name << " type("
                       << DataTypeInfo::Get(p.first).abbrev
                       << "), element size: " << DataTypeInfo::Get(p.first).elementSize
                       << ", element length: " << pUnit.maxElements;

                    if(!pUnit.cpuInput.current)
                    {
                        auto ptr
                            = std::shared_ptr<void>(std::malloc(size), [](auto p) { free(p); });
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
                        auto ptr
                            = std::shared_ptr<void>(std::malloc(size), [](auto p) { free(p); });
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
                        auto ptr
                            = std::shared_ptr<void>(std::malloc(size), [](auto p) { free(p); });
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

        void DataInitialization::allocNewGPUInputs()
        {
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
                    size_t size  = DataTypeInfo::Get(p.first).elementSize * pUnit.maxElements;

                    std::stringstream ss;
                    ss << "[" << tensorIdx << "]"
                       << "Failed to allocate gpu input " << it.name << " type("
                       << DataTypeInfo::Get(p.first).abbrev
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
                        pUnit.gpuInput.current = ptr;
                        std::string n          = "batch" + it.name;
                        auto        batch_ptr
                            = allocNewGPUBuffer<void*>(n.c_str(), sizeof(uint8_t*) * m_maxBatch);
                        if(batch_ptr == nullptr)
                            throw std::runtime_error("out of batch gpu memory");
                        pUnit.gpuInput.batch = batch_ptr;
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
        }

        void DataInitialization::initializeGPUBatchedInputs(ContractionProblemGemm const& problem)
        {
            auto batchIdxs = problem.batchIndices();
            // FIXME: batch not supported for bias
            for(size_t i = 0; i < 4 /*m_vdata.size()*/; i++)
            {
                auto it = m_vdata[i].pristine.find(problem.tensors()[i].dataType());
                if(it == m_vdata[i].pristine.end())
                    continue;
                auto&               pUnit = m_vdata[i].pristine[problem.tensors()[i].dataType()];
                std::vector<size_t> batchIdx(batchIdxs.size(), 0);
                ptrdiff_t           padding = 0;
                for(size_t j = 0; j < batchIdxs.size(); j++)
                {
                    switch(i)
                    {
                    case 0:
                        batchIdx[j] = batchIdxs[j].a;
                        break;
                    case 1:
                        batchIdx[j] = batchIdxs[j].b;
                        break;
                    case 2:
                        batchIdx[j] = batchIdxs[j].c;
                        break;
                    case 3:
                        batchIdx[j] = batchIdxs[j].d;
                        break;
                    }
                }
                if(m_curBoundsCheck == BoundsCheckMode::NaN)
                {
                    padding
                        = (pUnit.maxElements - problem.tensors()[i].totalAllocatedElements()) / 2;
                }
                else if(m_curBoundsCheck == BoundsCheckMode::GuardPageBack)
                {
                    padding = pUnit.maxElements - problem.tensors()[i].totalAllocatedElements();

                    if((problem.swizzleTensorA() && i == ContractionProblemGemm::TENSOR::A)
                       || (problem.swizzleTensorB() && i == ContractionProblemGemm::TENSOR::B))
                    {
                        //TODO: support more swizzle types,
                        //      currently, if A then it means MiM = 16, if B then it means MiN = 16
                        size_t MiM_N = 16, MiK = 0, MiKv = 0, PackK = 0;
                        calculateKforSwizzling(problem.tensors()[i].dataType(), MiK, MiKv, PackK);
                        padding = pUnit.maxElements
                                  - getSwizzledTensorNumAllocatedElements(
                                      problem.tensors()[i], MiM_N, MiK, PackK);
                    }
                }
                padding *= DataTypeInfo::Get(problem.tensors()[i].dataType()).elementSize;
                uint8_t* offset = (uint8_t*)pUnit.gpuInput.current.get();
                initGPUBatchedInput((void*)(offset + padding),
                                    pUnit.gpuInput.batch.get(),
                                    problem.tensors()[i],
                                    batchIdx);

                if(problem.useBias() && problem.biasSrc() == i)
                {
                    auto& pUnitBias = m_vdata[ContractionProblemGemm::TENSOR::BIAS]
                                          .pristine[problem.bias().dataType()];
                    if(m_curBoundsCheck == BoundsCheckMode::NaN)
                    {
                        padding = (pUnitBias.maxElements
                                   - problem.tensors()[ContractionProblemGemm::TENSOR::BIAS]
                                         .totalAllocatedElements())
                                  / 2;
                    }
                    else if(m_curBoundsCheck == BoundsCheckMode::GuardPageBack)
                    {
                        padding = pUnitBias.maxElements
                                  - problem.tensors()[ContractionProblemGemm::TENSOR::BIAS]
                                        .totalAllocatedElements();
                    }
                    padding
                        *= DataTypeInfo::Get(
                               problem.tensors()[ContractionProblemGemm::TENSOR::BIAS].dataType())
                               .elementSize;
                    uint8_t* offset = (uint8_t*)pUnitBias.gpuInput.current.get();
                    initGPUBatchedInput((void*)(offset + padding),
                                        pUnitBias.gpuInput.batch.get(),
                                        problem.tensors()[ContractionProblemGemm::TENSOR::BIAS],
                                        batchIdx);
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
                        padding *= DataTypeInfo::Get(t.dataType()).elementSize;
                        return padding;
                    };

                    auto& pUnitM = m_vdata[ContractionProblemGemm::TENSOR::METADATA]
                                       .pristine[problem.metadata().dataType()];

                    padding = caculate_padding(
                        m_curBoundsCheck,
                        pUnitM,
                        problem.tensors()[ContractionProblemGemm::TENSOR::METADATA]);
                    offset = (uint8_t*)pUnitM.gpuInput.current.get();
                    initGPUBatchedInput((void*)(offset + padding),
                                        pUnitM.gpuInput.batch.get(),
                                        problem.tensors()[ContractionProblemGemm::TENSOR::METADATA],
                                        batchIdx);

                    auto& pUnitCp = m_vdata[ContractionProblemGemm::TENSOR::COMPRESSED]
                                        .pristine[problem.compressed().dataType()];
                    padding = caculate_padding(
                        m_curBoundsCheck,
                        pUnitCp,
                        problem.tensors()[ContractionProblemGemm::TENSOR::COMPRESSED]);
                    offset = (uint8_t*)pUnitCp.gpuInput.current.get();
                    initGPUBatchedInput(
                        (void*)(offset + padding),
                        pUnitCp.gpuInput.batch.get(),
                        problem.tensors()[ContractionProblemGemm::TENSOR::COMPRESSED],
                        batchIdx);
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
                            if(p.second.initDescriptor[j] != tensors[i])
                            {
                                p.second.initDescriptor[j] = tensors[i];
                                initArray(p.first,
                                          m_vdata[i].init,
                                          (void*)((int8_t*)p.second.cpuInput.valid.get()
                                                  + gemmInitOffset),
                                          tensors[i]);
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
                                    rocisa::DataType        tDataType;
                                    if(problem.gemms[j].sparse() == 2)
                                    {
                                        tDim      = problem.gemms[j].boundIndices()[0].b;
                                        tDataType = problem.gemms[j].b().dataType();
                                    }
                                    else
                                    {
                                        tDim      = problem.gemms[j].boundIndices()[0].a;
                                        tDataType = problem.gemms[j].a().dataType();
                                    }

                                    const TensorDescriptor& tM = problem.gemms[j].metadata();
                                    const TensorDescriptor& tC = problem.gemms[j].compressed();
                                    auto& pUnitM = m_vdata[ContractionProblemGemm::TENSOR::METADATA]
                                                       .pristine[p.first];
                                    auto& pUnitCp
                                        = m_vdata[ContractionProblemGemm::TENSOR::COMPRESSED]
                                              .pristine[p.first];
                                    pUnitM.initDescriptor[j]
                                        = tensors[ContractionProblemGemm::TENSOR::METADATA];
                                    pUnitCp.initDescriptor[j]
                                        = tensors[ContractionProblemGemm::TENSOR::COMPRESSED];
                                    initCPUSparseInput(
                                        m_pruneMode,
                                        (char*)p.second.cpuInput.valid.get() + gemmInitOffset,
                                        (char*)pUnitCp.cpuInput.valid.get() + gemmInitOffset,
                                        (char*)pUnitM.cpuInput.valid.get() + gemmInitOffset,
                                        t,
                                        tC,
                                        tM,
                                        tDim);
                                }
                            }
                            gemmInitOffset
                                += p.second.groupedGemmOffsets[j] * tensors[i].elementBytes();
                        }
                    }
                }
            }
        }

        void DataInitialization::initializeCPUInputs(ContractionProblemGemm const& problem)
        {
            auto& tensors = problem.tensors();
            for(size_t i = 0; i < m_vdata.size(); i++)
            {
                if(i == ContractionProblemGemm::TENSOR::COMPRESSED
                   or i == ContractionProblemGemm::TENSOR::METADATA)
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
                                                   tDim);
                            }
                        }
                    }
                }
            }
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
                    case rocisa::DataType::Int64:
                    case rocisa::DataType::XFloat32:
                    case rocisa::DataType::Count:
                    case rocisa::DataType::Float8BFloat8:
                    case rocisa::DataType::BFloat8Float8:
                    case rocisa::DataType::Float8BFloat8_fnuz:
                    case rocisa::DataType::BFloat8Float8_fnuz:;
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
                                            hipMemcpyKind                     kind)
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
                                                      kind);
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
                                                      swizzlePadding);
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
                                                   kind);
                        if(ptr == nullptr)
                        {
                            std::runtime_error("output ptr is null when copy input");
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
                                             hipMemcpyKind                     kind)
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
                                               kind);
                    if(ptr == nullptr)
                    {
                        std::runtime_error("output ptr is null when copy input");
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

        void DataInitialization::copyValidToGPUBuffer(ContractionProblemGemm const& problem)
        {
            for(size_t i = 0; i < m_vdata.size(); i++)
            {
                bool needSwizzle
                    = (problem.swizzleTensorA() && i == ContractionProblemGemm::TENSOR::A)
                      || (problem.swizzleTensorB() && i == ContractionProblemGemm::TENSOR::B);
                //Copy swizzle tensor would be in copySwizzledToGPUBuffer
                if(needSwizzle)
                    continue;
                void* ptr  = nullptr;
                auto& desc = problem.tensors()[i];
                auto  it   = m_vdata[i].pristine.find(desc.dataType());
                if(it == m_vdata[i].pristine.end())
                    continue;
                auto& p = m_vdata[i].pristine[desc.dataType()];
                if(p.gpuInput.valid.get() == nullptr || p.cpuInput.valid.get() == nullptr)
                    continue;
                ptr = copyInputBuffers(desc,
                                       p.gpuInput.valid.get(),
                                       p.cpuInput.valid.get(),
                                       p.maxElements,
                                       hipMemcpyHostToDevice);
                if(ptr == nullptr)
                    std::__throw_runtime_error("error");
            }
        }

        void DataInitialization::copySwizzledToGPUBuffer(ContractionProblemGemm const& problem)
        {
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

                void* ptr{};

                if(needSwizzle)
                {
                    using Tensor = Tensor::Manipulation::Tensor;
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
                            Tensor& permuted = g_swizzleCache.at(swizzleKey);
                            ptr              = copyInputBuffers(desc,
                                                   p.gpuInput.valid.get(),
                                                   permuted.as<void>(),
                                                   permuted.getDesc().flattenSize(),
                                                   hipMemcpyHostToDevice);
                        }
                        else
                        {
                            ptr = p.gpuInput.valid.get();
                        }
                    }
                    else
                    {
                        auto tmpTensor = Tensor({tiledSize, unrolledSize}, desc.elementBytes());

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
                        Tensor permuted = permute(paddedTensor, {0, 2, 3, 1, 4});
                        ptr             = copyInputBuffers(desc,
                                               p.gpuInput.valid.get(),
                                               permuted.as<void>(),
                                               permuted.getDesc().flattenSize(),
                                               hipMemcpyHostToDevice);
                        g_swizzleCache.emplace(swizzleKey, std::move(permuted));
                    }
                }
                else
                {
                    ptr = copyInputBuffers(desc,
                                           p.gpuInput.valid.get(),
                                           p.cpuInput.valid.get(),
                                           p.maxElements,
                                           hipMemcpyHostToDevice);
                }

                if(ptr == nullptr)
                    std::__throw_runtime_error("error");
            }
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

                u8Ptr[ContractionProblemGemm::TENSOR::A]
                    += offsets[ContractionProblemGemm::TENSOR::A][idx] * problem.a().elementBytes();
                u8Ptr[ContractionProblemGemm::TENSOR::B]
                    += offsets[ContractionProblemGemm::TENSOR::B][idx] * problem.b().elementBytes();
                u8Ptr[ContractionProblemGemm::TENSOR::C]
                    += offsets[ContractionProblemGemm::TENSOR::C][idx] * problem.c().elementBytes();
                u8Ptr[ContractionProblemGemm::TENSOR::D]
                    += offsets[ContractionProblemGemm::TENSOR::D][idx] * problem.d().elementBytes();
                if(u8Ptr[ContractionProblemGemm::TENSOR::E] != nullptr)
                {
                    u8Ptr[ContractionProblemGemm::TENSOR::E]
                        += offsets[ContractionProblemGemm::TENSOR::E][idx]
                           * problem.tensors()[ContractionProblemGemm::TENSOR::E].elementBytes();
                }
                if(u8Ptr[ContractionProblemGemm::TENSOR::BIAS] != nullptr)
                {
                    u8Ptr[ContractionProblemGemm::TENSOR::BIAS]
                        += offsets[ContractionProblemGemm::TENSOR::BIAS][idx]
                           * problem.tensors()[ContractionProblemGemm::TENSOR::BIAS].elementBytes();
                }
                if(u8Ptr[ContractionProblemGemm::TENSOR::SCALEA] != nullptr)
                {
                    u8Ptr[ContractionProblemGemm::TENSOR::SCALEA]
                        += offsets[ContractionProblemGemm::TENSOR::SCALEA][idx]
                           * problem.tensors()[ContractionProblemGemm::TENSOR::SCALEA]
                                 .elementBytes();
                }
                if(u8Ptr[ContractionProblemGemm::TENSOR::SCALEB] != nullptr)
                {
                    u8Ptr[ContractionProblemGemm::TENSOR::SCALEB]
                        += offsets[ContractionProblemGemm::TENSOR::SCALEB][idx]
                           * problem.tensors()[ContractionProblemGemm::TENSOR::SCALEB]
                                 .elementBytes();
                }
                if(u8Ptr[ContractionProblemGemm::TENSOR::SCALEC] != nullptr)
                {
                    u8Ptr[ContractionProblemGemm::TENSOR::SCALEC]
                        += offsets[ContractionProblemGemm::TENSOR::SCALEC][idx]
                           * problem.tensors()[ContractionProblemGemm::TENSOR::SCALEC]
                                 .elementBytes();
                }
                if(u8Ptr[ContractionProblemGemm::TENSOR::SCALED] != nullptr)
                {
                    u8Ptr[ContractionProblemGemm::TENSOR::SCALED]
                        += offsets[ContractionProblemGemm::TENSOR::SCALED][idx]
                           * problem.tensors()[ContractionProblemGemm::TENSOR::SCALED]
                                 .elementBytes();
                }
                if(u8Ptr[ContractionProblemGemm::TENSOR::SCALEALPHAVEC] != nullptr)
                {
                    u8Ptr[ContractionProblemGemm::TENSOR::SCALEALPHAVEC]
                        += offsets[ContractionProblemGemm::TENSOR::SCALEALPHAVEC][idx]
                           * problem.tensors()[ContractionProblemGemm::TENSOR::SCALEALPHAVEC]
                                 .elementBytes();
                }
                if(u8Ptr[ContractionProblemGemm::TENSOR::Synchronizer] != nullptr)
                {
                    u8Ptr[ContractionProblemGemm::TENSOR::Synchronizer]
                        += offsets[ContractionProblemGemm::TENSOR::Synchronizer][idx]
                           * problem.tensors()[ContractionProblemGemm::TENSOR::Synchronizer]
                                 .elementBytes();
                }
            }
        }

        // For GEMM only
        std::shared_ptr<ProblemInputs>
            DataInitialization::ConvertToProblemInputs(ContractionProblemGemm const& problem,
                                                       bool                          isGPU)
        {
            using std::static_pointer_cast;
            std::shared_ptr<ProblemInputs> result;
            if(m_groupedOffsets[0].empty())
            {
                auto inputs = new ContractionInputs();
                if(isGPU)
                    setContractionInputs(m_gpuPtrs,
                                         m_gpuBatchPtrs,
                                         m_workspacePristine.get(),
                                         m_cdata,
                                         m_maxElements,
                                         isGPU,
                                         inputs);
                else
                {
                    auto dummyBatchPtrs = std::vector<void**>(
                        ContractionProblemGemm::TENSOR::TENSOR_COUNT, nullptr);
                    setContractionInputs(m_cpuPtrs,
                                         dummyBatchPtrs,
                                         m_workspacePristine.get(),
                                         m_cdata,
                                         m_maxElements,
                                         isGPU,
                                         inputs);
                }
                result = static_pointer_cast<ProblemInputs>(
                    std::shared_ptr<ContractionInputs>(inputs));
            }
            else
            {
                auto inputs = new ContractionGroupedInputs();
                // Currently grouped gemm does not support batch, so we use a dummy batch vector here.
                auto dummyBatchPtrs
                    = std::vector<void**>(ContractionProblemGemm::TENSOR::TENSOR_COUNT, nullptr);
                if(isGPU)
                    setContractionGroupedInputs(m_gpuPtrs,
                                                dummyBatchPtrs,
                                                m_workspacePristine.get(),
                                                m_cdata,
                                                isGPU,
                                                problem,
                                                m_groupedOffsets,
                                                inputs);
                else
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
                    = min(maxRotatingBufferNum, ceil((float)m_rotatingBuffer / rotatingSize))
                      - 1; // Minus the original buffer.

                // <= 0 means don't rotating
                rotatingNum = max(0, rotatingNum);

                int32_t totalRotatingSizeNeeded = rotatingNum * rotatingSize;
                std::cout << "Rotating buffer set to: " << m_rotatingBuffer
                          << ". Rotating num: " << rotatingNum << std::endl;
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
                    = min(maxRotatingBufferNum, ceil((float)m_rotatingBuffer / rotatingSize))
                      - 1; // Minus the original buffer.

                // <= 0 means don't rotating
                rotatingNum = max(0, rotatingNum);

                int32_t totalRotatingSizeNeeded = rotatingNum * rotatingSize;
                std::cout << "Rotating buffer set to: " << m_rotatingBuffer
                          << ". Rotating num: " << rotatingNum << std::endl;
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

        DataInitialization::~DataInitialization() {}
    } // namespace Client
} // namespace TensileLite
