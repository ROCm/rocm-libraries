/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include <boost/program_options.hpp>

#include <Tensile/ContractionProblem.hpp>

#include "ClientProblemFactory.hpp"

#include <cstddef>
#include <random>

#include "RunListener.hpp"

namespace po = boost::program_options;

namespace Tensile
{
    namespace Client
    {
        // Problem-indept. from 0~7, and 16 (fixed values for every problem)
        // And problem-dept. from 8~15 (values depend on problem)
        enum class InitMode
        {
            Zero = 0, // 0
            One, // 1
            Two, // 2
            Random, // 3
            NaN, // 4
            Inf, // 5
            BadInput, // 6
            BadOutput, // 7
            SerialIdx, // 8
            SerialDim0, // 9
            SerialDim1, // 10
            Identity, // 11
            TrigSin, // 12
            TrigCos, // 13
            TrigAbsSin, // 14
            TrigAbsCos, // 15
            RandomNarrow, // 16
            NegOne, // 17
            Max, // 18
            DenormMin, // 19
            DenormMax, // 20
            Count
        };

        static bool IsProblemDependent(InitMode const& mode)
        {
            return mode == InitMode::SerialIdx || mode == InitMode::SerialDim0
                   || mode == InitMode::SerialDim1 || mode == InitMode::Identity
                   || mode == InitMode::TrigSin || mode == InitMode::TrigCos
                   || mode == InitMode::TrigAbsSin || mode == InitMode::TrigAbsCos;
        }

        std::string ToString(InitMode mode);

        std::ostream& operator<<(std::ostream& stream, InitMode const& mode);
        std::istream& operator>>(std::istream& stream, InitMode& mode);

        const int pageSize = 2 * 1024 * 1024;

        enum class BoundsCheckMode
        {
            Disable = 0,
            NaN,
            GuardPageFront,
            GuardPageBack,
            GuardPageAll,
            MaxMode
        };

        std::ostream& operator<<(std::ostream& stream, BoundsCheckMode const& mode);
        std::istream& operator>>(std::istream& stream, BoundsCheckMode& mode);

        template <typename TypedInputs>
        class TypedDataInitialization;

        class DataInitialization : public RunListener
        {
        public:
            static double GetRepresentativeBetaValue(po::variables_map const& args);

            /**
   * Factory function.
   */
            static std::shared_ptr<DataInitialization>
                Get(po::variables_map const&    args,
                    ClientProblemFactory const& problemFactory,
                    size_t                      maxWorkspaceSize = 0);

            template <typename TypedInputs>
            static std::shared_ptr<TypedDataInitialization<TypedInputs>>
                GetTyped(po::variables_map const&    args,
                         ClientProblemFactory const& problemFactory,
                         size_t                      maxWorkspaceSize = 0);

            DataInitialization(po::variables_map const&    args,
                               ClientProblemFactory const& problemFactory,
                               size_t                      maxWorkspaceSize = 0);
            ~DataInitialization();

            /**
   * Returns a ContractionInputs object with pointers to CPU memory,
   * suitable for using to calculate reference results.
   */
            virtual std::shared_ptr<ContractionInputs>
                prepareCPUInputs(ContractionProblem const& problem) = 0;

            /**
   * Returns a ContractionInputs object with pointers to GPU memory,
   * suitable for using to run the kernel.
   */
            virtual std::shared_ptr<ContractionInputs>
                prepareGPUInputs(ContractionProblem const& problem) = 0;

            virtual std::shared_ptr<ContractionInputs> cpuConvInputs() const = 0;

            template <typename T>
            static T getValue(InitMode mode)
            {
                switch(mode)
                {
                case InitMode::Zero:
                    return getValue<T, InitMode::Zero>();
                case InitMode::One:
                    return getValue<T, InitMode::One>();
                case InitMode::Two:
                    return getValue<T, InitMode::Two>();
                case InitMode::Random:
                    return getValue<T, InitMode::Random>();
                case InitMode::RandomNarrow:
                    return getValue<T, InitMode::RandomNarrow>();
                case InitMode::NaN:
                    return getValue<T, InitMode::NaN>();
                case InitMode::Inf:
                    return getValue<T, InitMode::Inf>();
                case InitMode::BadInput:
                    return getValue<T, InitMode::BadInput>();
                case InitMode::BadOutput:
                    return getValue<T, InitMode::BadOutput>();
                case InitMode::NegOne:
                    return getValue<T, InitMode::NegOne>();
                case InitMode::Max:
                    return getValue<T, InitMode::Max>();
                case InitMode::DenormMin:
                    return getValue<T, InitMode::DenormMin>();
                case InitMode::DenormMax:
                    return getValue<T, InitMode::DenormMax>();
                case InitMode::SerialIdx:
                case InitMode::SerialDim0:
                case InitMode::SerialDim1:
                case InitMode::Identity:
                case InitMode::TrigSin:
                case InitMode::TrigCos:
                case InitMode::TrigAbsSin:
                case InitMode::TrigAbsCos:
                case InitMode::Count:
                    throw std::runtime_error("Invalid InitMode.");
                }
            }

            template <typename T, InitMode Mode>
            static inline T getValue();

            template <typename T>
            static inline T getTrigValue(int idx, bool useCos, bool useAbs);

            template <typename T>
            static bool isBadInput(T value);

            template <typename T>
            static bool isBadOutput(T value);

            // Fills max buffer size
            template <typename T>
            void initArray(InitMode mode, T* array, size_t elements)
            {
                switch(mode)
                {
                case InitMode::Zero:
                    initArray<T, InitMode::Zero>(array, elements);
                    break;
                case InitMode::One:
                    initArray<T, InitMode::One>(array, elements);
                    break;
                case InitMode::Two:
                    initArray<T, InitMode::Two>(array, elements);
                    break;
                case InitMode::Random:
                    initArray<T, InitMode::Random>(array, elements);
                    break;
                case InitMode::RandomNarrow:
                    initArray<T, InitMode::RandomNarrow>(array, elements);
                    break;
                case InitMode::NaN:
                    initArray<T, InitMode::NaN>(array, elements);
                    break;
                case InitMode::Inf:
                    initArray<T, InitMode::Inf>(array, elements);
                    break;
                case InitMode::BadInput:
                    initArray<T, InitMode::BadInput>(array, elements);
                    break;
                case InitMode::BadOutput:
                    initArray<T, InitMode::BadOutput>(array, elements);
                    break;
                case InitMode::NegOne:
                    initArray<T, InitMode::NegOne>(array, elements);
                    break;
                case InitMode::Max:
                    initArray<T, InitMode::Max>(array, elements);
                    break;
                case InitMode::DenormMin:
                    initArray<T, InitMode::DenormMin>(array, elements);
                    break;
                case InitMode::DenormMax:
                    initArray<T, InitMode::DenormMax>(array, elements);
                    break;
                case InitMode::SerialIdx:
                case InitMode::SerialDim0:
                case InitMode::SerialDim1:
                case InitMode::Identity:
                case InitMode::TrigSin:
                case InitMode::TrigCos:
                case InitMode::TrigAbsSin:
                case InitMode::TrigAbsCos:
                case InitMode::Count:
                    throw std::runtime_error("Invalid InitMode.");
                }
            }

            // For problem dependent data initialization
            template <typename T>
            void initArray(InitMode mode, T* array, TensorDescriptor const& tensor)
            {
                switch(mode)
                {
                case InitMode::Zero:
                    initArray<T, InitMode::Zero>(array, tensor);
                    break;
                case InitMode::One:
                    initArray<T, InitMode::One>(array, tensor);
                    break;
                case InitMode::Two:
                    initArray<T, InitMode::Two>(array, tensor);
                    break;
                case InitMode::Random:
                    initArray<T, InitMode::Random>(array, tensor);
                    break;
                case InitMode::RandomNarrow:
                    initArray<T, InitMode::RandomNarrow>(array, tensor);
                    break;
                case InitMode::NaN:
                    initArray<T, InitMode::NaN>(array, tensor);
                    break;
                case InitMode::Inf:
                    initArray<T, InitMode::Inf>(array, tensor);
                    break;
                case InitMode::BadInput:
                    initArray<T, InitMode::BadInput>(array, tensor);
                    break;
                case InitMode::BadOutput:
                    initArray<T, InitMode::BadOutput>(array, tensor);
                    break;
                case InitMode::NegOne:
                    initArray<T, InitMode::NegOne>(array, tensor);
                    break;
                case InitMode::Max:
                    initArray<T, InitMode::Max>(array, tensor);
                    break;
                case InitMode::DenormMin:
                    initArray<T, InitMode::DenormMin>(array, tensor);
                    break;
                case InitMode::DenormMax:
                    initArray<T, InitMode::DenormMax>(array, tensor);
                    break;
                case InitMode::SerialIdx:
                    initArraySerialIdx<T>(array, tensor);
                    break;
                case InitMode::SerialDim0:
                    initArraySerialDim<T>(array, 0, tensor);
                    break;
                case InitMode::SerialDim1:
                    initArraySerialDim<T>(array, 1, tensor);
                    break;
                case InitMode::Identity:
                    initArrayIdentity<T>(array, tensor);
                    break;
                case InitMode::TrigSin:
                    initArrayTrig<T, false, false>(array, tensor);
                    break;
                case InitMode::TrigCos:
                    initArrayTrig<T, true, false>(array, tensor);
                    break;
                case InitMode::TrigAbsSin:
                    initArrayTrig<T, false, true>(array, tensor);
                    break;
                case InitMode::TrigAbsCos:
                    initArrayTrig<T, true, true>(array, tensor);
                    break;
                case InitMode::Count:
                    throw std::runtime_error("Invalid InitMode.");
                }
            }

            template <typename T, InitMode Mode>
            void initArray(T* array, size_t elements)
            {
                for(size_t i = 0; i < elements; i++)
                {
                    array[i] = getValue<T, Mode>();
                }
            }

            template <typename T, InitMode Mode>
            void initArray(T* array, TensorDescriptor const& tensor)
            {
                size_t elements = tensor.totalAllocatedElements();
                initArray<T, Mode>(array, elements);
            }

            template <typename T>
            void initArraySerialIdx(T* array, TensorDescriptor const& tensor)
            {
                auto const&         sizes = tensor.sizes();
                auto                count = CoordCount(sizes.begin(), sizes.end());
                std::vector<size_t> coord(tensor.dimensions(), 0);
                for(size_t idx = 0; idx < count; idx++)
                {
                    CoordNumbered(idx, coord.begin(), coord.end(), sizes.begin(), sizes.end());
                    array[tensor.index(coord)] = static_cast<T>(idx);
                }
            }

            template <typename T>
            void initArraySerialDim(T* array, int dim, TensorDescriptor const& tensor)
            {
                auto const&         sizes = tensor.sizes();
                auto                count = CoordCount(sizes.begin(), sizes.end());
                std::vector<size_t> coord(tensor.dimensions(), 0);
                for(size_t idx = 0; idx < count; idx++)
                {
                    CoordNumbered(idx, coord.begin(), coord.end(), sizes.begin(), sizes.end());
                    array[tensor.index(coord)] = static_cast<T>(coord[dim]);
                }
            }

            template <typename T>
            void initArrayIdentity(T* array, TensorDescriptor const& tensor)
            {
                auto const&         sizes = tensor.sizes();
                auto                count = CoordCount(sizes.begin(), sizes.end());
                std::vector<size_t> coord(tensor.dimensions(), 0);
                for(size_t idx = 0; idx < count; idx++)
                {
                    CoordNumbered(idx, coord.begin(), coord.end(), sizes.begin(), sizes.end());
                    array[tensor.index(coord)] = static_cast<T>(coord[0] == coord[1] ? 1 : 0);
                }
            }

            template <typename T, bool useCos, bool useAbs>
            void initArrayTrig(T* array, TensorDescriptor const& tensor)
            {
                auto const&         sizes = tensor.sizes();
                auto                count = CoordCount(sizes.begin(), sizes.end());
                std::vector<size_t> coord(tensor.dimensions(), 0);
                for(size_t idx = 0; idx < count; idx++)
                {
                    CoordNumbered(idx, coord.begin(), coord.end(), sizes.begin(), sizes.end());
                    array[tensor.index(coord)] = getTrigValue<T>(idx, useCos, useAbs);
                }
            }

            size_t workspaceSize() const
            {
                return m_workspaceSize;
            }

            BoundsCheckMode getCurBoundsCheck()
            {
                return m_curBoundsCheck;
            }

            virtual bool needMoreBenchmarkRuns() const override
            {
                return false;
            };
            virtual void preBenchmarkRun() override{};
            virtual void postBenchmarkRun() override{};
            virtual void preProblem(ContractionProblem const& problem) override{};
            virtual void postProblem() override{};
            virtual void preSolution(ContractionSolution const& solution) override{};
            virtual void postSolution() override{};
            virtual bool needMoreRunsInSolution() const override
            {
                return m_numRunsInSolution < m_numRunsPerSolution;
            };

            virtual size_t numWarmupRuns() override
            {
                return 0;
            };
            virtual void setNumWarmupRuns(size_t count) override{};
            virtual void preWarmup() override{};
            virtual void postWarmup() override{};
            virtual void validateWarmups(std::shared_ptr<ContractionInputs> inputs,
                                         TimingEvents const&                startEvents,
                                         TimingEvents const&                stopEvents) override
            {
                m_numRunsInSolution++;
            };

            virtual size_t numSyncs() override
            {
                return 0;
            };
            virtual void setNumSyncs(size_t count) override{};
            virtual void preSyncs() override{};
            virtual void postSyncs() override{};

            virtual size_t numEnqueuesPerSync() override
            {
                return 0;
            };
            virtual void setNumEnqueuesPerSync(size_t count) override{};
            virtual void preEnqueues() override{};
            virtual void postEnqueues(TimingEvents const& startEvents,
                                      TimingEvents const& stopEvents) override{};
            virtual void validateEnqueues(std::shared_ptr<ContractionInputs> inputs,
                                          TimingEvents const&                startEvents,
                                          TimingEvents const&                stopEvents) override{};

            virtual void finalizeReport() override{};

            virtual int error() const override
            {
                return 0;
            };

        protected:
            InitMode m_aInit, m_bInit, m_cInit, m_dInit;
            InitMode m_alphaInit, m_betaInit;

            size_t m_aMaxElements;
            size_t m_bMaxElements;
            size_t m_cMaxElements;
            size_t m_dMaxElements;
            size_t m_maxBatch;

            size_t m_workspaceSize;

            bool m_stridedBatched;

            bool m_cEqualsD;
            bool m_convolutionVsContraction;

            int m_elementsToValidate = 0;

            /// If true, we will allocate an extra copy of the inputs on the GPU.
            /// This will improve performance as we don't have to copy from the CPU
            /// with each kernel launch, but it will use extra memory.
            bool m_keepPristineCopyOnGPU = true;

            /// If set "::NaN", we will initialize all out-of-bounds inputs to NaN, and
            /// all out-of-bounds outputs to a known value. This allows us to
            /// verify that out-of-bounds values are not used or written to.
            /// If set "::GuardPageFront/::GuardPageBack", we will allocate matrix memory
            /// with page aligned, and put matrix start/end address to memory start/end address.
            /// Out-of-bounds access would trigger memory segmentation faults.
            /// m_boundsCheck keep the setting from args.
            /// m_curBoundsCheck keep the current running boundsCheck mode.
            /// If set "::GuardPageAll", DataInit would need 2 runs per solution.
            /// First run would apply "::GuardPageFront" and second run would apply "::GuardPageBack".
            BoundsCheckMode m_boundsCheck        = BoundsCheckMode::Disable;
            BoundsCheckMode m_curBoundsCheck     = BoundsCheckMode::Disable;
            int             m_numRunsPerSolution = 0;
            int             m_numRunsInSolution  = 0;

            /// If true, the data is dependent on the problem size (e.g. serial)
            /// and must be reinitialized for each problem. Pristine copy on GPU
            /// cannot be used with problem dependent data.
            bool m_problemDependentData = false;
        };

        template <>
        inline float DataInitialization::getValue<float, InitMode::Zero>()
        {
            return 0.0f;
        }
        template <>
        inline float DataInitialization::getValue<float, InitMode::One>()
        {
            return 1.0f;
        }
        template <>
        inline float DataInitialization::getValue<float, InitMode::Two>()
        {
            return 2.0f;
        }
        template <>
        inline float DataInitialization::getValue<float, InitMode::NegOne>()
        {
            return -1.0f;
        }
        template <>
        inline float DataInitialization::getValue<float, InitMode::Max>()
        {
            return std::numeric_limits<float>::max();
        }
        template <>
        inline float DataInitialization::getValue<float, InitMode::DenormMin>()
        {
            return std::numeric_limits<float>::denorm_min();
        }
        template <>
        inline float DataInitialization::getValue<float, InitMode::DenormMax>()
        {
            union
            {
                uint32_t bits;
                float    value;
            } x;

            x.bits = 0x007FFFFF;
            return x.value;
        }
        template <>
        inline float DataInitialization::getValue<float, InitMode::NaN>()
        {
            return std::numeric_limits<float>::quiet_NaN();
        }
        template <>
        inline float DataInitialization::getValue<float, InitMode::Inf>()
        {
            return std::numeric_limits<float>::infinity();
        }

        template <>
        inline float DataInitialization::getValue<float, InitMode::Random>()
        {
            return static_cast<float>((rand() % 201) - 100);
        }

        template <>
        inline float DataInitialization::getValue<float, InitMode::BadInput>()
        {
            return getValue<float, InitMode::NaN>();
        }
        template <>
        inline float DataInitialization::getValue<float, InitMode::BadOutput>()
        {
            return getValue<float, InitMode::Inf>();
        }

        template <>
        inline double DataInitialization::getValue<double, InitMode::Zero>()
        {
            return 0.0;
        }
        template <>
        inline double DataInitialization::getValue<double, InitMode::One>()
        {
            return 1.0;
        }
        template <>
        inline double DataInitialization::getValue<double, InitMode::Two>()
        {
            return 2.0;
        }
        template <>
        inline double DataInitialization::getValue<double, InitMode::NegOne>()
        {
            return -1.0;
        }
        template <>
        inline double DataInitialization::getValue<double, InitMode::Max>()
        {
            return std::numeric_limits<double>::max();
        }
        template <>
        inline double DataInitialization::getValue<double, InitMode::DenormMin>()
        {
            return std::numeric_limits<double>::denorm_min();
        }
        template <>
        inline double DataInitialization::getValue<double, InitMode::DenormMax>()
        {
            union
            {
                uint64_t bits;
                double   value;
            } x;

            x.bits = 0x000FFFFFFFFFFFFF;
            return x.value;
        }
        template <>
        inline double DataInitialization::getValue<double, InitMode::NaN>()
        {
            return std::numeric_limits<double>::quiet_NaN();
        }
        template <>
        inline double DataInitialization::getValue<double, InitMode::Inf>()
        {
            return std::numeric_limits<double>::infinity();
        }

        template <>
        inline double DataInitialization::getValue<double, InitMode::Random>()
        {
            return static_cast<double>((rand() % 2001) - 1000);
        }

        template <>
        inline double DataInitialization::getValue<double, InitMode::BadInput>()
        {
            return getValue<double, InitMode::NaN>();
        }
        template <>
        inline double DataInitialization::getValue<double, InitMode::BadOutput>()
        {
            return getValue<double, InitMode::Inf>();
        }

        template <>
        inline std::complex<float>
            DataInitialization::getValue<std::complex<float>, InitMode::Zero>()
        {
            return std::complex<float>(0.0f, 0.0f);
        }
        template <>
        inline std::complex<float>
            DataInitialization::getValue<std::complex<float>, InitMode::One>()
        {
            return std::complex<float>(1.0f, 0.0f);
        }
        template <>
        inline std::complex<float>
            DataInitialization::getValue<std::complex<float>, InitMode::Two>()
        {
            return std::complex<float>(2.0f, 0.0f);
        }
        template <>
        inline std::complex<float>
            DataInitialization::getValue<std::complex<float>, InitMode::NegOne>()
        {
            return std::complex<float>(-1.0f, 0.0f);
        }

        template <>
        inline std::complex<float>
            DataInitialization::getValue<std::complex<float>, InitMode::Max>()
        {
            return std::complex<float>(std::numeric_limits<float>::max(),
                                       std::numeric_limits<float>::max());
        }

        template <>
        inline std::complex<float>
            DataInitialization::getValue<std::complex<float>, InitMode::DenormMin>()
        {
            return std::complex<float>(std::numeric_limits<float>::denorm_min(),
                                       std::numeric_limits<float>::denorm_min());
        }

        template <>
        inline std::complex<float>
            DataInitialization::getValue<std::complex<float>, InitMode::DenormMax>()
        {
            union
            {
                uint32_t bits;
                float    value;
            } x;

            x.bits = 0x007FFFFF;
            return std::complex<float>(x.value, x.value);
        }

        template <>
        inline std::complex<float>
            DataInitialization::getValue<std::complex<float>, InitMode::NaN>()
        {
            return std::complex<float>(std::numeric_limits<float>::quiet_NaN(),
                                       std::numeric_limits<float>::quiet_NaN());
        }

        template <>
        inline std::complex<float>
            DataInitialization::getValue<std::complex<float>, InitMode::Inf>()
        {
            return std::complex<float>(std::numeric_limits<float>::infinity(),
                                       std::numeric_limits<float>::infinity());
        }

        template <>
        inline std::complex<float>
            DataInitialization::getValue<std::complex<float>, InitMode::Random>()
        {
            return std::complex<float>(getValue<float, InitMode::Random>(),
                                       getValue<float, InitMode::Random>());
        }

        template <>
        inline std::complex<float>
            DataInitialization::getValue<std::complex<float>, InitMode::BadInput>()
        {
            return getValue<std::complex<float>, InitMode::NaN>();
        }
        template <>
        inline std::complex<float>
            DataInitialization::getValue<std::complex<float>, InitMode::BadOutput>()
        {
            return getValue<std::complex<float>, InitMode::Inf>();
        }

        template <>
        inline std::complex<double>
            DataInitialization::getValue<std::complex<double>, InitMode::Zero>()
        {
            return std::complex<double>(0.0, 0.0);
        }
        template <>
        inline std::complex<double>
            DataInitialization::getValue<std::complex<double>, InitMode::One>()
        {
            return std::complex<double>(1.0, 0.0);
        }
        template <>
        inline std::complex<double>
            DataInitialization::getValue<std::complex<double>, InitMode::Two>()
        {
            return std::complex<double>(2.0, 0.0);
        }
        template <>
        inline std::complex<double>
            DataInitialization::getValue<std::complex<double>, InitMode::NegOne>()
        {
            return std::complex<double>(-1.0, 0.0);
        }
        template <>
        inline std::complex<double>
            DataInitialization::getValue<std::complex<double>, InitMode::Max>()
        {
            return std::complex<double>(std::numeric_limits<double>::max(),
                                        std::numeric_limits<double>::max());
        }
        template <>
        inline std::complex<double>
            DataInitialization::getValue<std::complex<double>, InitMode::DenormMin>()
        {
            return std::complex<double>(std::numeric_limits<double>::denorm_min(),
                                        std::numeric_limits<double>::denorm_min());
        }
        template <>
        inline std::complex<double>
            DataInitialization::getValue<std::complex<double>, InitMode::DenormMax>()
        {
            union
            {
                uint64_t bits;
                double   value;
            } x;

            x.bits = 0x000FFFFFFFFFFFFF;
            return std::complex<double>(x.value, x.value);
        }
        template <>
        inline std::complex<double>
            DataInitialization::getValue<std::complex<double>, InitMode::NaN>()
        {
            return std::complex<double>(std::numeric_limits<double>::quiet_NaN(),
                                        std::numeric_limits<double>::quiet_NaN());
        }
        template <>
        inline std::complex<double>
            DataInitialization::getValue<std::complex<double>, InitMode::Inf>()
        {
            return std::complex<double>(std::numeric_limits<double>::infinity(),
                                        std::numeric_limits<double>::infinity());
        }

        template <>
        inline std::complex<double>
            DataInitialization::getValue<std::complex<double>, InitMode::Random>()
        {
            return std::complex<double>(getValue<double, InitMode::Random>(),
                                        getValue<double, InitMode::Random>());
        }

        template <>
        inline std::complex<double>
            DataInitialization::getValue<std::complex<double>, InitMode::BadInput>()
        {
            return getValue<std::complex<double>, InitMode::NaN>();
        }
        template <>
        inline std::complex<double>
            DataInitialization::getValue<std::complex<double>, InitMode::BadOutput>()
        {
            return getValue<std::complex<double>, InitMode::Inf>();
        }

        template <>
        inline int32_t DataInitialization::getValue<int32_t, InitMode::Zero>()
        {
            return 0;
        }
        template <>
        inline int32_t DataInitialization::getValue<int32_t, InitMode::One>()
        {
            return 1;
        }
        template <>
        inline int32_t DataInitialization::getValue<int32_t, InitMode::Two>()
        {
            return 2;
        }
        template <>
        inline int32_t DataInitialization::getValue<int32_t, InitMode::NegOne>()
        {
            return -1;
        }
        template <>
        inline int32_t DataInitialization::getValue<int32_t, InitMode::Max>()
        {
            return std::numeric_limits<int32_t>::max();
        }
        template <>
        inline int32_t DataInitialization::getValue<int32_t, InitMode::DenormMin>()
        {
            throw std::runtime_error("DenormMin not available for int32_t.");
        }
        template <>
        inline int32_t DataInitialization::getValue<int32_t, InitMode::DenormMax>()
        {
            throw std::runtime_error("DenormMax not available for int32_t.");
        }
        template <>
        inline int32_t DataInitialization::getValue<int32_t, InitMode::NaN>()
        {
            throw std::runtime_error("NaN not available for int32_t.");
        }
        template <>
        inline int32_t DataInitialization::getValue<int32_t, InitMode::Inf>()
        {
            throw std::runtime_error("Inf not available for int32_t.");
        }

        template <>
        inline int32_t DataInitialization::getValue<int32_t, InitMode::Random>()
        {
            return rand() % 7 - 3;
        }

        template <>
        inline int32_t DataInitialization::getValue<int32_t, InitMode::BadInput>()
        {
            return std::numeric_limits<int32_t>::max();
        }
        template <>
        inline int32_t DataInitialization::getValue<int32_t, InitMode::BadOutput>()
        {
            return std::numeric_limits<int32_t>::min();
        }

        template <>
        inline Int8x4 DataInitialization::getValue<Int8x4, InitMode::Zero>()
        {
            return Int8x4{0, 0, 0, 0};
        }
        template <>
        inline Int8x4 DataInitialization::getValue<Int8x4, InitMode::One>()
        {
            return Int8x4{1, 1, 1, 1};
        }
        template <>
        inline Int8x4 DataInitialization::getValue<Int8x4, InitMode::Two>()
        {
            return Int8x4{2, 2, 2, 2};
        }
        template <>
        inline Int8x4 DataInitialization::getValue<Int8x4, InitMode::NegOne>()
        {
            return Int8x4{-1, -1, -1, -1};
        }
        template <>
        inline Int8x4 DataInitialization::getValue<Int8x4, InitMode::Max>()
        {
            return Int8x4{127, 127, 127, 127};
        }
        template <>
        inline Int8x4 DataInitialization::getValue<Int8x4, InitMode::DenormMin>()
        {
            throw std::runtime_error("DenormMin not available for Int8x4.");
        }
        template <>
        inline Int8x4 DataInitialization::getValue<Int8x4, InitMode::DenormMax>()
        {
            throw std::runtime_error("DenormMax not available for Int8x4.");
        }
        template <>
        inline Int8x4 DataInitialization::getValue<Int8x4, InitMode::NaN>()
        {
            throw std::runtime_error("NaN not available for Int8x4.");
        }
        template <>
        inline Int8x4 DataInitialization::getValue<Int8x4, InitMode::Inf>()
        {
            throw std::runtime_error("Inf not available for Int8x4.");
        }

        template <>
        inline Int8x4 DataInitialization::getValue<Int8x4, InitMode::Random>()
        {
            return Int8x4{static_cast<int8_t>((rand() % 7) - 3),
                          static_cast<int8_t>((rand() % 7) - 3),
                          static_cast<int8_t>((rand() % 7) - 3),
                          static_cast<int8_t>((rand() % 7) - 3)};
        }

        template <>
        inline Int8x4 DataInitialization::getValue<Int8x4, InitMode::BadInput>()
        {
            auto val = std::numeric_limits<int8_t>::max();
            return Int8x4{val, val, val, val};
        }

        template <>
        inline Int8x4 DataInitialization::getValue<Int8x4, InitMode::BadOutput>()
        {
            auto val = std::numeric_limits<int8_t>::min();
            return Int8x4{val, val, val, val};
        }

        template <>
        inline Half DataInitialization::getValue<Half, InitMode::Zero>()
        {
            return static_cast<Half>(0);
        }
        template <>
        inline Half DataInitialization::getValue<Half, InitMode::One>()
        {
            return static_cast<Half>(1);
        }
        template <>
        inline Half DataInitialization::getValue<Half, InitMode::Two>()
        {
            return static_cast<Half>(2);
        }
        template <>
        inline Half DataInitialization::getValue<Half, InitMode::NegOne>()
        {
            return static_cast<Half>(-1);
        }
        template <>
        inline Half DataInitialization::getValue<Half, InitMode::Max>()
        {
            return static_cast<Half>(65504.0);
        }
        template <>
        inline Half DataInitialization::getValue<Half, InitMode::DenormMin>()
        {
            union
            {
                uint16_t bits;
                Half     value;
            } x;

            x.bits = 0x0001;
            return x.value;
        }
        template <>
        inline Half DataInitialization::getValue<Half, InitMode::DenormMax>()
        {
            union
            {
                uint16_t bits;
                Half     value;
            } x;

            x.bits = 0x03FF;
            return x.value;
        }
        template <>
        inline Half DataInitialization::getValue<Half, InitMode::NaN>()
        {
            union
            {
                uint16_t bits;
                Half     value;
            } x;

            x.bits = 0xFFFF;
            return x.value;
        }
        template <>
        inline Half DataInitialization::getValue<Half, InitMode::Inf>()
        {
            union
            {
                uint16_t bits;
                Half     value;
            } x;

            x.bits = 0x7C00;
            return x.value;
        }

        template <>
        inline Half DataInitialization::getValue<Half, InitMode::Random>()
        {
            return static_cast<Half>((rand() % 7) - 3);
        }

        template <>
        inline Half DataInitialization::getValue<Half, InitMode::BadInput>()
        {
            return getValue<Half, InitMode::NaN>();
        }
        template <>
        inline Half DataInitialization::getValue<Half, InitMode::BadOutput>()
        {
            return getValue<Half, InitMode::Inf>();
        }

        template <>
        inline BFloat16 DataInitialization::getValue<BFloat16, InitMode::Zero>()
        {
            return static_cast<BFloat16>(0);
        }
        template <>
        inline BFloat16 DataInitialization::getValue<BFloat16, InitMode::One>()
        {
            return static_cast<BFloat16>(1);
        }
        template <>
        inline BFloat16 DataInitialization::getValue<BFloat16, InitMode::Two>()
        {
            return static_cast<BFloat16>(2);
        }
        template <>
        inline BFloat16 DataInitialization::getValue<BFloat16, InitMode::NegOne>()
        {
            return static_cast<BFloat16>(-1);
        }
        template <>
        inline BFloat16 DataInitialization::getValue<BFloat16, InitMode::Max>()
        {
            return static_cast<BFloat16>(DataInitialization::getValue<float, InitMode::Max>());
        }
        template <>
        inline BFloat16 DataInitialization::getValue<BFloat16, InitMode::DenormMin>()
        {
            BFloat16 bf16;
            bf16.data = 0x0001;
            return bf16;
        }
        template <>
        inline BFloat16 DataInitialization::getValue<BFloat16, InitMode::DenormMax>()
        {
            BFloat16 bf16;
            bf16.data = 0x007F;
            return bf16;
        }
        template <>
        inline BFloat16 DataInitialization::getValue<BFloat16, InitMode::NaN>()
        {
            return static_cast<BFloat16>(std::numeric_limits<float>::quiet_NaN());
        }
        template <>
        inline BFloat16 DataInitialization::getValue<BFloat16, InitMode::Inf>()
        {
            return static_cast<BFloat16>(std::numeric_limits<float>::infinity());
        }

        template <>
        inline BFloat16 DataInitialization::getValue<BFloat16, InitMode::Random>()
        {
            return static_cast<BFloat16>((rand() % 7) - 3);
        }

        template <>
        inline BFloat16 DataInitialization::getValue<BFloat16, InitMode::BadInput>()
        {
            return getValue<BFloat16, InitMode::NaN>();
        }
        template <>
        inline BFloat16 DataInitialization::getValue<BFloat16, InitMode::BadOutput>()
        {
            return getValue<BFloat16, InitMode::Inf>();
        }

        // Float8
        // NOTE: f8/bf8 has special overloading with unit8_t, so using float val and downcast it to f8/bf8 here
        template <>
        inline Float8 DataInitialization::getValue<Float8, InitMode::Zero>()
        {
            return static_cast<Float8>(0.0f);
        }
        template <>
        inline Float8 DataInitialization::getValue<Float8, InitMode::One>()
        {
            return static_cast<Float8>(1.0f);
        }
        template <>
        inline Float8 DataInitialization::getValue<Float8, InitMode::Two>()
        {
            return static_cast<Float8>(2.0f);
        }
        template <>
        inline Float8 DataInitialization::getValue<Float8, InitMode::NegOne>()
        {
            return static_cast<Float8>(-1.0f);
        }
        template <>
        inline Float8 DataInitialization::getValue<Float8, InitMode::Max>()
        {
            // NOTE: sign=0 exp=1111, mantissa=111.. bit pattern : 0x7F
            //return static_cast<Float8>(240.0f);
            union
            {
                uint8_t bits;
                Float8  value;
            } x;

            x.bits = 0x7F;
            return x.value;
        }
        template <>
        inline Float8 DataInitialization::getValue<Float8, InitMode::DenormMin>()
        {
            // NOTE: sign=0 exp=0000, mantissa=001.. bit pattern : 0x01
            //return static_cast<Float8>(0.0009765625f);
            union
            {
                uint8_t bits;
                Float8  value;
            } x;

            x.bits = 0x01;
            return x.value;
        }
        template <>
        inline Float8 DataInitialization::getValue<Float8, InitMode::DenormMax>()
        {
            // NOTE: sign=0 exp=0000, mantissa=111.. bit pattern : 0x07
            union
            {
                uint8_t bits;
                Float8  value;
            } x;

            x.bits = 0x07;
            return x.value;
        }
        template <>
        inline Float8 DataInitialization::getValue<Float8, InitMode::NaN>()
        {
            if(!get_hip_f8_bias_mode())
                return static_cast<Float8>(std::numeric_limits<float>::quiet_NaN());
            else // optimal mode is unlike IEEE, nan and inf is 0x80 (-0)
            {
                union
                {
                    uint8_t bits;
                    Float8  value;
                } x;

                x.bits = 0x80;
                return x.value;
            }
        }
        template <>
        inline Float8 DataInitialization::getValue<Float8, InitMode::Inf>()
        {
            if(!get_hip_f8_bias_mode())
                return static_cast<Float8>(std::numeric_limits<float>::infinity());
            else // optimal mode is unlike IEEE, nan and inf is 0x80 (-0)
            {
                union
                {
                    uint8_t bits;
                    Float8  value;
                } x;

                x.bits = 0x80;
                return x.value;
            }
        }

        template <>
        inline Float8 DataInitialization::getValue<Float8, InitMode::Random>()
        {
            return static_cast<Float8>((float)((rand() % 7) - 3));
        }

        template <>
        inline Float8 DataInitialization::getValue<Float8, InitMode::BadInput>()
        {
            return getValue<Float8, InitMode::NaN>();
        }
        template <>
        inline Float8 DataInitialization::getValue<Float8, InitMode::BadOutput>()
        {
            return getValue<Float8, InitMode::Inf>();
        }

        // BFloat8
        template <>
        inline BFloat8 DataInitialization::getValue<BFloat8, InitMode::Zero>()
        {
            return static_cast<BFloat8>(0.0f);
        }
        template <>
        inline BFloat8 DataInitialization::getValue<BFloat8, InitMode::One>()
        {
            return static_cast<BFloat8>(1.0f);
        }
        template <>
        inline BFloat8 DataInitialization::getValue<BFloat8, InitMode::Two>()
        {
            return static_cast<BFloat8>(2.0f);
        }
        template <>
        inline BFloat8 DataInitialization::getValue<BFloat8, InitMode::NegOne>()
        {
            return static_cast<BFloat8>(-1.0f);
        }
        template <>
        inline BFloat8 DataInitialization::getValue<BFloat8, InitMode::NaN>()
        {
            if(!get_hip_f8_bias_mode())
                return static_cast<BFloat8>(std::numeric_limits<float>::quiet_NaN());
            else // optimal mode is unlike IEEE, nan and inf is 0x80 (-0)
            {
                union
                {
                    uint8_t bits;
                    BFloat8 value;
                } x;

                x.bits = 0x80;
                return x.value;
            }
        }
        template <>
        inline BFloat8 DataInitialization::getValue<BFloat8, InitMode::Inf>()
        {
            if(!get_hip_f8_bias_mode())
                return static_cast<BFloat8>(std::numeric_limits<float>::infinity());
            else // optimal mode is unlike IEEE, nan and inf is 0x80 (-0)
            {
                union
                {
                    uint8_t bits;
                    BFloat8 value;
                } x;

                x.bits = 0x80;
                return x.value;
            }
        }

        template <>
        inline BFloat8 DataInitialization::getValue<BFloat8, InitMode::Random>()
        {
            return static_cast<BFloat8>((float)((rand() % 7) - 3));
        }

        template <>
        inline BFloat8 DataInitialization::getValue<BFloat8, InitMode::BadInput>()
        {
            return getValue<BFloat8, InitMode::NaN>();
        }
        template <>
        inline BFloat8 DataInitialization::getValue<BFloat8, InitMode::BadOutput>()
        {
            return getValue<BFloat8, InitMode::Inf>();
        }
        template <>
        inline BFloat8 DataInitialization::getValue<BFloat8, InitMode::Max>()
        {
            // NOTE: sign=0 exp=11111, mantissa=11.. bit pattern : 0x7F
            //return static_cast<BFloat8>(57344.0f);
            //return static_cast<BFloat8>((uint8_t)0x7F);
            union
            {
                uint8_t bits;
                BFloat8 value;
            } x;

            x.bits = 0x7F;
            return x.value;
        }
        template <>
        inline BFloat8 DataInitialization::getValue<BFloat8, InitMode::DenormMin>()
        {
            // NOTE: sign=0 exp=00000, mantissa=01.. bit pattern : 0x01
            //return static_cast<BFloat8>(0.0000076294f);
            union
            {
                uint8_t bits;
                BFloat8 value;
            } x;

            x.bits = 0x01;
            return x.value;
        }
        template <>
        inline BFloat8 DataInitialization::getValue<BFloat8, InitMode::DenormMax>()
        {
            // NOTE: sign=0 exp=00000, mantissa=11.. bit pattern : 0x03
            union
            {
                uint8_t bits;
                BFloat8 value;
            } x;

            x.bits = 0x03;
            return x.value;
        }

        template <>
        inline int8_t DataInitialization::getValue<int8_t, InitMode::Zero>()
        {
            return 0;
        }
        template <>
        inline int8_t DataInitialization::getValue<int8_t, InitMode::One>()
        {
            return 1;
        }
        template <>
        inline int8_t DataInitialization::getValue<int8_t, InitMode::Two>()
        {
            return 2;
        }
        template <>
        inline int8_t DataInitialization::getValue<int8_t, InitMode::NegOne>()
        {
            return -1;
        }
        template <>
        inline int8_t DataInitialization::getValue<int8_t, InitMode::Max>()
        {
            return 127;
        }
        template <>
        inline int8_t DataInitialization::getValue<int8_t, InitMode::DenormMin>()
        {
            throw std::runtime_error("DenormMin not available for int8_t.");
        }
        template <>
        inline int8_t DataInitialization::getValue<int8_t, InitMode::DenormMax>()
        {
            throw std::runtime_error("DenormMax not available for int8_t.");
        }
        template <>
        inline int8_t DataInitialization::getValue<int8_t, InitMode::NaN>()
        {
            throw std::runtime_error("NaN not available for int8_t.");
        }
        template <>
        inline int8_t DataInitialization::getValue<int8_t, InitMode::Inf>()
        {
            throw std::runtime_error("Inf not available for int8_t.");
        }

        template <>
        inline int8_t DataInitialization::getValue<int8_t, InitMode::Random>()
        {
            return static_cast<int8_t>((rand() % 7) - 3);
        }

        template <>
        inline int8_t DataInitialization::getValue<int8_t, InitMode::BadInput>()
        {
            return std::numeric_limits<int8_t>::max();
        }
        template <>
        inline int8_t DataInitialization::getValue<int8_t, InitMode::BadOutput>()
        {
            return std::numeric_limits<int8_t>::min();
        }

        template <>
        inline bool DataInitialization::isBadInput<float>(float value)
        {
            return std::isnan(value);
        }

        template <>
        inline bool DataInitialization::isBadInput<double>(double value)
        {
            return std::isnan(value);
        }

        template <>
        inline bool DataInitialization::isBadInput<std::complex<float>>(std::complex<float> value)
        {
            return std::isnan(value.real()) && std::isnan(value.imag());
        }

        template <>
        inline bool DataInitialization::isBadInput<std::complex<double>>(std::complex<double> value)
        {
            return std::isnan(value.real()) && std::isnan(value.imag());
        }

        template <>
        inline bool DataInitialization::isBadInput<int32_t>(int32_t value)
        {
            return value == DataInitialization::getValue<int32_t, InitMode::BadInput>();
        }

        template <>
        inline bool DataInitialization::isBadInput<Int8x4>(Int8x4 value)
        {
            return value == DataInitialization::getValue<Int8x4, InitMode::BadInput>();
        }

        template <>
        inline bool DataInitialization::isBadInput<Half>(Half value)
        {
            return std::isnan(static_cast<float>(value));
        }

        template <>
        inline bool DataInitialization::isBadInput<BFloat16>(BFloat16 value)
        {
            return std::isnan(value);
        }

        template <>
        inline bool DataInitialization::isBadInput<Float8>(Float8 value)
        {
            return std::isnan(value);
        }

        template <>
        inline bool DataInitialization::isBadInput<BFloat8>(BFloat8 value)
        {
            return std::isnan(value);
        }

        template <>
        inline bool DataInitialization::isBadInput<int8_t>(int8_t value)
        {
            return value == DataInitialization::getValue<int8_t, InitMode::BadInput>();
        }

        template <>
        inline bool DataInitialization::isBadOutput<float>(float value)
        {
            return std::isinf(value);
        }

        template <>
        inline bool DataInitialization::isBadOutput<double>(double value)
        {
            return std::isinf(value);
        }

        template <>
        inline bool DataInitialization::isBadOutput<std::complex<float>>(std::complex<float> value)
        {
            return std::isinf(value.real()) && std::isinf(value.imag());
        }

        template <>
        inline bool
            DataInitialization::isBadOutput<std::complex<double>>(std::complex<double> value)
        {
            return std::isinf(value.real()) && std::isinf(value.imag());
        }

        template <>
        inline bool DataInitialization::isBadOutput<int32_t>(int32_t value)
        {
            return value == DataInitialization::getValue<int32_t, InitMode::BadOutput>();
        }

        template <>
        inline bool DataInitialization::isBadOutput<Int8x4>(Int8x4 value)
        {
            return value == DataInitialization::getValue<Int8x4, InitMode::BadOutput>();
        }

        template <>
        inline bool DataInitialization::isBadOutput<Half>(Half value)
        {
            return std::isinf(static_cast<float>(value));
        }

        template <>
        inline bool DataInitialization::isBadOutput<BFloat16>(BFloat16 value)
        {
            return std::isinf(value);
        }

        template <>
        inline bool DataInitialization::isBadOutput<Float8>(Float8 value)
        {
            return std::isinf(value);
        }

        template <>
        inline bool DataInitialization::isBadOutput<BFloat8>(BFloat8 value)
        {
            return std::isinf(value);
        }

        template <>
        inline bool DataInitialization::isBadOutput<int8_t>(int8_t value)
        {
            return value == DataInitialization::getValue<int8_t, InitMode::BadOutput>();
        }

        template <>
        inline float DataInitialization::getTrigValue<float>(int idx, bool useCos, bool useAbs)
        {
            float val = useCos ? cos(idx) : sin(idx);
            if(useAbs)
                val = abs(val);
            return val;
        }

        template <>
        inline double DataInitialization::getTrigValue<double>(int idx, bool useCos, bool useAbs)
        {
            double val = useCos ? cos(idx) : sin(idx);
            if(useAbs)
                val = abs(val);
            return val;
        }

        template <>
        inline Half DataInitialization::getTrigValue<Half>(int idx, bool useCos, bool useAbs)
        {
            return static_cast<Half>(getTrigValue<float>(idx, useCos, useAbs));
        }

        template <>
        inline BFloat16
            DataInitialization::getTrigValue<BFloat16>(int idx, bool useCos, bool useAbs)
        {
            return static_cast<BFloat16>(getTrigValue<float>(idx, useCos, useAbs));
        }

        template <>
        inline Float8 DataInitialization::getTrigValue<Float8>(int idx, bool useCos, bool useAbs)
        {
            return static_cast<Float8>(getTrigValue<float>(idx, useCos, useAbs));
        }

        template <>
        inline BFloat8 DataInitialization::getTrigValue<BFloat8>(int idx, bool useCos, bool useAbs)
        {
            return static_cast<BFloat8>(getTrigValue<float>(idx, useCos, useAbs));
        }

        template <>
        inline int32_t DataInitialization::getTrigValue<int32_t>(int idx, bool useCos, bool useAbs)
        {
            throw std::runtime_error("Trig not available for int32_t.");
        }

        template <>
        inline Int8x4 DataInitialization::getTrigValue<Int8x4>(int idx, bool useCos, bool useAbs)
        {
            throw std::runtime_error("Trig not available for Int8x4.");
        }

        template <>
        inline int8_t DataInitialization::getTrigValue<int8_t>(int idx, bool useCos, bool useAbs)
        {
            throw std::runtime_error("Trig not available for Int8.");
        }

        template <>
        inline std::complex<float>
            DataInitialization::getTrigValue<std::complex<float>>(int idx, bool useCos, bool useAbs)
        {
            return std::complex<float>(getTrigValue<float>(idx, useCos, useAbs),
                                       getTrigValue<float>(idx, useCos, useAbs));
        }

        template <>
        inline std::complex<double> DataInitialization::getTrigValue<std::complex<double>>(
            int idx, bool useCos, bool useAbs)
        {
            return std::complex<double>(getTrigValue<double>(idx, useCos, useAbs),
                                        getTrigValue<double>(idx, useCos, useAbs));
        }

        template <typename>
        struct FP_PARAM;

        template <>
        struct FP_PARAM<double>
        {
            using UINT_T                = uint64_t;
            static constexpr int NUMSIG = 52;
            static constexpr int NUMEXP = 11;
        };

        template <>
        struct FP_PARAM<float>
        {
            using UINT_T                = uint32_t;
            static constexpr int NUMSIG = 23;
            static constexpr int NUMEXP = 8;
        };

        template <>
        struct FP_PARAM<BFloat16>
        {
            using UINT_T                = uint16_t;
            static constexpr int NUMSIG = 7;
            static constexpr int NUMEXP = 8;
        };

        template <>
        struct FP_PARAM<Half>
        {
            using UINT_T                = uint16_t;
            static constexpr int NUMSIG = 10;
            static constexpr int NUMEXP = 5;
        };

        template <>
        struct FP_PARAM<Float8>
        {
            using UINT_T                = uint8_t;
            static constexpr int NUMSIG = 3;
            static constexpr int NUMEXP = 4;
        };

        template <>
        struct FP_PARAM<BFloat8>
        {
            using UINT_T                = uint8_t;
            static constexpr int NUMSIG = 2;
            static constexpr int NUMEXP = 5;
        };

        template <typename T>
        struct rocm_random_common : FP_PARAM<T>
        {
            using typename FP_PARAM<T>::UINT_T;
            using FP_PARAM<T>::NUMSIG;
            using FP_PARAM<T>::NUMEXP;
            using random_fp_int_dist = std::uniform_int_distribution<UINT_T>;

            static_assert(sizeof(UINT_T) == sizeof(T), "Type sizes do not match");
            static constexpr UINT_T expmask = (((UINT_T)1 << NUMEXP) - 1) << NUMSIG;
            static constexpr UINT_T expbias = ((UINT_T)1 << (NUMEXP - 1)) - 1;
            inline static T         signsig_exp(UINT_T signsig, UINT_T exp)
            {
                union
                {
                    UINT_T u;
                    T      fp;
                };
                u = signsig & ~expmask | ((exp + expbias) << NUMSIG) & expmask;
                return fp;
            }
        };

        template <>
        inline BFloat16
            rocm_random_common<BFloat16>::signsig_exp(FP_PARAM<BFloat16>::UINT_T signsig,
                                                      FP_PARAM<BFloat16>::UINT_T exp)
        {
            FP_PARAM<BFloat16>::UINT_T u;
            u = signsig & ~expmask | ((exp + expbias) << NUMSIG) & expmask;
            return static_cast<BFloat16>(u);
        }

        template <typename T, int LOW_EXP, int HIGH_EXP>
        struct rocm_random : rocm_random_common<T>
        {
            using typename rocm_random_common<T>::random_fp_int_dist;
            __attribute__((flatten)) T operator()()
            {
                static std::mt19937 rng;
                int                 exp = std::uniform_int_distribution<int>{}(rng);
                exp                     = exp % (HIGH_EXP - LOW_EXP + 1) + LOW_EXP;
                return this->signsig_exp(random_fp_int_dist{}(rng), exp);
            }
        };

        template <typename T>
        struct rocm_random_narrow_range;

        template <>
        struct rocm_random_narrow_range<double> : rocm_random<double, -189, 0>
        {
        };

        template <>
        struct rocm_random_narrow_range<float> : rocm_random<float, -100, 0>
        {
        };

        template <>
        struct rocm_random_narrow_range<BFloat16> : rocm_random<BFloat16, -100, 0>
        {
        };

        template <>
        struct rocm_random_narrow_range<Half> : rocm_random<Half, -100, 0>
        {
        };

        // TODO: need to check for fp8 and bf8
        template <>
        struct rocm_random_narrow_range<Float8> : rocm_random<Float8, -100, 0>
        {
        };

        template <>
        struct rocm_random_narrow_range<BFloat8> : rocm_random<BFloat8, -100, 0>
        {
        };

        template <>
        inline float DataInitialization::getValue<float, InitMode::RandomNarrow>()
        {
            return rocm_random_narrow_range<float>{}();
        }

        template <>
        inline double DataInitialization::getValue<double, InitMode::RandomNarrow>()
        {
            return rocm_random_narrow_range<double>{}();
        }

        template <>
        inline BFloat16 DataInitialization::getValue<BFloat16, InitMode::RandomNarrow>()
        {
            return rocm_random_narrow_range<BFloat16>{}();
        }

        template <>
        inline Half DataInitialization::getValue<Half, InitMode::RandomNarrow>()
        {
            return rocm_random_narrow_range<Half>{}();
        }

        template <>
        inline Float8 DataInitialization::getValue<Float8, InitMode::RandomNarrow>()
        {
            return rocm_random_narrow_range<Float8>{}();
        }

        template <>
        inline BFloat8 DataInitialization::getValue<BFloat8, InitMode::RandomNarrow>()
        {
            return rocm_random_narrow_range<BFloat8>{}();
        }

        template <>
        inline std::complex<float>
            DataInitialization::getValue<std::complex<float>, InitMode::RandomNarrow>()
        {
            return std::complex<float>(rocm_random_narrow_range<float>{}(),
                                       rocm_random_narrow_range<float>{}());
        }

        template <>
        inline std::complex<double>
            DataInitialization::getValue<std::complex<double>, InitMode::RandomNarrow>()
        {
            return std::complex<double>(rocm_random_narrow_range<double>{}(),
                                        rocm_random_narrow_range<double>{}());
        }

        template <>
        inline int32_t DataInitialization::getValue<int32_t, InitMode::RandomNarrow>()
        {
            return getValue<int32_t, InitMode::Random>();
        }

        template <>
        inline Int8x4 DataInitialization::getValue<Int8x4, InitMode::RandomNarrow>()
        {
            return getValue<Int8x4, InitMode::Random>();
        }

        template <>
        inline int8_t DataInitialization::getValue<int8_t, InitMode::RandomNarrow>()
        {
            return getValue<int8_t, InitMode::Random>();
        }
    } // namespace Client
} // namespace Tensile
