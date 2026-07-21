/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <Tensile/ArithmeticUnitTypes.hpp>
#include <Tensile/KernelLanguageTypes.hpp>
#include <Tensile/PerformanceMetricTypes.hpp>
#include <Tensile/ScalarValueTypes.hpp>
#include <Tensile/Tensile.hpp>

#include <Tensile/ContractionProblem_fwd.hpp>
#include <Tensile/ContractionSolution_fwd.hpp>

#include <Tensile/TensorDescriptor.hpp>
#include <Tensile/TensorOps.hpp>
#include <Tensile/Utils.hpp>

namespace Tensile
{
    /**
 * \addtogroup Problem
 * @{
 */

    /**
 * Describes a tensor contraction in by using TensorDescriptor objects for
 * each input or output tensor as well as indices describing transposes,
 * summations, etc. This is decoupled from any particular pointers, which
 * are provided in ContractionInputs objects.
 */
    class TENSILE_API ContractionProblem : public Problem
    {
    public:
        using Solution = ContractionSolution;
        using Inputs   = ContractionInputs;

        ContractionProblem() = default;

        /**
   * Zero-padding description
   */
        struct ZeroPad
        {
            ZeroPad(int32_t ai = -1, int32_t bi = -1, int64_t ps = 0, int64_t pe = 0)
                : anchorIndex(ai)
                , anchorPos(-1)
                , boundIndex(bi)
                , padStart(ps)
                , padEnd(pe){};

            int32_t anchorIndex;
            int32_t anchorPos; //! position of anchorIndex in A or B tensor
            int32_t boundIndex;
            int32_t boundPos; //! position of anchroIndex in A or B tensor
            int64_t padStart;
            int64_t padEnd;

            bool valid() const
            {
                return anchorIndex != -1;
            };
            std::string description() const;
        };
        using ZeroPads = std::vector<ZeroPad>;

        /**
   * Represents a pair of free indices in a tensor contraction.
   */
        struct FreeIndex
        {
            bool   isA; //< True=index is in A; False=index is in B
            size_t i; //< Dimension in A or B (depending on isA)
            size_t c; //< Dimension of C which corresponds for this index
            size_t d; //< Dimension of D which corresponds for this index
        };
        using FreeIndices = std::vector<FreeIndex>;

        /**
   * Represents a batched index in a tensor contraction.
   */
        struct BatchIndex
        {
            size_t a, b, c, d;
        };
        using BatchIndices = std::vector<BatchIndex>;

        /**
   * Represents a bound (or summed) index in a tensor contraction.
   */
        struct BoundIndex
        {
            BoundIndex(size_t xa = 0, size_t xb = 0, bool aMirror = false, bool bMirror = false)
                : a(xa)
                , b(xb)
                , aMirror(aMirror)
                , bMirror(bMirror){};
            size_t  a, b; //! positions in a or b tensor
            ZeroPad aZeroPad;
            ZeroPad bZeroPad;
            bool    aMirror, bMirror;
        };
        using BoundIndices = std::vector<BoundIndex>;

        virtual std::string description() const;

        /**
   * Create a ContractionProblem representing a batched GEMM, specifying
   * strides between matrices.
   */
        static ContractionProblem GEMM_Strides(bool     transA,
                                               bool     transB,
                                               DataType aType,
                                               DataType bType,
                                               DataType cType,
                                               DataType dType,
                                               size_t   m,
                                               size_t   n,
                                               size_t   k,
                                               size_t   batchSize,
                                               size_t   lda,
                                               size_t   aStride,
                                               size_t   ldb,
                                               size_t   bStride,
                                               size_t   ldc,
                                               size_t   cStride,
                                               size_t   ldd,
                                               size_t   dStride,
                                               double   beta);

        /**
   * Create a ContractionProblem representing a batched GEMM, specifying
   * strides between matrices.
   */
        static ContractionProblem GEMM_Strides(bool     transA,
                                               bool     transB,
                                               DataType aType,
                                               DataType bType,
                                               DataType cType,
                                               DataType dType,
                                               size_t   m,
                                               size_t   n,
                                               size_t   k,
                                               size_t   batchSize,
                                               size_t   lda,
                                               size_t   aStride,
                                               size_t   aOffset,
                                               size_t   ldb,
                                               size_t   bStride,
                                               size_t   bOffset,
                                               size_t   ldc,
                                               size_t   cStride,
                                               size_t   cOffset,
                                               size_t   ldd,
                                               size_t   dStride,
                                               size_t   dOffset,
                                               double   beta);

        /**
   * Create a ContractionProblem representing a batched SGEMM, with
   * leading dimensions, but no strides.
   */
        static ContractionProblem GEMM(bool   transA,
                                       bool   transB,
                                       size_t m,
                                       size_t n,
                                       size_t k,
                                       size_t lda,
                                       size_t ldb,
                                       size_t ldc,
                                       double beta,
                                       bool   unused,
                                       size_t batchCount);

        /**
   * Create a ContractionProblem representing a batched SGEMM, with
   * leading dimensions, but no strides.
   */
        static ContractionProblem GEMM(bool   transA,
                                       bool   transB,
                                       size_t m,
                                       size_t n,
                                       size_t k,
                                       size_t lda,
                                       size_t offsetA,
                                       size_t ldb,
                                       size_t offsetB,
                                       size_t ldc,
                                       size_t offsetC,
                                       double beta,
                                       bool   unused,
                                       size_t batchCount);

        /**
   * Create a ContractionProblem representing a batched GEMM based on the
   * dimensions of each of the tensors.
   */
        static ContractionProblem GEMM(bool                    transA,
                                       bool                    transB,
                                       TensorDescriptor const& a,
                                       TensorOps const&        aOps,
                                       TensorDescriptor const& b,
                                       TensorOps const&        bOps,
                                       TensorDescriptor const& c,
                                       TensorOps const&        cOps,
                                       TensorDescriptor const& d,
                                       TensorOps const&        dOps,
                                       double                  beta);

        /**
   * Converts an identifier such as `Contraction_l_AlikC_Bjlk_Cijk_Dijk`
   * into a set of indices and operations.
   */
        static void IdentifierToIndices(std::string const& identifier,
                                        FreeIndices&       freeIndices,
                                        BatchIndices&      batchIndices,
                                        BoundIndices&      boundIndices,
                                        TensorOps&         aOps,
                                        TensorOps&         bOps,
                                        TensorOps&         cOps,
                                        TensorOps&         dOps);

        /**
   * Create a ContractionProblem from a definition of each index, the
   * size of each index, the strides of each tensor, and any operations.
   *
   * @param freeIndices  Free indices
   * @param batchIndices Batch indices
   * @param boundIndices Bound indices
   * @param indexSizes   Size of each index, in the order of appearance in
   *                     the D tensor.
   *
   * @param aType    Data type of A
   * @param aStrides Strides of A
   * @param aOps     Operations to apply to A as it is read
   *
   * @param bType    Data type of B
   * @param bStrides Strides of B
   * @param bOps     Operations to apply to B as it is read
   *
   * @param cType    Data type of C
   * @param cStrides Strides of C
   * @param cOps     Operations to apply to C as it is read
   *
   * @param dType    Data type of D
   * @param dStrides Strides of D
   * @param dOps     Operations to apply to D as it is read
   *
   * @param beta Representative value of beta. Is only used to possibly
   *             select a more efficient kernel if we know that
   *             `beta == 0` or `beta == 1`.
   */
        static ContractionProblem FromIndexSizes(FreeIndices const&         freeIndices,
                                                 BatchIndices const&        batchIndices,
                                                 BoundIndices const&        boundIndices,
                                                 std::vector<size_t> const& indexSizes,
                                                 DataType                   aType,
                                                 std::vector<size_t> const& aStrides,
                                                 TensorOps const&           aOps,
                                                 DataType                   bType,
                                                 std::vector<size_t> const& bStrides,
                                                 TensorOps const&           bOps,
                                                 DataType                   cType,
                                                 std::vector<size_t> const& cStrides,
                                                 TensorOps const&           cOps,
                                                 DataType                   dType,
                                                 std::vector<size_t> const& dStrides,
                                                 TensorOps const&           dOps,
                                                 double                     beta);

        /**
   * Create a ContractionProblem from a definition of each index, the
   * size of each index, the strides of each tensor, and any operations.
   *
   * @param freeIndices  Free indices
   * @param batchIndices Batch indices
   * @param boundIndices Bound indices
   * @param indexSizes   Size of each index, in the order of appearance in
   *                     the D tensor.
   *
   * @param aType    Data type of A
   * @param aStrides Strides of A
   * @param aOps     Operations to apply to A as it is read
   * @param aOffset  start offset of buffer A
   *
   * @param bType    Data type of B
   * @param bStrides Strides of B
   * @param bOps     Operations to apply to B as it is read
   * @param bOffset  start offset of buffer B
   *
   * @param cType    Data type of C
   * @param cStrides Strides of C
   * @param cOps     Operations to apply to C as it is read
   * @param cOffset  start offset of buffer C
   *
   * @param dType    Data type of D
   * @param dStrides Strides of D
   * @param dOps     Operations to apply to D as it is read
   * @param dOffset  start offset of buffer D
   *
   * @param beta Representative value of beta. Is only used to possibly
   *             select a more efficient kernel if we know that
   *             `beta == 0` or `beta == 1`.
   */
        static ContractionProblem FromIndexSizes(FreeIndices const&         freeIndices,
                                                 BatchIndices const&        batchIndices,
                                                 BoundIndices const&        boundIndices,
                                                 std::vector<size_t> const& indexSizes,
                                                 DataType                   aType,
                                                 std::vector<size_t> const& aStrides,
                                                 TensorOps const&           aOps,
                                                 size_t                     aOffset,
                                                 DataType                   bType,
                                                 std::vector<size_t> const& bStrides,
                                                 TensorOps const&           bOps,
                                                 size_t                     bOffset,
                                                 DataType                   cType,
                                                 std::vector<size_t> const& cStrides,
                                                 TensorOps const&           cOps,
                                                 size_t                     cOffset,
                                                 DataType                   dType,
                                                 std::vector<size_t> const& dStrides,
                                                 TensorOps const&           dOps,
                                                 size_t                     dOffset,
                                                 double                     beta);

        /**
   * Create a ContractionProblem based on an operation identifier such as
   * `Contraction_l_AlikC_Bjlk_Cijk_Dijk` and individual index sizes.
   *
   * @param operationIdentifier String that represents this exact
   *                            operation in terms of transposes, data
   *                            types, and operations.
   * @param indexSizes   Size of each index, in the order of appearance in
   *                     the D tensor.
   *
   * @param aType    Data type of A
   * @param aStrides Strides of A
   *
   * @param bType    Data type of B
   * @param bStrides Strides of B
   *
   * @param cType    Data type of C
   * @param cStrides Strides of C
   *
   * @param dType    Data type of D
   * @param dStrides Strides of D
   *
   * @param beta Representative value of beta. Is only used to possibly
   *             select a more efficient kernel if we know that
   *             `beta == 0` or `beta == 1`.
   */
        static ContractionProblem FromIndexSizes(std::string const&         operationIdentifier,
                                                 std::vector<size_t> const& indexSizes,
                                                 DataType                   aType,
                                                 std::vector<size_t> const& aStrides,
                                                 DataType                   bType,
                                                 std::vector<size_t> const& bStrides,
                                                 DataType                   cType,
                                                 std::vector<size_t> const& cStrides,
                                                 DataType                   dType,
                                                 std::vector<size_t> const& dStrides,
                                                 double                     beta);

        /**
   * Create a ContractionProblem based on an operation identifier such as
   * `Contraction_l_AlikC_Bjlk_Cijk_Dijk` and individual index sizes.
   *
   * @param operationIdentifier String that represents this exact
   *                            operation in terms of transposes, data
   *                            types, and operations.
   * @param indexSizes   Size of each index, in the order of appearance in
   *                     the D tensor.
   *
   * @param aType    Data type of A
   * @param aStrides Strides of A
   * @param aOffset  Data offset of buffer A
   *
   * @param bType    Data type of B
   * @param bStrides Strides of B
   * @param bOffset  Data offset of buffer B
   *
   * @param cType    Data type of C
   * @param cStrides Strides of C
   * @param cOffset  Data offset of buffer C
   *
   * @param dType    Data type of D
   * @param dStrides Strides of D
   * @param dOffset  Data offset of buffer D
   *
   * @param beta Representative value of beta. Is only used to possibly
   *             select a more efficient kernel if we know that
   *             `beta == 0` or `beta == 1`.
   */
        static ContractionProblem FromIndexSizes(std::string const&         operationIdentifier,
                                                 std::vector<size_t> const& indexSizes,
                                                 DataType                   aType,
                                                 std::vector<size_t> const& aStrides,
                                                 size_t                     aOffset,
                                                 DataType                   bType,
                                                 std::vector<size_t> const& bStrides,
                                                 size_t                     bOffset,
                                                 DataType                   cType,
                                                 std::vector<size_t> const& cStrides,
                                                 size_t                     cOffset,
                                                 DataType                   dType,
                                                 std::vector<size_t> const& dStrides,
                                                 size_t                     dOffset,
                                                 double                     beta);

        ContractionProblem(TensorDescriptor const& a,
                           TensorOps const&        aOps,
                           TensorDescriptor const& b,
                           TensorOps const&        bOps,
                           TensorDescriptor const& c,
                           TensorOps const&        cOps,
                           TensorDescriptor const& d,
                           TensorOps const&        dOps,
                           FreeIndices const&      freeIndices,
                           BatchIndices const&     batchIndices,
                           BoundIndices const&     boundIndices,
                           double                  beta,
                           size_t                  workspaceSize = 0);

        //! Returns size given original index assignment (in range
        //! 0..NumIndicesC+boundSizes)
        size_t size(size_t idx) const;

        size_t freeSizeA(size_t idx) const;
        size_t freeSizeB(size_t idx) const;

        size_t batchSize(size_t idx) const;
        size_t boundSize(size_t idx) const;

        size_t toAPos(size_t idx) const;
        size_t toBPos(size_t idx) const;

        // Translate specified index into a position of that index in the d tensor.
        // Since d tensor order is always index order this is 1:1 translation if the
        // index is in-bounds:
        size_t toDPos(size_t idx) const
        {
            if(idx < d().dimensions())
                return idx;
            else
                throw std::runtime_error("requested index not in D");
        }

        size_t toBoundsPos(size_t idx) const
        {
            if(idx < d().dimensions())
                throw std::runtime_error("invalid bounds index (is free or batch)");
            else if(idx > d().dimensions() + boundIndices().size())
                throw std::runtime_error("invalid bounds index (out-of-bounds)");
            else
                return idx - d().dimensions();
        }

        std::vector<size_t> const& problemSizes() const
        {
            return m_problemSizes;
        }

        std::vector<size_t> const& problemStrides() const
        {
            return m_problemStrides;
        }

        std::vector<size_t> const& convProblemSizes() const
        {
            return m_convProblemSizes;
        }
        void setConvProblemSizes(std::vector<size_t>& convProblemSizes)
        {
            m_convProblemSizes.assign(convProblemSizes.begin(), convProblemSizes.end());
        }

        void setCEqualsD(bool cEqualsD)
        {
            m_cEqualsD = cEqualsD;
        }

        bool cEqualsD() const
        {
            return m_cEqualsD;
        }

        double arithmeticIntensity() const
        {
            return m_arithmeticIntensity;
        }

        void setAlphaType(DataType type)
        {
            m_alphaType = type;
        }

        DataType alphaType() const
        {
            return m_alphaType;
        }

        void setAlphaRestriction(ScalarValue alpha)
        {
            m_alphaRestriction = alpha;
        }

        ScalarValue alphaRestriction() const
        {
            return m_alphaRestriction;
        }

        void setBetaType(DataType type)
        {
            m_betaType = type;
        }

        DataType betaType() const
        {
            return m_betaType;
        }

        void setBetaRestriction(ScalarValue beta)
        {
            m_betaRestriction = beta;
        }

        ScalarValue betaRestriction() const
        {
            return m_betaRestriction;
        }

        void setStridedBatched(bool value)
        {
            m_stridedBatched = value;
        }

        bool stridedBatched() const
        {
            return m_stridedBatched;
        }

        void setHighPrecisionAccumulate(bool value)
        {
            m_highPrecisionAccumulate = value;
        }

        bool highPrecisionAccumulate() const
        {
            return m_highPrecisionAccumulate;
        }

        void setArithmeticUnit(ArithmeticUnit value)
        {
            m_arithmeticUnit = value;
        }
        ArithmeticUnit arithmeticUnit() const
        {
            return m_arithmeticUnit;
        }

        void setKernelLanguage(KernelLanguage value)
        {
            m_kernelLanguage = value;
        }
        KernelLanguage kernelLanguage() const
        {
            return m_kernelLanguage;
        }

        void setPerformanceMetric(PerformanceMetric value)
        {
            m_performanceMetric = value;
        }

        PerformanceMetric performanceMetric() const
        {
            int  experimental = Debug::Instance().useExperimentalSelection();
            auto option       = static_cast<ExperimentalOption>(experimental);

            switch(option)
            {
            case ExperimentalOption::None:
                return m_performanceMetric;

            case ExperimentalOption::Grid:
                return PerformanceMetric::ExperimentalGrid;

            case ExperimentalOption::DTree:
                return PerformanceMetric::ExperimentalDTree;

            default:
                // warning?
                return m_performanceMetric;
            }
        }

        void setDeterministicMode(bool value)
        {
            m_deterministicMode = value;
        }
        bool deterministicMode() const
        {
            return m_deterministicMode;
        }

        void setFp16AltImpl(bool value)
        {
            m_fp16AltImpl = value;
        }

        bool fp16AltImpl() const
        {
            return m_fp16AltImpl;
        }

        void setStochasticRounding(bool value)
        {
            m_stochasticRounding = value;
        }

        bool stochasticRounding() const
        {
            return m_stochasticRounding;
        }

        void setF32XdlMathOp(DataType value)
        {
            m_f32XdlMathOp = value;
        }

        DataType f32XdlMathOp() const
        {
            return m_f32XdlMathOp;
        }

        void setFp16AltImplRound(bool value)
        {
            m_fp16AltImplRound = value;
        }

        bool fp16AltImplRound() const
        {
            return m_fp16AltImplRound;
        }

        /// Largest of the free and bound indices.  Does not include batch size.
        size_t maxProblemSize() const
        {
            return m_maxProblemSize;
        }

        /// Allocated elements excluding batch dimensions
        /// Used in assembly kernels to determine buffer limits, if batch dimes not
        /// packed
        size_t allocatedElementsNonBatchA() const
        {
            return m_allocatedElementsNonBatchA;
        }
        size_t allocatedElementsNonBatchB() const
        {
            return m_allocatedElementsNonBatchB;
        }

        size_t flopsPerMac() const;
        size_t flopCount() const;

        TensorDescriptor const& a() const
        {
            return m_a;
        }
        TensorDescriptor const& b() const
        {
            return m_b;
        }
        TensorDescriptor const& c() const
        {
            return m_c;
        }
        TensorDescriptor const& d() const
        {
            return m_d;
        }

        TensorOps const& aOps() const
        {
            return m_aOps;
        }
        TensorOps const& bOps() const
        {
            return m_bOps;
        }
        TensorOps const& cOps() const
        {
            return m_cOps;
        }
        TensorOps const& dOps() const
        {
            return m_dOps;
        }

        FreeIndices const& freeIndicesA() const
        {
            return m_freeIndicesA;
        }
        FreeIndices const& freeIndicesB() const
        {
            return m_freeIndicesB;
        }
        FreeIndices const& freeIndices() const
        {
            return m_freeIndices;
        }
        BatchIndices const& batchIndices() const
        {
            return m_batchIndices;
        }
        BoundIndices const& boundIndices() const
        {
            return m_boundIndices;
        }

        ZeroPads const& aZeroPad() const
        {
            return m_aZeroPads;
        }
        ZeroPads const& bZeroPad() const
        {
            return m_bZeroPads;
        }

        void addAZeroPad(const ZeroPad& zp);
        void addBZeroPad(const ZeroPad& zp);

        bool transposeC01() const
        {
            return m_transposeC01;
        };

        double beta() const
        {
            return m_beta;
        }

        std::string const& aNames() const
        {
            return m_aNames;
        }
        std::string const& bNames() const
        {
            return m_bNames;
        }
        std::string const& cNames() const
        {
            return m_cNames;
        }
        std::string const& dNames() const
        {
            return m_dNames;
        }
        std::string const& sumNames() const
        {
            return m_sumNames;
        }

        bool transA() const
        {
            return m_aNames == "lik";
        }
        bool transB() const
        {
            return m_bNames == "jlk";
        }

        std::string        operationName() const;
        std::string const& operationIdentifier() const
        {
            return m_operationIdentifier;
        }
        std::string operationDescription() const
        {
            return getOperationDescription();
        }

        void setWorkspaceSize(size_t size)
        {
            m_workspaceSize = size;
        }

        size_t workspaceSize() const
        {
            return m_workspaceSize;
        }

        size_t getNumTiles(SizeMapping const& sizeMapping) const;
        size_t getItersPerTile(SizeMapping const& sizeMapping) const;

        void checkPersistentKernelEligibility(ContractionSolution const& solution,
                                              Hardware const&            hardware);

        bool getPersistentKernelEligibility() const
        {
            return m_eligibleForPK;
        }

    private:
        TensorDescriptor m_a;
        TensorDescriptor m_b;
        TensorDescriptor m_c;
        TensorDescriptor m_d;
        TensorOps        m_aOps;
        TensorOps        m_bOps;
        TensorOps        m_cOps;
        TensorOps        m_dOps;

        std::string m_aNames;
        std::string m_bNames;
        std::string m_cNames;
        std::string m_dNames;
        std::string m_sumNames;
        std::string m_operationIdentifier;

        bool              m_transA;
        bool              m_transB;
        bool              m_cEqualsD                = false;
        bool              m_stridedBatched          = true;
        bool              m_highPrecisionAccumulate = false;
        bool              m_deterministicMode       = false;
        bool              m_eligibleForPK           = true;
        bool              m_fp16AltImpl             = false;
        bool              m_fp16AltImplRound        = false;
        bool              m_stochasticRounding      = false;
        DataType          m_f32XdlMathOp            = DataType::Float;
        ArithmeticUnit    m_arithmeticUnit          = ArithmeticUnit::Any;
        KernelLanguage    m_kernelLanguage          = KernelLanguage::Any;
        PerformanceMetric m_performanceMetric       = PerformanceMetric::DeviceEfficiency;

        double m_arithmeticIntensity;

        DataType m_alphaType = DataType::None; // if not assigned, will follow d-type
        DataType m_betaType  = DataType::None; // for bwd-compatible

        ScalarValue m_alphaRestriction = ScalarValue::Any; // restrictions on the alpha value used
        ScalarValue m_betaRestriction  = ScalarValue::Any; // restrictions on the beta value used

        FreeIndices  m_freeIndicesA; //< in same order as IndexAssignmentsA
        FreeIndices  m_freeIndicesB; //< in same order as IndexAssignmentsB
        FreeIndices  m_freeIndices;
        BatchIndices m_batchIndices;
        BoundIndices m_boundIndices;

        ZeroPads m_aZeroPads;
        ZeroPads m_bZeroPads;

        std::vector<size_t> m_freeSizesA;
        std::vector<size_t> m_freeSizesB;
        std::vector<size_t> m_batchSizes;
        std::vector<size_t> m_boundSizes;

        std::vector<size_t> m_problemSizes;
        std::vector<size_t> m_problemStrides;
        std::vector<size_t> m_convProblemSizes;

        bool   m_transposeC01;
        double m_beta;

        size_t m_maxProblemSize = 1;

        size_t m_allocatedElementsNonBatchA;
        size_t m_allocatedElementsNonBatchB;

        size_t m_workspaceSize;

        void normalize();
        void consistencyCheck() const;
        void calcArithmeticIntensity();

        void getIndexNames(std::string& aNames,
                           std::string& bNames,
                           std::string& cNames,
                           std::string& dNames,
                           std::string& sumNames) const;

        std::string getOperationIdentifier() const;
        std::string getOperationDescription() const;
    };

    struct TENSILE_API ContractionInputs : public ProblemInputs
    {
        ContractionInputs();
        virtual ~ContractionInputs();

        constexpr static uint32_t TypeId(DataType aType,
                                         DataType bType,
                                         DataType cType,
                                         DataType dType,
                                         DataType alphaType,
                                         DataType betaType)
        {
            static_assert(BitFieldGenerator::ElementWidth((uint32_t)DataType::Count) * 6
                              <= BitFieldGenerator::maxBitFieldWidth,
                          "Max bitfield width exceeded");

            return BitFieldGenerator::GenerateBitField(
                BitFieldGenerator::ElementWidth((uint32_t)DataType::Count),
                (uint32_t)aType,
                (uint32_t)bType,
                (uint32_t)cType,
                (uint32_t)dType,
                (uint32_t)alphaType,
                (uint32_t)betaType);
        }
    };

    /**
 * Contains actual pointer and argument values for a particular set of
 * inputs.
 */
    template <typename A, typename B, typename C, typename D, typename Alpha, typename Beta>
    struct TENSILE_API TypedContractionInputs : public ContractionInputs
    {
        using AType     = A;
        using BType     = B;
        using CType     = C;
        using DType     = D;
        using AlphaType = Alpha;
        using BetaType  = Beta;

        TypedContractionInputs();

        TypedContractionInputs(A const* _a,
                               B const* _b,
                               C const* _c,
                               D*       _d,
                               Alpha    _alpha,
                               Beta     _beta,
                               void*    _ws = nullptr)
            : TypedContractionInputs(
                _a, _b, _c, _d, nullptr, nullptr, nullptr, nullptr, _alpha, _beta){};

        TypedContractionInputs(A const*        _a,
                               B const*        _b,
                               C const*        _c,
                               D*              _d,
                               A const* const* _batchA,
                               B const* const* _batchB,
                               C const* const* _batchC,
                               D* const*       _batchD,
                               Alpha           _alpha,
                               Beta            _beta,
                               void*           _ws = nullptr);

        ~TypedContractionInputs();

        A const* a = nullptr;
        B const* b = nullptr;
        C const* c = nullptr;
        D*       d = nullptr;

        A const* const* batchA = nullptr;
        B const* const* batchB = nullptr;
        C const* const* batchC = nullptr;
        D* const*       batchD = nullptr;

        void* ws = nullptr;

        Alpha alpha = static_cast<Alpha>(0);
        Beta  beta  = static_cast<Beta>(0);

        constexpr static uint32_t TypeId()
        {
            return ContractionInputs::TypeId(TypeInfo<A>::Enum,
                                             TypeInfo<B>::Enum,
                                             TypeInfo<C>::Enum,
                                             TypeInfo<D>::Enum,
                                             TypeInfo<Alpha>::Enum,
                                             TypeInfo<Beta>::Enum);
        }
    };

    // Commonly used contraction input type groupings
    // Naming: _[Ti_To_Tc]_:
    // S=float, D=double, C=complex<float>, Z=complex<double>,
    // H=Half, B=BF16, I8x4=Int8x4, I32=int32_t
    using ContractionInputs_S_S_S = TypedContractionInputs<float>;
    using ContractionInputs_D_D_D = TypedContractionInputs<double>;
    using ContractionInputs_C_C_C = TypedContractionInputs<std::complex<float>>;
    using ContractionInputs_Z_Z_Z = TypedContractionInputs<std::complex<double>>;
#ifdef TENSILE_USE_HALF
    using ContractionInputs_H_H_H = TypedContractionInputs<Half>;
    using ContractionInputs_H_H_S = TypedContractionInputs<Half, Half, Half, Half, float, float>;
    using ContractionInputs_H_S_S = TypedContractionInputs<Half, Half, float, float>;
#endif // TENSILE_USE_HALF
    using ContractionInputs_I8x4_I32_I32 = TypedContractionInputs<Int8x4, Int8x4, int32_t, int32_t>;
    using ContractionInputs_I8_I32_I32   = TypedContractionInputs<int8_t, int8_t, int32_t, int32_t>;
    using ContractionInputs_I32_I32_I32  = TypedContractionInputs<int32_t>;
#ifdef TENSILE_USE_BF16
    using ContractionInputs_B_B_S
        = TypedContractionInputs<BFloat16, BFloat16, BFloat16, BFloat16, float, float>;
    using ContractionInputs_B_S_S = TypedContractionInputs<BFloat16, BFloat16, float, float>;
#endif // TENSILE_USE_BF16
#ifdef TENSILE_USE_FP8_BF8
    using ContractionInputs_F8_F8_S
        = TypedContractionInputs<Float8, Float8, Float8, Float8, float, float>;
    using ContractionInputs_F8_S_S = TypedContractionInputs<Float8, Float8, float, float>;
    using ContractionInputs_B8_B8_S
        = TypedContractionInputs<BFloat8, BFloat8, BFloat8, BFloat8, float, float>;
    using ContractionInputs_B8_S_S = TypedContractionInputs<BFloat8, BFloat8, float, float>;
    // hybrid cases: F8B8SS, B8F8SS
    using ContractionInputs_F8B8_S_S = TypedContractionInputs<Float8, BFloat8, float, float>;
    using ContractionInputs_B8F8_S_S = TypedContractionInputs<BFloat8, Float8, float, float>;
    // hybrid cases with To = B8
    using ContractionInputs_F8B8_B8_S
        = TypedContractionInputs<Float8, BFloat8, BFloat8, BFloat8, float, float>;
    using ContractionInputs_B8F8_B8_S
        = TypedContractionInputs<BFloat8, Float8, BFloat8, BFloat8, float, float>;
    // cases with To = f16
    using ContractionInputs_F8_H_S
        = TypedContractionInputs<Float8, Float8, Half, Half, float, float>;
    using ContractionInputs_B8_H_S
        = TypedContractionInputs<BFloat8, BFloat8, Half, Half, float, float>;
    using ContractionInputs_F8B8_H_S
        = TypedContractionInputs<Float8, BFloat8, Half, Half, float, float>;
    using ContractionInputs_B8F8_H_S
        = TypedContractionInputs<BFloat8, Float8, Half, Half, float, float>;
#endif // TENSILE_USE_FP8_BF8

    TENSILE_API std::ostream& operator<<(std::ostream&             stream,
                                         ContractionProblem const& contraction);

    TENSILE_API std::ostream& operator<<(std::ostream&                        stream,
                                         ContractionProblem::FreeIndex const& free);
    TENSILE_API std::ostream& operator<<(std::ostream&                         stream,
                                         ContractionProblem::BatchIndex const& batch);
    TENSILE_API std::ostream& operator<<(std::ostream&                         stream,
                                         ContractionProblem::BoundIndex const& bound);

    TENSILE_API std::istream& operator>>(std::istream& stream, ContractionProblem::FreeIndex& free);
    TENSILE_API std::istream& operator>>(std::istream&                   stream,
                                         ContractionProblem::BatchIndex& batch);
    TENSILE_API std::istream& operator>>(std::istream&                   stream,
                                         ContractionProblem::BoundIndex& bound);

    /**
 * @}
 */
} // namespace Tensile
