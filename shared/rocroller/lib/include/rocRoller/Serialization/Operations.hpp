/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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

#ifdef ROCROLLER_USE_LLVM
#include <llvm/ObjectYAML/YAML.h>
#endif

#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Operations/Operations_fwd.hpp>

#include "Base.hpp"
#include "Containers.hpp"
#include "Enum.hpp"
#include "Variant.hpp"

#include "XOps.hpp"

namespace rocRoller
{
    namespace Serialization
    {
        template <>
        struct ScalarTraits<Operations::OperationTag>
        {
            static std::string output(Operations::OperationTag const& x)
            {
                std::stringstream ss;
                ss << x.value;
                return ss.str();
            }

            static void input(std::string const& scalar, Operations::OperationTag& x)
            {
                std::stringstream ss(scalar);
                ss >> x.value;
            }
        };

        template <typename IO>
        struct SequenceTraits<std::vector<Operations::OperationTag>, IO>
            : public DefaultSequenceTraits<std::vector<Operations::OperationTag>, IO, true>
        {
        };

        template <typename IO, typename Context>
        struct MappingTraits<Operations::Tensor, IO, Context>
        {
            using TOp = Operations::Tensor;
            using iot = IOTraits<IO>;

            static void mapping(IO& io, TOp& op, Context& ctx)
            {
                iot::mapRequired(io, "tag", op.m_tag);
                iot::mapRequired(io, "variableType", op.m_variableType);
                iot::mapRequired(io, "numDims", op.m_numDims);
                iot::mapRequired(io, "sizes", op.m_sizes);
                iot::mapRequired(io, "pointer", op.m_pointer);
                iot::mapRequired(io, "extent", op.m_extent);
                iot::mapRequired(io, "strides", op.m_strides);
                iot::mapRequired(io, "literalStrides", op.m_literalStrides);
            }

            static void mapping(IO& io, TOp& val)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));
                Context ctx;
                mapping(io, val, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<Operations::Scalar, IO, Context>
        {
            using TOp = Operations::Scalar;
            using iot = IOTraits<IO>;

            static void mapping(IO& io, TOp& op, Context& ctx)
            {
                iot::mapRequired(io, "tag", op.m_tag);
                iot::mapRequired(io, "variableType", op.m_variableType);
                iot::mapRequired(io, "pointer", op.m_pointer);
            }

            static void mapping(IO& io, TOp& val)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));
                Context ctx;
                mapping(io, val, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<Operations::Literal, IO, Context>
        {
            using TOp = Operations::Literal;
            using iot = IOTraits<IO>;

            static void mapping(IO& io, TOp& op, Context& ctx)
            {
                iot::mapRequired(io, "tag", op.m_tag);
                iot::mapRequired(io, "value", op.m_value);
            }

            static void mapping(IO& io, TOp& val)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));
                Context ctx;
                mapping(io, val, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<Operations::BlockScale, IO, Context>
        {
            using TOp = Operations::BlockScale;
            using iot = IOTraits<IO>;

            static void mapping(IO& io, TOp& op, Context& ctx)
            {
                iot::mapRequired(io, "tag", op.m_tag);
                iot::mapRequired(io, "dataTag", op.m_data);
                iot::mapRequired(io, "strides", op.m_strides);

                // XXX Put this somewhere that can be re-used by others
                if(iot::outputting(io))
                {
                    if(op.m_scale.has_value())
                        iot::mapRequired(io, "scaleTag", *op.m_scale);
                }
                else
                {
                    Operations::OperationTag scaleTag{-1};
                    iot::mapOptional(io, "scaleTag", scaleTag);
                    if(scaleTag.value != -1)
                        op.m_scale = scaleTag;
                }
            }

            static void mapping(IO& io, TOp& val)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));
                Context ctx;
                mapping(io, val, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<Operations::T_Load_Linear, IO, Context>
        {
            using TOp = Operations::T_Load_Linear;
            using iot = IOTraits<IO>;

            static void mapping(IO& io, TOp& op, Context& ctx)
            {
                iot::mapRequired(io, "tag", op.m_tag);
                iot::mapRequired(io, "tensorTag", op.m_tensorTag);
            }

            static void mapping(IO& io, TOp& val)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));
                Context ctx;
                mapping(io, val, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<Operations::T_Load_Scalar, IO, Context>
        {
            using TOp = Operations::T_Load_Scalar;
            using iot = IOTraits<IO>;

            static void mapping(IO& io, TOp& op, Context& ctx)
            {
                iot::mapRequired(io, "tag", op.m_tag);
                iot::mapRequired(io, "scalarTag", op.m_scalarTag);
            }

            static void mapping(IO& io, TOp& val)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));
                Context ctx;
                mapping(io, val, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<Operations::T_Load_Tiled, IO, Context>
        {
            using TOp = Operations::T_Load_Tiled;
            using iot = IOTraits<IO>;

            static void mapping(IO& io, TOp& op, Context& ctx)
            {
                iot::mapRequired(io, "tag", op.m_tag);
                iot::mapRequired(io, "tensorTag", op.m_tensorTag);
            }

            static void mapping(IO& io, TOp& val)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));
                Context ctx;
                mapping(io, val, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<Operations::T_Mul, IO, Context>
        {
            using TOp = Operations::T_Mul;
            using iot = IOTraits<IO>;

            static void mapping(IO& io, TOp& op, Context& ctx)
            {
                iot::mapRequired(io, "tag", op.m_tag);
                iot::mapRequired(io, "a", op.a);
                iot::mapRequired(io, "b", op.b);
            }

            static void mapping(IO& io, TOp& val)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));
                Context ctx;
                mapping(io, val, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<Operations::T_Store_Linear, IO, Context>
        {
            using TOp = Operations::T_Store_Linear;
            using iot = IOTraits<IO>;

            static void mapping(IO& io, TOp& op, Context& ctx)
            {
                iot::mapRequired(io, "tag", op.m_tag);
                iot::mapRequired(io, "srcTag", op.m_srcTag);
                iot::mapRequired(io, "dstTag", op.m_dstTag);
            }

            static void mapping(IO& io, TOp& val)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));
                Context ctx;
                mapping(io, val, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<Operations::T_Store_Tiled, IO, Context>
        {
            using TOp = Operations::T_Store_Tiled;
            using iot = IOTraits<IO>;

            static void mapping(IO& io, TOp& op, Context& ctx)
            {
                iot::mapRequired(io, "tag", op.m_tag);
                iot::mapRequired(io, "srcTag", op.m_srcTag);
                iot::mapRequired(io, "dstTag", op.m_dstTag);
            }

            static void mapping(IO& io, TOp& val)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));
                Context ctx;
                mapping(io, val, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<Operations::Nop, IO, Context>
        {
            using TOp = Operations::Nop;
            using iot = IOTraits<IO>;

            static void mapping(IO& io, TOp& op, Context& ctx) {}

            static void mapping(IO& io, TOp& val)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, val, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<Operations::RandomNumberGenerator, IO, Context>
        {
            using TOp = Operations::RandomNumberGenerator;
            using iot = IOTraits<IO>;

            static void mapping(IO& io, TOp& op, Context& ctx)
            {
                iot::mapRequired(io, "tag", op.m_tag);
                iot::mapRequired(io, "seed", op.seed);
                iot::mapRequired(io, "mode", op.mode);
            }

            static void mapping(IO& io, TOp& val)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, val, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<Operations::T_Execute, IO, Context>
        {
            using TOp = Operations::T_Execute;
            using iot = IOTraits<IO>;

            static void mapping(IO& io, TOp& op, Context& ctx)
            {
                iot::mapRequired(io, "tag", op.m_tag);
                iot::mapRequired(io, "xops", op.m_xops);
                iot::mapRequired(io, "nextTag", op.m_nextTag);

                if(iot::outputting(io))
                {
                    std::vector<Operations::OperationTag> inputs, outputs;
                    std::copy(op.m_inputs.cbegin(), op.m_inputs.cend(), std::back_inserter(inputs));
                    std::copy(
                        op.m_outputs.cbegin(), op.m_outputs.cend(), std::back_inserter(outputs));
                    iot::mapRequired(io, "inputs", inputs);
                    iot::mapRequired(io, "outputs", outputs);
                }
                else
                {
                    std::vector<Operations::OperationTag> inputs, outputs;
                    iot::mapRequired(io, "inputs", inputs);
                    iot::mapRequired(io, "outputs", outputs);
                    std::copy(inputs.cbegin(),
                              inputs.cend(),
                              std::inserter(op.m_inputs, end(op.m_inputs)));
                    std::copy(outputs.cbegin(),
                              outputs.cend(),
                              std::inserter(op.m_outputs, end(op.m_outputs)));
                }
            }

            static void mapping(IO& io, TOp& val)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));
                Context ctx;
                mapping(io, val, ctx);
            }
        };

        template <>
        struct DefaultConstruct<Operations::Operation, Operations::Operation>
        {
            static Operations::Operation call()
            {
                return Operations::Nop();
            }
        };

        template <>
        struct DefaultConstruct<Operations::Operation, Operations::Tensor>
        {
            static Operations::Operation call()
            {
                return Operations::Tensor(-1, DataType::None);
            }
        };

        template <>
        struct DefaultConstruct<Operations::Operation, Operations::Scalar>
        {
            static Operations::Operation call()
            {
                return Operations::Scalar(DataType::None);
            }
        };

        template <>
        struct DefaultConstruct<Operations::Operation, Operations::Literal>
        {
            static Operations::Operation call()
            {
                return Operations::Literal(0.f);
            }
        };

        template <>
        struct DefaultConstruct<Operations::Operation, Operations::BlockScale>
        {
            static Operations::Operation call()
            {
                return Operations::BlockScale(Operations::OperationTag(-1), 1);
            }
        };

        template <>
        struct DefaultConstruct<Operations::Operation, Operations::RandomNumberGenerator>
        {
            static Operations::Operation call()
            {
                return Operations::RandomNumberGenerator(Operations::OperationTag(-1));
            }
        };

        template <>
        struct DefaultConstruct<Operations::Operation, Operations::T_Load_Linear>
        {
            static Operations::Operation call()
            {
                return Operations::T_Load_Linear(Operations::OperationTag(-1));
            }
        };

        template <>
        struct DefaultConstruct<Operations::Operation, Operations::T_Load_Scalar>
        {
            static Operations::Operation call()
            {
                return Operations::T_Load_Scalar(Operations::OperationTag(-1));
            }
        };

        template <>
        struct DefaultConstruct<Operations::Operation, Operations::T_Load_Tiled>
        {
            static Operations::Operation call()
            {
                return Operations::T_Load_Tiled(Operations::OperationTag(-1));
            }
        };

        template <>
        struct DefaultConstruct<Operations::Operation, Operations::T_Mul>
        {
            static Operations::Operation call()
            {
                return Operations::T_Mul(Operations::OperationTag(-1),
                                         Operations::OperationTag(-1));
            }
        };

        template <>
        struct DefaultConstruct<Operations::Operation, Operations::T_Store_Linear>
        {
            static Operations::Operation call()
            {
                return Operations::T_Store_Linear(Operations::OperationTag(-1),
                                                  Operations::OperationTag(-1));
            }
        };

        template <>
        struct DefaultConstruct<Operations::Operation, Operations::T_Store_Tiled>
        {
            static Operations::Operation call()
            {
                return Operations::T_Store_Tiled(Operations::OperationTag(-1),
                                                 Operations::OperationTag(-1));
            }
        };

        template <>
        struct DefaultConstruct<Operations::Operation, Operations::T_Execute>
        {
            static Operations::Operation call()
            {
                return Operations::T_Execute(Operations::OperationTag(-1));
            }
        };

        template <>
        struct DefaultConstruct<Operations::Operation, Operations::Nop>
        {
            static Operations::Operation call()
            {
                return Operations::Nop();
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<Operations::Operation, IO, Context>
            : public DefaultVariantMappingTraits<Operations::Operation, IO, Context>
        {
        };

        template <typename IO, typename Context>
        struct MappingTraits<std::shared_ptr<Operations::Operation>, IO, Context>
            : public SharedPointerMappingTraits<std::shared_ptr<Operations::Operation>,
                                                IO,
                                                Context,
                                                false>
        {
        };

        template <typename IO>
        struct SequenceTraits<std::vector<std::shared_ptr<Operations::Operation>>, IO>
            : public DefaultSequenceTraits<std::vector<std::shared_ptr<Operations::Operation>>,
                                           IO,
                                           false>
        {
        };
    }
}
