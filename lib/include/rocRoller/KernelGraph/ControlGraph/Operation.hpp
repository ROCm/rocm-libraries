#pragma once

#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

#include <rocRoller/KernelGraph/CoordinateGraph/Dimension.hpp>

#include "Expression_fwd.hpp"
#include "InstructionValues/Register_fwd.hpp"
#include "Operation_fwd.hpp"
#include "Utilities/Utils.hpp"

namespace rocRoller
{
    namespace KernelGraph::ControlGraph
    {
        /*
         * Control flow graph nodes.
         */

        /**
         * BaseOperation - base class for representing commands.
         */
        struct BaseOperation
        {
            virtual std::string toString() const = 0;
        };

        /**
         * Kernel - represents the start of a kernel.
         */
        struct Kernel
        {
            std::string toString() const
            {
                return "Kernel";
            }
        };

        /**
         * Scope - represents a register scope.
         */
        struct Scope
        {
            std::string toString() const
            {
                return "Scope";
            }
        };

        /**
         * SetCoordinate - Sets the value of a Coordinate
         */
        struct SetCoordinate
        {
            SetCoordinate(Expression::ExpressionPtr value)
                : value(value)
            {
            }

            Expression::ExpressionPtr value;

            std::string toString() const
            {
                return "SetCoordinate";
            }
        };

        /**
         * ForLoopOp - Represents a for loop.
         *
         * Must have nodes connected via the following outgoing edges:
         *
         * - Initialize: Always executed once, when entering the for loop
         * - Body: The loop body.
         * - ForLoopIncrement: Executed after each iteration.
         *
         * There may be multiple outgoing edges for any of these.  Code that follows the for loop should be connected via a Sequence edge.
         *
         * condition is a scalar condition and is executed before each iteration to determine if we must exit the for loop.
         *
         * Currently generates code that behaves like:
         *
         * <Initialize>
         * if(!condition) goto for_bottom
         * for_top:
         * <Body>
         * <ForLoopIncrement>
         * if(condition) goto for_top
         * for_bottom:
         * <Sequence>
         */
        struct ForLoopOp
        {
            Expression::ExpressionPtr condition;

            std::string toString() const
            {
                return concatenate("ForLoopOp: ", condition);
            }
        };

        /**
         * UnrollOp - a kernel unroll.
         */
        struct UnrollOp
        {
            Expression::ExpressionPtr size;

            std::string toString() const
            {
                return concatenate("UnrollOp");
            }
        };

        /*
         * Computes the value of `expression` and stores it into the associated register.
         *
         * If the register already exists, it must be of type 'regType'.  If not, `regType`
         * specifies which type of register will be allocated.
         */
        struct Assign
        {
            Register::Type            regType;
            Expression::ExpressionPtr expression;

            size_t valueCount = 1;

            std::string toString() const
            {
                return concatenate("Assign ", regType, " ", expression);
            }
        };

        /**
         * @brief Represents a memory barrier
         *
         */
        struct Barrier
        {
            std::string toString() const
            {
                return "Barrier";
            }
        };

        /**
         * @brief Computes offsets and strides between coordinates.
         *
         * Offsets and strides into the `target` dimension, based on
         * incrementing the `increment` dimension.
         *
         * @param target Target dimension.
         * @param increment Increment dimension
         * @param base
         */
        struct ComputeIndex
        {
            // TODO: might be nicer to have UInt32 for strides; need
            // to allow user to specify stride types instead of
            // forcing size_t.
            ComputeIndex() = default;
            ComputeIndex(int                        target,
                         int                        increment,
                         int                        base,
                         int                        offset,
                         int                        stride,
                         int                        buffer,
                         bool                       forward,
                         DataType                   valueType,
                         std::initializer_list<int> zero       = {},
                         DataType                   offsetType = DataType::UInt64,
                         DataType                   strideType = DataType::UInt64)
                : target(target)
                , increment(increment)
                , base(base)
                , offset(offset)
                , stride(stride)
                , buffer(buffer)
                , forward(forward)
                , zero(zero)
                , valueType(valueType)
                , offsetType(offsetType)
                , strideType(strideType)
            {
            }

            bool             forward;
            int              target, increment, base, offset, stride, buffer;
            std::vector<int> zero;

            DataType valueType, offsetType, strideType;

            std::string toString() const
            {
                return "ComputeIndex";
            }
        };

        /**
         * @brief Deallocates a register.
         */
        struct Deallocate
        {
            std::string toString() const
            {
                return "Deallocate";
            }
        };

        /**
         * LoadLinear - Load linear dimension.
         */
        struct LoadLinear : public BaseOperation
        {
            LoadLinear() = delete;
            LoadLinear(rocRoller::VariableType const varType)
                : varType(varType)
            {
            }

            rocRoller::VariableType varType;

            std::string toString() const override
            {
                return "LoadLinear";
            }
        };

        /**
         * LoadTiled.  Loads a tile (typically a MacroTile or
         * WaveTile).
         *
         * Storage location (LDS, VGPR, etc) is specified by the
         * `MemoryType` member of the MacroTile node.
         *
         * When loading a WaveTile, the storage layout (for MFMA
         * instructions) is specified by the `LayoutType` member of
         * the the WaveTile node.
         */
        struct LoadTiled : public BaseOperation
        {
            LoadTiled() = delete;
            LoadTiled(VariableType const varType)
                : vtype(varType)
            {
            }

            VariableType vtype;

            std::string toString() const override
            {
                return concatenate("LoadTiled");
            }
        };

        /**
         * LoadVGPR - replaces LoadLinear.
         */
        struct LoadVGPR : public BaseOperation
        {
            LoadVGPR() = delete;
            LoadVGPR(VariableType const varType, bool const scalar = false)
                : vtype(varType)
                , scalar(scalar)
            {
            }

            VariableType vtype;
            bool         scalar;

            std::string toString() const override
            {
                return "LoadVGPR";
            }
        };

        /**
         * LoadLDSTile - loads a tile from LDS
         */
        struct LoadLDSTile : BaseOperation
        {
            LoadLDSTile() = delete;
            LoadLDSTile(VariableType const varType)
                : vtype(varType)
            {
            }

            VariableType vtype;

            std::string toString() const override
            {
                return "LoadLDSTile";
            }
        };

        /**
         * Multiply - Multiply two MacroTiles
         */
        struct Multiply : BaseOperation
        {
            std::string toString() const override
            {
                return "Multiply";
            }
        };

        /**
         * StoreLinear - Store linear dimension.
         */
        struct StoreLinear : public BaseOperation
        {
            std::string toString() const override
            {
                return "StoreLinear";
            }
        };

        /**
         * StoreTiled.  Stores a tile.
         *
         * Storage location and affinity is specified by the MacroTile
         * node.
         */
        struct StoreTiled : public BaseOperation
        {
            StoreTiled() = delete;
            StoreTiled(DataType const dtype)
                : dataType(dtype)
            {
            }

            DataType dataType;

            std::string toString() const override
            {
                return "StoreTiled";
            }
        };

        /**
         * StoreVGPR - replaces StoreLinear.
         */
        struct StoreVGPR : public BaseOperation
        {
            std::string toString() const override
            {
                return "StoreVGPR";
            }
        };

        /**
         * StoreLDSTile - store a tile into LDS
         */
        struct StoreLDSTile : public BaseOperation
        {
            StoreLDSTile() = delete;
            StoreLDSTile(DataType const dtype)
                : dataType(dtype)
            {
            }

            DataType dataType;

            std::string toString() const override
            {
                return "StoreLDSTile";
            }
        };

        /**
         * TensorContraction - Tensor contraction operation.
         */
        struct TensorContraction : public BaseOperation
        {
            TensorContraction() = delete;
            TensorContraction(int const               a,
                              int const               b,
                              std::vector<int> const& aContractedDimensions,
                              std::vector<int> const& bContractedDimensions)
                : a(a)
                , b(b)
                , aDims(aContractedDimensions)
                , bDims(bContractedDimensions)
            {
            }

            int              a, b;
            std::vector<int> aDims, bDims; // contracted dimensions

            std::string toString() const override
            {
                return concatenate("TensorContraction(", a, ", ", b, ")");
            }
        };

        /*
         * Helpers
         */

        inline std::string toString(const Operation& x)
        {
            return std::visit([](const auto& a) { return a.toString(); }, x);
        }
    }
}
