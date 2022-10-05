#pragma once

#include <string>
#include <vector>

#include <rocRoller/KernelGraph/CoordGraph/Dimension.hpp>
#include <rocRoller/Operations/T_Execute_fwd.hpp>

#include "Expression_fwd.hpp"
#include "InstructionValues/Register_fwd.hpp"
#include "Operation_fwd.hpp"
#include "Utilities/Utils.hpp"

namespace rocRoller
{
    namespace KernelGraph::ControlHypergraph
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
         * Kernel - a kernel unroll.
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
         * LoadLinear - Load linear dimension.
         */
        struct LoadLinear : public BaseOperation
        {
            virtual std::string toString() const override
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
            virtual std::string toString() const override
            {
                return concatenate("LoadTiled");
            }
        };

        /**
         * LoadVGPR - replaces LoadLinear.
         */
        struct LoadVGPR : public BaseOperation
        {
            virtual std::string toString() const override
            {
                return "LoadVGPR";
            }
        };

        /**
         * LoadLDSTile - loads a tile from LDS
         */
        struct LoadLDSTile : BaseOperation
        {
            virtual std::string toString() const
            {
                return "LoadLDSTile";
            }
        };

        /**
         * Multiply - Multiply two MacroTiles
         */
        struct Multiply
        {
            Multiply() = delete;
            // TODO
            // it's a link to a dimension in the coordinate graph
            // can have a type alias
            Multiply(int a, int b)
                : a(a)
                , b(b)
            {
            }

            int a, b;

            std::string toString() const
            {
                return concatenate("Multiply(", a, ", ", b, ")");
            }
        };

        /**
         * StoreLinear - Store linear dimension.
         */
        struct StoreLinear : public BaseOperation
        {
            virtual std::string toString() const override
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
            StoreTiled(DataType dtype)
                : dataType(dtype)
            {
            }

            DataType dataType;

            virtual std::string toString() const override
            {
                return "StoreTiled";
            }
        };

        /**
         * StoreVGPR - replaces StoreLinear.
         */
        struct StoreVGPR : public BaseOperation
        {
            virtual std::string toString() const override
            {
                return "StoreVGPR";
            }
        };

        /**
         * StoreLDSTile - store a tile into LDS
         */
        struct StoreLDSTile : public BaseOperation
        {
            virtual std::string toString() const
            {
                return "StoreLDSTile";
            }
        };

        /**
         * ElementOp - Elemental arithmetic operation.
         */
        struct ElementOp : public BaseOperation
        {
            ElementOp() = delete;
            ElementOp(std::shared_ptr<Operations::XOp> xop)
                : xop(xop)
            {
            }

            std::shared_ptr<Operations::XOp> xop;

            virtual std::string toString() const override
            {
                return "ElementOp";
            }
        };

        /**
         * TensorContraction - Tensor contraction operation.
         */
        struct TensorContraction : public BaseOperation
        {
            TensorContraction() = delete;
            TensorContraction(CoordGraph::MacroTile a,
                              CoordGraph::MacroTile b,
                              std::vector<int>      aContractedDimensions,
                              std::vector<int>      bContractedDimensions)
                : a(a)
                , b(b)
                , aDims(aContractedDimensions)
                , bDims(bContractedDimensions)
            {
            }

            CoordGraph::MacroTile a, b;
            std::vector<int>      aDims, bDims; // contracted dimensions

            virtual std::string toString() const override
            {
                return "TensorContraction";
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
