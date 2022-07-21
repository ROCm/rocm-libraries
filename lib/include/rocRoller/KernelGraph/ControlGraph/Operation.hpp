#pragma once

#include <string>
#include <vector>

#include <rocRoller/KernelGraph/CoordinateTransform/Dimension.hpp>
#include <rocRoller/KernelGraph/TagType.hpp>
#include <rocRoller/Operations/T_Execute_fwd.hpp>

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
         *
         * Requires a command tag which can be used to look up a
         * command within a Command() object.
         */
        struct BaseOperation
        {
            int tag;
            BaseOperation() = delete;
            BaseOperation(int tag)
                : tag(tag)
            {
            }

            virtual std::string toString() const = 0;
        };

        /**
         * Kernel - represents the start of a kernel.
         */
        struct Kernel
        {
            int tag = -100;

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
            int tag = -100;

            TagType counterTag;

            Expression::ExpressionPtr condition;

            std::string toString() const
            {
                return concatenate("ForLoopOp(", tag, "): ", condition);
            }
        };

        /**
         * Kernel - a kernel unroll.
         */
        struct UnrollOp
        {
            int                       tag = -100;
            Expression::ExpressionPtr size;

            std::string toString() const
            {
                return concatenate("UnrollOp(", tag, "): ");
            }
        };

        /*
         * Computes the value of `expression` and stores it into the register associated with
         * data flow tag `destTag`.
         *
         * If the register already exists, it must be of type 'regType'.  If not, `regType`
         * specifies which type of register will be allocated.
         */
        struct Assign
        {
            int tag = -100;

            int destTag = -1;

            Register::Type            regType;
            Expression::ExpressionPtr expression;

            size_t valueCount = 1;

            std::string toString() const
            {
                return concatenate(
                    "Assign(", tag, "): tag(", destTag, ") = ", regType, " ", expression);
            }
        };

        /**
         * @brief Represents a memory barrier
         *
         */
        struct Barrier
        {
            int tag;

            Barrier() = delete;
            Barrier(int tag)
                : tag(tag)
            {
            }

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
            LoadLinear() = delete;
            LoadLinear(int tag, CoordinateTransform::User user, CoordinateTransform::Linear linear)
                : BaseOperation(tag)
                , user(user)
                , linear(linear)
            {
            }

            CoordinateTransform::User   user;
            CoordinateTransform::Linear linear;

            virtual std::string toString() const override
            {
                return "LoadLinear(" + std::to_string(tag) + ")";
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
            LoadTiled(int tag, CoordinateTransform::User user, CoordinateTransform::MacroTile tile)
                : BaseOperation(tag)
                , user(user)
                , tile(tile)
            {
            }

            CoordinateTransform::User      user;
            CoordinateTransform::MacroTile tile;

            virtual std::string toString() const override
            {
                return concatenate("LoadTiled(", tag, ", ", user, ", ", tile, ")");
            }
        };

        /**
         * LoadVGPR - replaces LoadLinear.
         */
        struct LoadVGPR : public BaseOperation
        {
            LoadVGPR() = delete;
            LoadVGPR(int tag, CoordinateTransform::User user)
                : BaseOperation(tag)
                , user(user)
            {
            }

            CoordinateTransform::User user;

            virtual std::string toString() const override
            {
                return "LoadVGPR(" + std::to_string(tag) + ")";
            }
        };

        /**
         * LoadLDSTile - loads a tile from LDS
         */
        struct LoadLDSTile
        {
            int                            tag;
            CoordinateTransform::MacroTile tile;
            CoordinateTransform::LDS       lds;

            LoadLDSTile() = delete;
            LoadLDSTile(int tag, CoordinateTransform::MacroTile tile, CoordinateTransform::LDS lds)
                : tag(tag)
                , tile(tile)
                , lds(lds)
            {
            }

            virtual std::string toString() const
            {
                return "LoadLDSTile(" + std::to_string(tag) + ")";
            }
        };

        /**
         * Multiply - Multiply two MacroTiles
         */
        struct Multiply
        {
            Multiply() = delete;
            Multiply(int tag, int a, int b)
                : tag(tag)
                , a(a)
                , b(b)
            {
            }

            int tag, a, b;

            std::string toString() const
            {
                return concatenate("Multiply(", tag, ", ", a, ", ", b, ")");
            }
        };

        /**
         * StoreLinear - Store linear dimension.
         */
        struct StoreLinear : public BaseOperation
        {
            StoreLinear() = delete;
            StoreLinear(int tag, CoordinateTransform::Linear linear, CoordinateTransform::User user)
                : BaseOperation(tag)
                , linear(linear)
                , user(user)
            {
            }

            CoordinateTransform::User   user;
            CoordinateTransform::Linear linear;

            virtual std::string toString() const override
            {
                return "StoreLinear(" + std::to_string(tag) + ")";
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
            StoreTiled(int tag, CoordinateTransform::MacroTile tile, CoordinateTransform::User user)
                : BaseOperation(tag)
                , user(user)
                , tile(tile)
            {
            }

            CoordinateTransform::User      user;
            CoordinateTransform::MacroTile tile;

            virtual std::string toString() const override
            {
                return "StoreTiled(" + std::to_string(tag) + ")";
            }
        };

        /**
         * StoreVGPR - replaces StoreLinear.
         */
        struct StoreVGPR : public BaseOperation
        {
            StoreVGPR() = delete;
            StoreVGPR(int tag, CoordinateTransform::User user)
                : BaseOperation(tag)
                , user(user)
            {
            }

            CoordinateTransform::User user;

            virtual std::string toString() const override
            {
                return "StoreVGPR(" + std::to_string(tag) + ")";
            }
        };

        /**
         * StoreLDSTile - store a tile into LDS
         */
        struct StoreLDSTile
        {
            int                            tag;
            CoordinateTransform::MacroTile tile;
            CoordinateTransform::LDS       lds;

            StoreLDSTile() = delete;
            StoreLDSTile(int tag, CoordinateTransform::LDS lds, CoordinateTransform::MacroTile tile)
                : tag(tag)
                , lds(lds)
                , tile(tile)
            {
            }

            virtual std::string toString() const
            {
                return "StoreLDSTile(" + std::to_string(tag) + ")";
            }
        };

        /**
         * ElementOp - Elemental arithmetic operation.
         */
        struct ElementOp : public BaseOperation
        {
            ElementOp() = delete;
            ElementOp(int tag, std::shared_ptr<Operations::XOp> xop)
                : BaseOperation(tag)
                , xop(xop)
            {
            }

            std::shared_ptr<Operations::XOp> xop;

            virtual std::string toString() const override
            {
                return "ElementOp(" + std::to_string(tag) + ")";
            }
        };

        /**
         * TensorContraction - Tensor contraction operation.
         */
        struct TensorContraction : public BaseOperation
        {
            TensorContraction() = delete;
            TensorContraction(int                            tag,
                              CoordinateTransform::MacroTile a,
                              CoordinateTransform::MacroTile b,
                              std::vector<int>               aContractedDimensions,
                              std::vector<int>               bContractedDimensions)
                : BaseOperation(tag)
                , a(a)
                , b(b)
                , aDims(aContractedDimensions)
                , bDims(bContractedDimensions)
            {
            }

            CoordinateTransform::MacroTile a, b;
            std::vector<int>               aDims, bDims; // contracted dimensions

            virtual std::string toString() const override
            {
                return "TensorContraction(" + std::to_string(tag) + ", " + std::to_string(a.tag)
                       + ", " + std::to_string(b.tag) + ")";
            }
        };

        /*
         * Helpers
         */

        inline TagType getTag(const Operation& x)
        {
            int dtag = std::visit([](const auto a) { return a.tag; }, x);
            return {dtag, 0, false, static_cast<int>(x.index())};
        }

        inline std::string toString(const Operation& x)
        {
            return std::visit([](const auto& a) { return a.toString(); }, x);
        }
    }
}
