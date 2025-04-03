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

#include <gtest/gtest.h>

#include <rocRoller/Assemblers/Assembler.hpp>
#include <rocRoller/AssertOpKinds.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureTarget.hpp>
#include <rocRoller/Graph/Hypergraph.hpp>
#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/InstructionValues/RegisterAllocator.hpp>
#include <rocRoller/KernelGraph/ControlGraph/ControlGraph.hpp>
#include <rocRoller/KernelGraph/ControlToCoordinateMapper.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/CoordinateEdge.hpp>
#include <rocRoller/Operations/BlockScale.hpp>
#include <rocRoller/Operations/CommandArguments.hpp>
#include <rocRoller/Operations/RandomNumberGenerator.hpp>
#include <rocRoller/Scheduling/Costs/Cost.hpp>
#include <rocRoller/Scheduling/Observers/RegisterLivenessObserver.hpp>
#include <rocRoller/Scheduling/Scheduler.hpp>
#include <rocRoller/Utilities/Generator.hpp>
#include <rocRoller/Utilities/Settings.hpp>

template <typename T>
void verify(std::initializer_list<std::pair<T, std::string_view>> const pairs)
{
    for(auto const p : pairs)
        EXPECT_EQ(toString(std::get<T>(p)), std::get<std::string_view>(p))
            << "toString: " << toString(std::get<T>(p)) << " "
            << "Expect: " << toString(std::get<T>(p));
}

TEST(EnumToStringTest, ALL)
{
    using namespace rocRoller;

    verify<AssemblerType>({
        {AssemblerType::InProcess, "InProcess"},
        {AssemblerType::Subprocess, "Subprocess"},
    });

    verify<AssertOpKind>({
        {AssertOpKind::NoOp, "NoOp"},
        {AssertOpKind::MemoryViolation, "MemoryViolation"},
        {AssertOpKind::STrap, "STrap"},
    });

    verify<MemoryInstructions::MemoryDirection>({
        {MemoryInstructions::MemoryDirection::Load, "Load"},
        {MemoryInstructions::MemoryDirection::Store, "Store"},
    });

    verify<MemoryInstructions::MemoryKind>({
        {MemoryInstructions::MemoryKind::Global, "Global"},
        {MemoryInstructions::MemoryKind::Scalar, "Scalar"},
        {MemoryInstructions::MemoryKind::Local, "Local"},
        {MemoryInstructions::MemoryKind::Buffer, "Buffer"},
        {MemoryInstructions::MemoryKind::Buffer2LDS, "Buffer2LDS"},
    });

    verify<DataDirection>({
        {DataDirection::ReadOnly, "read_only"},
        {DataDirection::WriteOnly, "write_only"},
        {DataDirection::ReadWrite, "read_write"},
    });

    verify<DataType>({
        {DataType::Float, "Float"},
        {DataType::Double, "Double"},
        {DataType::ComplexFloat, "ComplexFloat"},
        {DataType::ComplexDouble, "ComplexDouble"},
        {DataType::Half, "Half"},
        {DataType::Halfx2, "Halfx2"},
        {DataType::BFloat16, "BFloat16"},
        {DataType::BFloat16x2, "BFloat16x2"},
        {DataType::FP8, "FP8"},
        {DataType::FP8x4, "FP8x4"},
        {DataType::BF8, "BF8"},
        {DataType::BF8x4, "BF8x4"},
        {DataType::Int8x4, "Int8x4"},
        {DataType::Int8, "Int8"},
        {DataType::Int16, "Int16"},
        {DataType::Int32, "Int32"},
        {DataType::Int64, "Int64"},
        {DataType::Raw32, "Raw32"},
        {DataType::UInt8x4, "UInt8x4"},
        {DataType::UInt8, "UInt8"},
        {DataType::UInt16, "UInt16"},
        {DataType::UInt32, "UInt32"},
        {DataType::UInt64, "UInt64"},
        {DataType::Bool, "Bool"},
        {DataType::Bool32, "Bool32"},
        {DataType::Bool64, "Bool64"},
        {DataType::None, "None"},
    });

    verify<PointerType>({
        {PointerType::Value, "Value"},
        {PointerType::PointerLocal, "PointerLocal"},
        {PointerType::PointerGlobal, "PointerGlobal"},
        {PointerType::Buffer, "Buffer"},
    });

    verify<MemoryType>({
        {MemoryType::Global, "Global"},
        {MemoryType::LDS, "LDS"},
        {MemoryType::AGPR, "AGPR"},
        {MemoryType::VGPR, "VGPR"},
        {MemoryType::WAVE, "WAVE"},
        {MemoryType::WAVE_LDS, "WAVE_LDS"},
        {MemoryType::WAVE_SPLIT, "WAVE_SPLIT"},
        {MemoryType::Literal, "Literal"},
        {MemoryType::None, "None"},
    });

    verify<LayoutType>({
        {LayoutType::SCRATCH, "SCRATCH"},
        {LayoutType::MATRIX_A, "MATRIX_A"},
        {LayoutType::MATRIX_B, "MATRIX_B"},
        {LayoutType::MATRIX_ACCUMULATOR, "MATRIX_ACCUMULATOR"},
        {LayoutType::None, "None"},
    });

    verify<NaryArgument>({
        {NaryArgument::DEST, "DEST"},
        {NaryArgument::LHS, "LHS"},
        {NaryArgument::RHS, "RHS"},
    });

    verify<GPUArchitectureGFX>({
        {GPUArchitectureGFX::UNKNOWN, "gfxunknown"},
        {GPUArchitectureGFX::GFX908, "gfx908"},
        {GPUArchitectureGFX::GFX90A, "gfx90a"},
        {GPUArchitectureGFX::GFX942, "gfx942"},
        {GPUArchitectureGFX::GFX1012, "gfx1012"},
        {GPUArchitectureGFX::GFX1030, "gfx1030"},
        {GPUArchitectureGFX::GFX1200, "gfx1200"},
        {GPUArchitectureGFX::GFX1201, "gfx1201"},
    });

    verify<Graph::ElementType>({
        {Graph::ElementType::Node, "Node"},
        {Graph::ElementType::Edge, "Edge"},
    });

    verify<Graph::Direction>({
        {Graph::Direction::Upstream, "Upstream"},
        {Graph::Direction::Downstream, "Downstream"},
    });

    verify<Graph::GraphModification>({
        {Graph::GraphModification::DeleteElement, "DeleteElement"},
        {Graph::GraphModification::AddElement, "AddElement"},
        {Graph::GraphModification::SetElement, "SetElement"},
    });

    verify<Register::Type>({
        {Register::Type::Literal, "Literal"},
        {Register::Type::Scalar, "SGPR"},
        {Register::Type::Vector, "VGPR"},
        {Register::Type::Accumulator, "ACCVGPR"},
        {Register::Type::LocalData, "LDS"},
        {Register::Type::Label, "Label"},
        {Register::Type::NullLiteral, "NullLiteral"},
        {Register::Type::SCC, "SCC"},
        {Register::Type::M0, "M0"},
        {Register::Type::VCC, "VCC"},
        {Register::Type::VCC_LO, "VCC_LO"},
        {Register::Type::VCC_HI, "VCC_HI"},
        {Register::Type::EXEC, "EXEC"},
        {Register::Type::EXEC_LO, "EXEC_LO"},
        {Register::Type::EXEC_HI, "EXEC_HI"},
    });

    verify<Register::AllocationState>({
        {Register::AllocationState::Unallocated, "Unallocated"},
        {Register::AllocationState::Allocated, "Allocated"},
        {Register::AllocationState::Freed, "Freed"},
        {Register::AllocationState::NoAllocation, "NoAllocation"},
        {Register::AllocationState::Error, "Error"},
    });

    verify<Register::AllocatorScheme>({
        {Register::AllocatorScheme::FirstFit, "FirstFit"},
        {Register::AllocatorScheme::PerfectFit, "PerfectFit"},
    });

    {
        using namespace KernelGraph::ControlGraph;

        verify<NodeOrdering>({
            {NodeOrdering::LeftFirst, "LeftFirst"},
            {NodeOrdering::LeftInBodyOfRight, "LeftInBodyOfRight"},
            {NodeOrdering::Undefined, "Undefined"},
            {NodeOrdering::RightInBodyOfLeft, "RightInBodyOfLeft"},
            {NodeOrdering::RightFirst, "RightFirst"},
        });

        verify<CacheStatus>({
            {CacheStatus::Invalid, "Invalid"},
            {CacheStatus::Partial, "Partial"},
            {CacheStatus::Valid, "Valid"},
        });
    }

    {
        using namespace KernelGraph::CoordinateGraph;
        verify<EdgeType>({
            {EdgeType::CoordinateTransform, "CoordinateTransform"},
            {EdgeType::DataFlow, "DataFlow"},
            {EdgeType::Any, "Any"},
        });
    }

    {
        using namespace KernelGraph::Connections;
        verify<ComputeIndexArgument>({
            {ComputeIndexArgument::TARGET, "TARGET"},
            {ComputeIndexArgument::INCREMENT, "INCREMENT"},
            {ComputeIndexArgument::BASE, "BASE"},
            {ComputeIndexArgument::OFFSET, "OFFSET"},
            {ComputeIndexArgument::STRIDE, "STRIDE"},
            {ComputeIndexArgument::BUFFER, "BUFFER"},
        });
    }

    verify<Operations::RandomNumberGenerator::SeedMode>({
        {Operations::RandomNumberGenerator::SeedMode::Default, "Default"},
        {Operations::RandomNumberGenerator::SeedMode::PerThread, "PerThread"},
    });

    verify<Operations::ScaleMode>({
        {Operations::ScaleMode::None, "None"},
        {Operations::ScaleMode::SingleScale, "SingleScale"},
        {Operations::ScaleMode::Separate, "Separate"},
        {Operations::ScaleMode::Inline, "Inline"},
    });

    verify<ArgumentType>({
        {ArgumentType::Value, "Value"},
        {ArgumentType::Limit, "Limit"},
        {ArgumentType::Size, "Size"},
        {ArgumentType::Stride, "Stride"},
    });

    {
        using namespace Scheduling;
        verify<CostFunction>({
            {CostFunction::None, "None"},
            {CostFunction::Uniform, "Uniform"},
            {CostFunction::MinNops, "MinNops"},
            {CostFunction::WaitCntNop, "WaitCntNop"},
            {CostFunction::LinearWeighted, "LinearWeighted"},
        });

        verify<RegisterLiveState>({
            {RegisterLiveState::Dead, " "},
            {RegisterLiveState::Write, "^"},
            {RegisterLiveState::Read, "v"},
            {RegisterLiveState::ReadWrite, "x"},
            {RegisterLiveState::Live, ":"},
            {RegisterLiveState::Allocated, "_"},
        });

        verify<SchedulerProcedure>({
            {SchedulerProcedure::Sequential, "Sequential"},
            {SchedulerProcedure::RoundRobin, "RoundRobin"},
            {SchedulerProcedure::Random, "Random"},
            {SchedulerProcedure::Cooperative, "Cooperative"},
            {SchedulerProcedure::Priority, "Priority"},
        });

        verify<Dependency>({
            {Dependency::None, "None"},
            {Dependency::SCC, "SCC"},
            {Dependency::VCC, "VCC"},
            {Dependency::Branch, "Branch"},
            {Dependency::Unlock, "Unlock"},
            {Dependency::M0, "M0"},
        });
    }

    verify<GeneratorState>({
        {GeneratorState::NoValue, "NoValue"},
        {GeneratorState::HasValue, "HasValue"},
        {GeneratorState::HasRange, "HasRange"},
        {GeneratorState::HasRangeValue, "HasRangeValue"},
        {GeneratorState::HasCopiedValue, "HasCopiedValue"},
    });

    verify<LogLevel>({
        {LogLevel::None, "None"},
        {LogLevel::Critical, "Critical"},
        {LogLevel::Error, "Error"},
        {LogLevel::Warning, "Warning"},
        {LogLevel::Terse, "Terse"},
        {LogLevel::Info, "Info"},
        {LogLevel::Verbose, "Verbose"},
        {LogLevel::Debug, "Debug"},
        {LogLevel::Trace, "Trace"},
    });

    {
        using namespace Expression;
        verify<EvaluationTime>({
            {EvaluationTime::Translate, "Translate"},
            {EvaluationTime::KernelLaunch, "KernelLaunch"},
            {EvaluationTime::KernelExecute, "KernelExecute"},
        });

        verify<AlgebraicProperty>({
            {AlgebraicProperty::Commutative, "Commutative"},
            {AlgebraicProperty::Associative, "Associative"},
        });

        verify<Category>({
            {Category::Arithmetic, "Arithmetic"},
            {Category::Comparison, "Comparison"},
            {Category::Logical, "Logical"},
            {Category::Conversion, "Conversion"},
            {Category::Value, "Value"},
        });
    }
}
