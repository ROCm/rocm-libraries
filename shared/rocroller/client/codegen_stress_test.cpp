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

#include <rocRoller/Context.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/InstructionValues/LabelAllocator.hpp>
#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/Serialization/YAML.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Settings.hpp>
#include <rocRoller/Utilities/Timer.hpp>

#include <CLI/CLI.hpp>

using namespace rocRoller;

struct CodeGenProblem
{
    std::string name;

    int         instCount;
    std::string instructions;

    int numWarmUp;
    int numRuns;
};

struct CodeGenResult : CodeGenProblem
{
    CodeGenResult(CodeGenProblem const& prob)
        : CodeGenProblem(prob)
    {
    }

    size_t              kernelGenerate;
    size_t              kernelAssemble;
    std::vector<size_t> kernelExecute;
};

template <typename IO>
struct rocRoller::Serialization::
    MappingTraits<CodeGenResult, IO, rocRoller::Serialization::EmptyContext>
{
    static const bool flow = false;
    using iot              = IOTraits<IO>;

    static void mapping(IO& io, CodeGenResult& result)
    {
        iot::mapRequired(io, "client", result.name);
        iot::mapRequired(io, "instCount", result.instCount);
        iot::mapRequired(io, "instructions", result.instructions);
        iot::mapRequired(io, "numWarmUp", result.numWarmUp);
        iot::mapRequired(io, "numRuns", result.numRuns);
        iot::mapRequired(io, "kernelGenerate", result.kernelGenerate);
        iot::mapRequired(io, "kernelAssemble", result.kernelAssemble);
        iot::mapRequired(io, "kernelExecute", result.kernelExecute);
    }

    static void mapping(IO& io, CodeGenResult& arch, EmptyContext& ctx)
    {
        mapping(io, arch);
    }
};

using namespace rocRoller;

std::vector<rocRoller::Register::ValuePtr> createRegisters(ContextPtr m_context,
                                                           rocRoller::Register::Type const regType,
                                                           rocRoller::DataType const       dataType,
                                                           size_t const                    amount,
                                                           int const regCount = 1)
{
    std::vector<rocRoller::Register::ValuePtr> regs;
    for(size_t i = 0; i < amount; i++)
    {
        auto reg
            = std::make_shared<rocRoller::Register::Value>(m_context, regType, dataType, regCount);
        reg->allocateNow();
        regs.push_back(reg);
    }
    return regs;
}

Generator<Instruction> comments(ContextPtr m_context)
{
    co_yield_(Instruction::Comment("Stress Test Comment"));
}

Generator<Instruction> simple_mfma(ContextPtr m_context)
{
    auto v = createRegisters(m_context, Register::Type::Vector, DataType::Float, 3);
    auto a = createRegisters(m_context, Register::Type::Accumulator, DataType::Float, 1);
    while(true)
    {
        co_yield_(Instruction("v_or_b32", {v[2]}, {v[0], v[1]}, {}, ""));
        co_yield_(Instruction("v_mfma_f32_16x16x4f32", {a[0]}, {v[0], v[2], a[0]}, {}, ""));
    }
}

Generator<Instruction> complex_mfma_with_coop(ContextPtr m_context)
{
    auto mfma_v = createRegisters(m_context, Register::Type::Vector, DataType::Float, 16);
    auto or_v   = createRegisters(m_context, Register::Type::Vector, DataType::Float, 4);

    auto generator_one = [&]() -> Generator<Instruction> {
        std::string comment = "stream1";
        co_yield_(Instruction(
            "v_mfma_f32_32x32x1f32", {mfma_v[0]}, {mfma_v[1], mfma_v[2], mfma_v[3]}, {}, comment));
        co_yield_(Instruction(
            "v_mfma_f32_32x32x1f32", {mfma_v[4]}, {mfma_v[5], mfma_v[6], mfma_v[7]}, {}, comment));
        co_yield_(Instruction("v_mfma_f32_32x32x1f32",
                              {mfma_v[8]},
                              {mfma_v[9], mfma_v[10], mfma_v[11]},
                              {},
                              comment));
    };
    auto generator_two = [&]() -> Generator<Instruction> {
        std::string comment = "stream2";
        while(true)
        {
            co_yield_(Instruction("unrelated_op_2", {}, {}, {}, comment));
            co_yield_(Instruction("v_or_b32", {or_v[0]}, {mfma_v[0], mfma_v[1]}, {}, comment));
            co_yield_(Instruction("unrelated_op_3", {}, {}, {}, comment));
            co_yield_(Instruction("v_or_b32", {or_v[1]}, {mfma_v[8], mfma_v[9]}, {}, comment));
            co_yield_(Instruction("v_mfma_f32_32x32x1f32",
                                  {mfma_v[8]},
                                  {mfma_v[9], mfma_v[10], mfma_v[11]},
                                  {},
                                  comment));
        }
    };
    auto generator_three = [&]() -> Generator<Instruction> {
        std::string comment = "stream3";
        while(true)
        {
            co_yield_(Instruction("unrelated_op_4", {}, {}, {}, comment));
            co_yield_(Instruction("v_mfma_f32_32x32x1f32",
                                  {mfma_v[12]},
                                  {mfma_v[13], mfma_v[14], mfma_v[15]},
                                  {},
                                  comment));
            co_yield_(Instruction("v_or_b32", {or_v[2]}, {mfma_v[4], mfma_v[5]}, {}, comment));
            co_yield_(Instruction("unrelated_op_5", {}, {}, {}, comment));
            co_yield_(Instruction("v_or_b32", {or_v[3]}, {mfma_v[12], mfma_v[13]}, {}, comment));
        }
    };

    std::vector<Generator<Instruction>> generators;
    generators.push_back(generator_one());
    generators.push_back(generator_two());
    generators.push_back(generator_three());

    auto scheduler = Component::GetNew<Scheduling::Scheduler>(
        Scheduling::SchedulerProcedure::Cooperative, Scheduling::CostFunction::MinNops, m_context);
    for(auto& inst : (*scheduler)(generators))
    {
        co_yield inst;
    }
}

CodeGenResult CodeGen(CodeGenProblem const& prob)
{
    Settings::getInstance()->set(Settings::AllowUnkownInstructions, true);

    CodeGenResult result(prob);
    Generator<rocRoller::Instruction> (*generator)(rocRoller::ContextPtr);

    {
        // TODO: implement codegen stress tests that use WMMAs
        auto        m_context = Context::ForDefaultHipDevice(prob.name);
        auto const& arch      = m_context->targetArchitecture();
        if(!arch.HasCapability(GPUCapability::HasMFMA))
        {
            std::cout << "FIXME: codegen stress tests are not supported on "
                      << arch.target().toString() << std::endl;
            exit(0);
        }
    }

    if(prob.instructions == "comments")
    {
        generator = comments;
    }
    else if(prob.instructions == "simple_mfma")
    {
        generator = simple_mfma;
    }
    else if(prob.instructions == "complex_mfma_with_coop")
    {
        generator = complex_mfma_with_coop;
    }
    else
    {
        AssertFatal(false, "Invalid instructions selection.");
    }

    auto Program = [&](ContextPtr m_context) -> Generator<Instruction> {
        for(size_t i = 0; i < prob.instCount; i++)
        {
            for(auto& inst : generator(m_context))
            {
                co_yield inst;
                ++i;
                if(i >= prob.instCount)
                {
                    break;
                }
            }
        }
    };

    for(size_t i = 0; i < prob.numWarmUp; i++)
    {
        auto m_context = Context::ForDefaultHipDevice(prob.name);
        m_context->schedule(Program(m_context));
    }

    for(size_t i = 0; i < prob.numRuns; i++)
    {
        auto  m_context = Context::ForDefaultHipDevice(prob.name);
        Timer timer("CodeGen");
        timer.tic();
        m_context->schedule(Program(m_context));
        timer.toc();
        result.kernelExecute.push_back(timer.nanoseconds() / prob.instCount);
    }

    result.kernelAssemble = 0;
    result.kernelGenerate = 0;

    return result;
}

int main(int argc, const char* argv[])
{
    std::string    filename;
    CodeGenProblem prob;

    prob.name         = "CodeGenv00";
    prob.instCount    = 40000;
    prob.instructions = "simple_mfma";
    prob.numWarmUp    = 2;
    prob.numRuns      = 10;

    CLI::App app{"CodeGen Driver: Stress test instruction generation."};
    app.option_defaults()->ignore_case();
    app.add_option("--inst_count", prob.instCount, "Number of instructions to generate.");
    app.add_option("--instructions", prob.instructions, "Label of instructions to use.");
    app.add_option("--num_warmup", prob.numWarmUp, "Number of warm-up runs.");
    app.add_option("--num_runs", prob.numRuns, "Number of timed runs.");
    app.add_option("--yaml", filename, "Results.");
    CLI11_PARSE(app, argc, argv);

    CodeGenResult result(prob);
    result = CodeGen(prob);

    if(!filename.empty())
    {
        std::ofstream file(filename);
        Serialization::writeYAML(file, result);
    }

    return 0;
}
