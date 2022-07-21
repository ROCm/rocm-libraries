
#include <cassert>
#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <string>

#include <rocRoller/Generator.hpp>
#include <rocRoller/Instruction.hpp>
#include <rocRoller/Scheduling.hpp>

#if 0
#define ITER_COUNT 1000000
#else
#define ITER_COUNT 1
#endif

#define DECLARE_EMPLACE()                                                                     \
    double time_emplace(std::string filename)                                                 \
    {                                                                                         \
        auto start = std::chrono::steady_clock::now();                                        \
        auto ctx   = filename.empty() ? std::make_shared<rocRoller::Context>()                \
                                      : std::make_shared<rocRoller::Context>();               \
                                                                                              \
        for(size_t i = 0; i < ITER_COUNT; i++)                                                \
        {                                                                                     \
            for(auto& inst : sequenceA(ctx))                                                  \
                ctx->schedule(inst);                                                          \
                                                                                              \
            for(auto& inst : sequenceB(ctx))                                                  \
                ctx->schedule(inst);                                                          \
                                                                                              \
            for(auto& inst : sequenceC(ctx))                                                  \
                ctx->schedule(inst);                                                          \
                                                                                              \
            for(auto& inst : sequenceD(ctx))                                                  \
                ctx->schedule(inst);                                                          \
        }                                                                                     \
                                                                                              \
        auto                          end             = std::chrono::steady_clock::now();     \
        std::chrono::duration<double> elapsed_seconds = end - start;                          \
                                                                                              \
        std::cout << "Emplace: " << ctx->pc << " instructions in " << elapsed_seconds.count() \
                  << std::endl;                                                               \
                                                                                              \
        return elapsed_seconds.count();                                                       \
    }

#define DECLARE_INSERT()                                                                           \
    double time_insert(filename)                                                                   \
    {                                                                                              \
        auto start = std::chrono::steady_clock::now();                                             \
        auto ctx   = std::make_shared<Context>();                                                  \
                                                                                                   \
        std::vector<Instruction> result;                                                           \
                                                                                                   \
        for(size_t i = 0; i < ITER_COUNT; i++)                                                     \
        {                                                                                          \
            {                                                                                      \
                auto instructions = sequenceA(ctx);                                                \
                result.insert(result.end(), instructions.begin(), instructions.end());             \
            }                                                                                      \
            {                                                                                      \
                auto instructions = sequenceB(ctx);                                                \
                result.insert(result.end(), instructions.begin(), instructions.end());             \
            }                                                                                      \
            {                                                                                      \
                auto instructions = sequenceC(ctx);                                                \
                result.insert(result.end(), instructions.begin(), instructions.end());             \
            }                                                                                      \
            {                                                                                      \
                auto instructions = sequenceD(ctx);                                                \
                result.insert(result.end(), instructions.begin(), instructions.end());             \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        auto                          end             = std::chrono::steady_clock::now();          \
        std::chrono::duration<double> elapsed_seconds = end - start;                               \
                                                                                                   \
        std::cout << "Insert: " << result.size() << " instructions in " << elapsed_seconds.count() \
                  << std::endl;                                                                    \
                                                                                                   \
        return elapsed_seconds.count();                                                            \
    }

namespace co
{
    using namespace rocRoller;

    InstructionGenerator sequenceA(std::shared_ptr<rocRoller::Context> ctx)
    {
        //co_yield Instruction::Allocate(ctx, "A");
        co_yield Instruction::Normal(ctx, "v_add_u32 v0, v1, v2");
        co_yield Instruction::Normal(ctx, "v_add_u32 v0, v0, v4", 1);
        co_yield Instruction::Normal(ctx, "v_add_u32 v0, v0, v6", 1);
        co_yield Instruction::Normal(ctx, "v_add_u32 v0, v0, v7", 1);
    }

    InstructionGenerator sequenceB(std::shared_ptr<rocRoller::Context> ctx)
    {
#if 1
        co_yield Instruction::Allocate(ctx, "B");
        co_yield Instruction::Normal(ctx, "v_add_u32 v9, v10, v11");
        co_yield Instruction::Normal(ctx, "v_add_u32 v9, v9, v12", 4);
        co_yield Instruction::Normal(ctx, "v_add_u32 v9, v9, v13", 1);
#else
        std::vector<Instruction> vec;
        vec.emplace_back(Instruction::Allocate(ctx, "B"));
        vec.emplace_back(Instruction::Normal(ctx, "v_add_u32 v9, v10, v11"));
        vec.emplace_back(Instruction::Normal(ctx, "v_add_u32 v9, v9, v12", 4));
        vec.emplace_back(Instruction::Normal(ctx, "v_add_u32 v9, v9, v13", 1));

        co_yield vec;
#endif
    }

    InstructionGenerator schedule_sequential(std::vector<InstructionGenerator>& seqs)
    {
        for(auto& seq : seqs)
        {
            co_yield seq;
        }
    }

    // If the next instruction would stall, find the least-stalling subsequence.
    // Otherwise, continue with the current stream.
    InstructionGenerator schedule_cooperative(std::vector<InstructionGenerator>& seqs)
    {
        std::vector<InstructionGenerator::Iter> iterators;

        iterators.reserve(seqs.size());
        for(auto& s : seqs)
            iterators.emplace_back(s.begin());

        size_t idx = 0;

        while(true)
        {
            if(iterators[idx] == seqs[idx].end() || iterators[idx]->stallCycles() > 0)
            {
                size_t origIdx     = idx;
                int    minStall    = std::numeric_limits<int>::max();
                int    minStallIdx = -1;
                if(iterators[idx] != seqs[idx].end())
                {
                    //std::cout << "... trying to eliminate stall" << std::endl;
                    minStall    = iterators[idx]->stallCycles();
                    minStallIdx = idx;
                }

                idx = (idx + 1) % seqs.size();

                while(idx != origIdx)
                {
                    if(iterators[idx] != seqs[idx].end()
                       && iterators[idx]->stallCycles() < minStall)
                    {
                        minStall    = iterators[idx]->stallCycles();
                        minStallIdx = idx;
                    }

                    if(minStall == 0)
                        break;

                    idx = (idx + 1) % seqs.size();
                }

                if(minStallIdx == -1)
                    break;

                idx = minStallIdx;
            }

            co_yield *iterators[idx];
            ++iterators[idx];
        }
    }

    // Take the least stalling instruction from the first sequence available.
    InstructionGenerator schedule_priority(std::vector<InstructionGenerator>& seqs)
    {
        std::vector<InstructionGenerator::Iter> iterators;

        iterators.reserve(seqs.size());
        for(auto& s : seqs)
            iterators.emplace_back(s.begin());

        bool any = true;

        while(any)
        {
            any = false;

            int minStall    = std::numeric_limits<int>::max();
            int minStallIdx = -1;

            for(size_t idx = 0; idx < seqs.size(); idx++)
            {
                if(iterators[idx] == seqs[idx].end())
                    continue;

                any = true;

                int myStall = iterators[idx]->stallCycles();

                if(myStall < minStall)
                {
                    minStall    = myStall;
                    minStallIdx = idx;
                }

                if(myStall == 0)
                    break;
            }

            if(any)
            {
                co_yield *iterators[minStallIdx];
                ++iterators[minStallIdx];
            }
        }
    }

    InstructionGenerator sequenceC(std::shared_ptr<rocRoller::Context> ctx)
    {
        {
            std::vector<InstructionGenerator> seqs;
            seqs.emplace_back(sequenceA(ctx));
            seqs.emplace_back(sequenceB(ctx));
            seqs.emplace_back(sequenceB(ctx));

            auto subgen = schedule_cooperative(seqs);
            co_yield subgen;
        }

        co_yield Instruction::Normal(ctx, "v_add_u32 v25, v0, v9");

        auto sg = sequenceA(ctx);
        co_yield sg;
    }

    InstructionGenerator sequenceD(std::shared_ptr<rocRoller::Context> ctx)
    {
        std::vector<InstructionGenerator> seqs;
        seqs.emplace_back(sequenceA(ctx));
        seqs.emplace_back(sequenceB(ctx));
        seqs.emplace_back(sequenceC(ctx));

        auto sg = schedule_priority(seqs);
        co_yield sg;
    }

#if 1
    DECLARE_EMPLACE()
#else
    double time_emplace()
    {
        auto start = std::chrono::steady_clock::now();
        auto ctx = std::make_shared<Context>();

        std::vector<Instruction> result;

        for(size_t i = 0; i < ITER_COUNT; i++)
        {
            //for(auto const& inst: sequenceA(ctx))
            //    result.emplace_back(inst);

            //for(auto const& inst: sequenceB(ctx))
            //    result.emplace_back(inst);

            //for(auto const& inst: sequenceC(ctx))
            //    result.emplace_back(inst);

            for(auto const& inst : sequenceD(ctx))
                result.emplace_back(inst);

            //auto gen = sequenceD(ctx);
            //for(auto iter = gen.begin(); iter != gen.end(); ++iter)
            //{
            //    auto inst = *iter;
            //    result.emplace_back(inst);
            //}
        }

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;

        std::cout << "emplace: " << elapsed_seconds.count() << std::endl;

        return elapsed_seconds.count();
    }
#endif

}

namespace noco
{
    using namespace rocRoller;

    std::vector<Instruction> sequenceA(std::shared_ptr<rocRoller::Context> ctx)
    {
        std::vector<Instruction> rv;
        rv.emplace_back(Instruction::Allocate(ctx, "A"));
        rv.emplace_back(Instruction::Normal(ctx, "v_add_u32 v0, v1, v2"));
        rv.emplace_back(Instruction::Normal(ctx, "v_add_u32 v0, v0, v4", 1));
        rv.emplace_back(Instruction::Normal(ctx, "v_add_u32 v0, v0, v6", 1));
        rv.emplace_back(Instruction::Normal(ctx, "v_add_u32 v0, v0, v7", 1));
        return std::move(rv);
    }

    std::vector<Instruction> sequenceB(std::shared_ptr<rocRoller::Context> ctx)
    {
        std::vector<Instruction> rv;
        rv.emplace_back(Instruction::Allocate(ctx, "B"));
        rv.emplace_back(Instruction::Normal(ctx, "v_add_u32 v9, v10, v11"));
        rv.emplace_back(Instruction::Normal(ctx, "v_add_u32 v9, v9, v12", 4));
        rv.emplace_back(Instruction::Normal(ctx, "v_add_u32 v9, v9, v13", 1));
        return std::move(rv);
    }

    std::vector<Instruction> schedule_sequential(std::vector<std::vector<Instruction>>& seqs)
    {
        std::vector<Instruction> rv;

        for(auto& seq : seqs)
        {
            rv.insert(rv.end(), seq.begin(), seq.end());
        }

        return std::move(rv);
    }

    // If the next instruction would stall, find the least-stalling subsequence.
    // Otherwise, continue with the current stream.
    std::vector<Instruction> schedule_cooperative(std::vector<std::vector<Instruction>>& seqs)
    {
        std::vector<Instruction> rv;

        std::vector<std::vector<Instruction>::iterator> iterators;

        iterators.reserve(seqs.size());
        for(auto& s : seqs)
            iterators.emplace_back(s.begin());

        size_t idx = 0;

        while(true)
        {
            if(iterators[idx] == seqs[idx].end() || iterators[idx]->stallCycles() > 0)
            {
                size_t origIdx     = idx;
                int    minStall    = std::numeric_limits<int>::max();
                int    minStallIdx = -1;
                if(iterators[idx] != seqs[idx].end())
                {
                    //std::cout << "... trying to eliminate stall" << std::endl;
                    minStall    = iterators[idx]->stallCycles();
                    minStallIdx = idx;
                }

                idx = (idx + 1) % seqs.size();

                while(idx != origIdx)
                {
                    if(iterators[idx] != seqs[idx].end()
                       && iterators[idx]->stallCycles() < minStall)
                    {
                        minStall    = iterators[idx]->stallCycles();
                        minStallIdx = idx;
                    }

                    if(minStall == 0)
                        break;

                    idx = (idx + 1) % seqs.size();
                }

                if(minStallIdx == -1)
                    break;

                idx = minStallIdx;
            }

            rv.emplace_back(*iterators[idx]);
            ++iterators[idx];
        }
        return std::move(rv);
    }

    // Take the least stalling instruction from the first sequence available.
    std::vector<Instruction> schedule_priority(std::vector<std::vector<Instruction>>& seqs)
    {
        std::vector<Instruction>                        rv;
        std::vector<std::vector<Instruction>::iterator> iterators;

        iterators.reserve(seqs.size());
        for(auto& s : seqs)
            iterators.emplace_back(s.begin());

        bool any = true;

        while(any)
        {
            any = false;

            int minStall    = std::numeric_limits<int>::max();
            int minStallIdx = -1;

            for(size_t idx = 0; idx < seqs.size(); idx++)
            {
                if(iterators[idx] == seqs[idx].end())
                    continue;

                any = true;

                int myStall = iterators[idx]->stallCycles();

                if(myStall < minStall)
                {
                    minStall    = myStall;
                    minStallIdx = idx;
                }

                if(myStall == 0)
                    break;
            }

            if(any)
            {
                rv.emplace_back(*iterators[minStallIdx]);
                ++iterators[minStallIdx];
            }
        }
        return std::move(rv);
    }

    std::vector<Instruction> sequenceC(std::shared_ptr<rocRoller::Context> ctx)
    {
        std::vector<Instruction> rv;
        {
            std::vector<std::vector<Instruction>> seqs;
            seqs.reserve(3);
            seqs.emplace_back(sequenceA(ctx));
            seqs.emplace_back(sequenceB(ctx));
            seqs.emplace_back(sequenceB(ctx));

            auto subseq = schedule_cooperative(seqs);
            rv.insert(rv.end(), subseq.begin(), subseq.end());
        }

        rv.emplace_back(Instruction::Normal(ctx, "v_add_u32 v25, v0, v9"));

        for(auto& insn : sequenceA(ctx))
            rv.emplace_back(insn);

        return std::move(rv);
    }

    std::vector<Instruction> sequenceD(std::shared_ptr<rocRoller::Context> ctx)
    {
        std::vector<std::vector<Instruction>> seqs;
        seqs.emplace_back(sequenceA(ctx));
        seqs.emplace_back(sequenceB(ctx));
        seqs.emplace_back(sequenceC(ctx));

        return std::move(schedule_priority(seqs));
    }

    DECLARE_EMPLACE()
    //DECLARE_INSERT()

}

int main(int argc, const char* argv[])
{
    {
        //auto fibs = rocRoller::fibonacci<int>();
        //for(auto val: std::ranges::views::take{fibs, 12})
        //    std::cout << val << std::endl;

        std::vector<int> asdf(20);
        for(auto val : std::ranges::views::take(asdf, 12))
            std::cout << val << std::endl;
    }
    return 0;
    for(int i = 0; i < 100; i++)
    {
        //std::cout << "Direct ";
        //noco::time_insert("direct.txt");

        std::cout << "Coroutine ";
        //co::time_emplace("/dev/shm/coroutine.txt");
        co::time_emplace("");

        std::cout << "Direct ";
        noco::time_emplace("");
    }

    return 0;

#if 0
    std::cout << "Sequential" << std::endl;
    std::cout << "-=-=-=-=-=-=-=-=-=-=" << std::endl;

    {
        auto ctx = std::make_shared<Context>();
        std::vector<InstructionGenerator> seqs;
        seqs.emplace_back(sequenceA(ctx));
        seqs.emplace_back(sequenceA(ctx));
        seqs.emplace_back(sequenceB(ctx));

        for(auto & insn: schedule_sequential(seqs))
            ctx->schedule(insn);

        std::cout << "Stalls: " << ctx->total_stalls << std::endl;
    }

    std::cout << std::endl << "Cooperative" << std::endl;
    std::cout << "-=-=-=-=-=-=-=-=-=-=" << std::endl;

    {
        auto ctx = std::make_shared<Context>();
        std::vector<InstructionGenerator> seqs;
        seqs.emplace_back(sequenceA(ctx));
        seqs.emplace_back(sequenceA(ctx));
        seqs.emplace_back(sequenceB(ctx));

        for(auto & insn: schedule_cooperative(seqs))
            ctx->schedule(insn);

        std::cout << "Stalls: " << ctx->total_stalls << std::endl;
    }

    std::cout << std::endl << "Priority" << std::endl;
    std::cout << "-=-=-=-=-=-=-=-=-=-=" << std::endl;

    {
        auto ctx = std::make_shared<Context>();
        std::vector<InstructionGenerator> seqs;
        seqs.emplace_back(sequenceA(ctx));
        seqs.emplace_back(sequenceA(ctx));
        seqs.emplace_back(sequenceB(ctx));

        for(auto & insn: schedule_priority(seqs))
            ctx->schedule(insn);

        std::cout << "Stalls: " << ctx->total_stalls << std::endl;
    }

    std::cout << std::endl << "Complex" << std::endl;
    std::cout << "-=-=-=-=-=-=-=-=-=-=" << std::endl;

    {
        auto ctx = std::make_shared<Context>();

        for(auto & insn: sequenceC(ctx))
            ctx->schedule(insn);

        std::cout << "Stalls: " << ctx->total_stalls << std::endl;
    }

    return 0;
#endif
}
