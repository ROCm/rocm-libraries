/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <migraphx/analyze_streams.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/context.hpp>
#include "test.hpp"
#include "basic_ops.hpp"

struct record_event
{
    std::size_t event = 0;
    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::pack(f(self.event, "event"));
    }
    std::string name() const { return "record_event"; }
    migraphx::shape compute_shape(const std::vector<migraphx::shape>&) const { return {}; }

    migraphx::argument compute(const migraphx::shape&, const std::vector<migraphx::argument>&) const
    {
        return {};
    }
};

struct wait_event
{
    std::size_t event = 0;
    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::pack(f(self.event, "event"));
    }
    std::string name() const { return "wait_event"; }
    migraphx::shape compute_shape(const std::vector<migraphx::shape>&) const { return {}; }

    migraphx::argument compute(const migraphx::shape&, const std::vector<migraphx::argument>&) const
    {
        return {};
    }
};

struct set_stream
{
    std::size_t stream = 0;
    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::pack(f(self.stream, "stream"));
    }
    std::string name() const { return "set_stream"; }
    migraphx::shape compute_shape(const std::vector<migraphx::shape>&) const { return {}; }

    migraphx::argument compute(const migraphx::shape&, const std::vector<migraphx::argument>&) const
    {
        return {};
    }
};

struct test_stream_model
{
    std::size_t max_stream = 0;
    std::unordered_map<migraphx::instruction_ref, std::size_t> ins2stream{};
    std::size_t get_nstream() const { return max_stream + 1; }
    std::size_t get_stream(migraphx::instruction_ref ins) const { return ins2stream.at(ins); }
    std::size_t get_event_id(migraphx::instruction_ref ins) const
    {
        auto v = ins->get_operator().to_value();
        return v["event"].to<std::size_t>();
    }
    bool has_stream(migraphx::instruction_ref ins) const { return ins2stream.count(ins) > 0; }
    bool is_record(migraphx::instruction_ref ins) const { return ins->name() == "record_event"; }
    bool is_wait(migraphx::instruction_ref ins) const { return ins->name() == "wait_event"; }
};

struct program_model
{
    migraphx::program p;
    migraphx::module* mm = p.get_main_module();
    std::unordered_map<migraphx::instruction_ref, std::size_t> ins2stream{};
    std::size_t max_stream = 0;

    template <class... Ts>
    migraphx::instruction_ref add_literal(Ts... xs)
    {
        return mm->add_literal(xs...);
    }

    template <class... Ts>
    migraphx::instruction_ref add_instruction(Ts... xs)
    {
        return mm->add_instruction(xs...);
    }

    template <class... Ts>
    migraphx::instruction_ref add_instruction_stream(std::size_t n, Ts... xs)
    {
        max_stream      = std::max(max_stream, n);
        auto ins        = mm->add_instruction(xs...);
        ins2stream[ins] = n;
        return ins;
    }

    template <class... Ts>
    migraphx::instruction_ref add_return(Ts... xs)
    {
        return mm->add_return({xs...});
    }

    template <class... Ts>
    migraphx::instruction_ref add_return_stream(std::size_t n, Ts... xs)
    {
        max_stream      = std::max(max_stream, n);
        auto ins        = mm->add_return({xs...});
        ins2stream[ins] = n;
        return ins;
    }

    test_stream_model get_stream_model() const { return {max_stream, ins2stream}; }

    std::vector<migraphx::stream_race> analyze() const
    {
        return migraphx::analyze_streams(*p.get_main_module(), get_stream_model());
    }

    void debug_print() const { p.debug_print(); }

    void debug_print(const std::vector<migraphx::stream_race>& races) const
    {
        for(auto&& race : races)
        {
            std::cout << "Race:\n";
            mm->debug_print(race.ins);
            mm->debug_print(race.before);
        }
    }
};

TEST_CASE(simple_race1)
{
    program_model pm;
    auto one   = pm.add_literal(1);
    auto pass1 = pm.add_instruction_stream(0, pass_op{}, one);
    auto pass2 = pm.add_instruction_stream(1, pass_op{}, one);
    auto pass3 = pm.add_instruction_stream(0, pass_op{}, pass1, pass2);
    auto races = pm.analyze();

    EXPECT(races.size() == 1);
    EXPECT(bool{races.front().ins == pass3});
    EXPECT(bool{races.front().before == pass2});
}

TEST_CASE(simple_race2)
{
    program_model pm;
    auto one    = pm.add_literal(1);
    auto pass1  = pm.add_instruction_stream(0, pass_op{}, one);
    auto pass2  = pm.add_instruction_stream(1, pass_op{}, one);
    auto pass21 = pm.add_instruction_stream(1, pass_op{}, pass2);
    auto pass3  = pm.add_instruction_stream(0, pass_op{}, pass1, pass21);
    auto races  = pm.analyze();

    EXPECT(races.size() == 1);
    EXPECT(bool{races.front().ins == pass3});
    EXPECT(bool{races.front().before == pass21});
}

TEST_CASE(simple_race3)
{
    program_model pm;
    auto one    = pm.add_literal(1);
    auto pass1  = pm.add_instruction_stream(0, pass_op{}, one);
    auto pass11 = pm.add_instruction_stream(0, pass_op{}, pass1);
    auto pass2  = pm.add_instruction_stream(1, pass_op{}, one);
    auto pass3  = pm.add_instruction_stream(0, pass_op{}, pass11, pass2);
    auto races  = pm.analyze();

    EXPECT(races.size() == 1);
    EXPECT(bool{races.front().ins == pass3});
    EXPECT(bool{races.front().before == pass2});
}

TEST_CASE(simple_race4)
{
    program_model pm;
    auto one    = pm.add_literal(1);
    auto pass1  = pm.add_instruction_stream(0, pass_op{}, one);
    auto pass11 = pm.add_instruction_stream(0, pass_op{}, pass1);
    auto pass2  = pm.add_instruction_stream(1, pass_op{}, one);
    auto pass21 = pm.add_instruction_stream(1, pass_op{}, pass2);
    auto pass3  = pm.add_instruction_stream(0, pass_op{}, pass11, pass21);
    auto races  = pm.analyze();

    EXPECT(races.size() == 1);
    EXPECT(bool{races.front().ins == pass3});
    EXPECT(bool{races.front().before == pass21});
}

TEST_CASE(simple_race5)
{
    program_model pm;
    auto one    = pm.add_literal(1);
    auto pass1  = pm.add_instruction_stream(0, pass_op{}, one);
    auto pass2  = pm.add_instruction_stream(1, pass_op{}, one);
    auto pass11 = pm.add_instruction_stream(0, pass_op{}, pass1);
    auto pass21 = pm.add_instruction_stream(1, pass_op{}, pass2);
    auto pass3  = pm.add_instruction_stream(0, pass_op{}, pass11, pass21);
    auto races  = pm.analyze();

    EXPECT(races.size() == 1);
    EXPECT(bool{races.front().ins == pass3});
    EXPECT(bool{races.front().before == pass21});
}

TEST_CASE(simple_race_record_wait_wrong_stream)
{
    program_model pm;
    auto one   = pm.add_literal(1);
    auto pass1 = pm.add_instruction_stream(0, pass_op{}, one);
    auto pass2 = pm.add_instruction_stream(1, pass_op{}, one);
    pm.add_instruction_stream(0, record_event{1});
    pm.add_instruction_stream(1, wait_event{1});
    auto pass3 = pm.add_instruction_stream(0, pass_op{}, pass1, pass2);
    auto races = pm.analyze();

    EXPECT(races.size() == 1);
    EXPECT(bool{races.front().ins == pass3});
    EXPECT(bool{races.front().before == pass2});
}

TEST_CASE(simple_race_record_wait_same_stream1)
{
    program_model pm;
    auto one   = pm.add_literal(1);
    auto pass1 = pm.add_instruction_stream(0, pass_op{}, one);
    auto pass2 = pm.add_instruction_stream(1, pass_op{}, one);
    pm.add_instruction_stream(1, record_event{1});
    pm.add_instruction_stream(1, wait_event{1});
    auto pass3 = pm.add_instruction_stream(0, pass_op{}, pass1, pass2);
    auto races = pm.analyze();

    EXPECT(races.size() == 1);
    EXPECT(bool{races.front().ins == pass3});
    EXPECT(bool{races.front().before == pass2});
}

TEST_CASE(simple_race_record_wait_same_stream2)
{
    program_model pm;
    auto one   = pm.add_literal(1);
    auto pass1 = pm.add_instruction_stream(0, pass_op{}, one);
    auto pass2 = pm.add_instruction_stream(1, pass_op{}, one);
    pm.add_instruction_stream(0, record_event{1});
    pm.add_instruction_stream(0, wait_event{1});
    auto pass3 = pm.add_instruction_stream(0, pass_op{}, pass1, pass2);
    auto races = pm.analyze();

    EXPECT(races.size() == 1);
    EXPECT(bool{races.front().ins == pass3});
    EXPECT(bool{races.front().before == pass2});
}

TEST_CASE(simple_race_sync)
{
    program_model pm;
    auto one   = pm.add_literal(1);
    auto pass1 = pm.add_instruction_stream(0, pass_op{}, one);
    auto pass2 = pm.add_instruction_stream(1, pass_op{}, one);
    pm.add_instruction_stream(1, record_event{1});
    pm.add_instruction_stream(0, wait_event{1});
    pm.add_instruction_stream(0, pass_op{}, pass1, pass2);
    auto races = pm.analyze();

    EXPECT(races.empty());
}

TEST_CASE(race_return)
{
    program_model pm;
    auto one   = pm.add_literal(1);
    auto pass1 = pm.add_instruction_stream(0, pass_op{}, one);
    auto pass2 = pm.add_instruction_stream(1, pass_op{}, one);
    auto r     = pm.add_return_stream(0, pass1, pass2);
    auto races = pm.analyze();

    EXPECT(races.size() == 1);
    EXPECT(bool{races.front().ins == r});
    EXPECT(bool{races.front().before == pass2});
}

TEST_CASE(race_return_sync)
{
    program_model pm;
    auto one   = pm.add_literal(1);
    auto pass1 = pm.add_instruction_stream(0, pass_op{}, one);
    auto pass2 = pm.add_instruction_stream(1, pass_op{}, one);
    pm.add_instruction_stream(1, record_event{1});
    pm.add_instruction_stream(0, wait_event{1});
    pm.add_return_stream(0, pass1, pass2);
    auto races = pm.analyze();

    EXPECT(races.empty());
}

TEST_CASE(race_double_wait1)
{
    program_model pm;
    auto one   = pm.add_literal(1);
    auto pass1 = pm.add_instruction_stream(0, pass_op{}, one);
    auto pass2 = pm.add_instruction_stream(1, pass_op{}, one);
    pm.add_instruction_stream(1, record_event{1});
    auto pass3 = pm.add_instruction_stream(2, pass_op{}, one);
    pm.add_instruction_stream(2, wait_event{1});
    auto pass4 = pm.add_instruction_stream(2, pass_op{}, pass3, pass2);
    pm.add_instruction_stream(2, record_event{2});
    auto pass5 = pm.add_instruction_stream(0, pass_op{}, pass1, pass2);
    pm.add_instruction_stream(0, record_event{3});
    pm.add_instruction_stream(1, wait_event{3});
    pm.add_instruction_stream(1, wait_event{2});
    pm.add_instruction_stream(1, pass_op{}, pass4, pass5);
    auto races = pm.analyze();

    EXPECT(races.size() == 1);
    EXPECT(bool{races.front().ins == pass5});
    EXPECT(bool{races.front().before == pass2});
}

TEST_CASE(race_double_wait2)
{
    program_model pm;
    auto one   = pm.add_literal(1);
    auto pass1 = pm.add_instruction_stream(0, pass_op{}, one);
    auto pass2 = pm.add_instruction_stream(1, pass_op{}, one);
    pm.add_instruction_stream(1, record_event{1});
    auto pass3 = pm.add_instruction_stream(2, pass_op{}, one);
    auto pass4 = pm.add_instruction_stream(2, pass_op{}, pass3, pass2);
    pm.add_instruction_stream(2, record_event{2});
    pm.add_instruction_stream(0, wait_event{1});
    auto pass5 = pm.add_instruction_stream(0, pass_op{}, pass1, pass2);
    pm.add_instruction_stream(0, record_event{3});
    pm.add_instruction_stream(1, wait_event{3});
    pm.add_instruction_stream(1, wait_event{2});
    pm.add_instruction_stream(1, pass_op{}, pass4, pass5);
    auto races = pm.analyze();

    EXPECT(races.size() == 1);
    EXPECT(bool{races.front().ins == pass4});
    EXPECT(bool{races.front().before == pass2});
}

TEST_CASE(race_double_wait3)
{
    program_model pm;
    auto one   = pm.add_literal(1);
    auto pass1 = pm.add_instruction_stream(0, pass_op{}, one);
    auto pass2 = pm.add_instruction_stream(1, pass_op{}, one);
    pm.add_instruction_stream(1, record_event{1});
    auto pass3 = pm.add_instruction_stream(2, pass_op{}, one);
    pm.add_instruction_stream(2, wait_event{1});
    auto pass4 = pm.add_instruction_stream(2, pass_op{}, pass3, pass2);
    pm.add_instruction_stream(2, record_event{2});
    pm.add_instruction_stream(0, wait_event{1});
    auto pass5 = pm.add_instruction_stream(0, pass_op{}, pass1, pass2);
    pm.add_instruction_stream(1, wait_event{2});
    auto pass6 = pm.add_instruction_stream(1, pass_op{}, pass4, pass5);
    auto races = pm.analyze();

    EXPECT(races.size() == 1);
    EXPECT(bool{races.front().ins == pass6});
    EXPECT(bool{races.front().before == pass5});
}

TEST_CASE(race_double_wait4)
{
    program_model pm;
    auto one   = pm.add_literal(1);
    auto pass1 = pm.add_instruction_stream(0, pass_op{}, one);
    auto pass2 = pm.add_instruction_stream(1, pass_op{}, one);
    pm.add_instruction_stream(1, record_event{1});
    auto pass3 = pm.add_instruction_stream(2, pass_op{}, one);
    pm.add_instruction_stream(2, wait_event{1});
    auto pass4 = pm.add_instruction_stream(2, pass_op{}, pass3, pass2);
    pm.add_instruction_stream(2, record_event{2});
    pm.add_instruction_stream(0, wait_event{1});
    auto pass5 = pm.add_instruction_stream(0, pass_op{}, pass1, pass2);
    pm.add_instruction_stream(0, record_event{3});
    pm.add_instruction_stream(1, wait_event{3});
    auto pass6 = pm.add_instruction_stream(1, pass_op{}, pass4, pass5);
    auto races = pm.analyze();

    EXPECT(races.size() == 1);
    EXPECT(bool{races.front().ins == pass6});
    EXPECT(bool{races.front().before == pass4});
}

TEST_CASE(race_double_wait_sync)
{
    program_model pm;
    auto one   = pm.add_literal(1);
    auto pass1 = pm.add_instruction_stream(0, pass_op{}, one);
    auto pass2 = pm.add_instruction_stream(1, pass_op{}, one);
    pm.add_instruction_stream(1, record_event{1});
    auto pass3 = pm.add_instruction_stream(2, pass_op{}, one);
    pm.add_instruction_stream(2, wait_event{1});
    auto pass4 = pm.add_instruction_stream(2, pass_op{}, pass3, pass2);
    pm.add_instruction_stream(2, record_event{2});
    pm.add_instruction_stream(0, wait_event{1});
    auto pass5 = pm.add_instruction_stream(0, pass_op{}, pass1, pass2);
    pm.add_instruction_stream(0, record_event{3});
    pm.add_instruction_stream(1, wait_event{3});
    pm.add_instruction_stream(1, wait_event{2});
    pm.add_instruction_stream(1, pass_op{}, pass4, pass5);
    auto races = pm.analyze();

    EXPECT(races.empty());
}

TEST_CASE(race_multi_wait1)
{
    program_model pm;
    auto one   = pm.add_literal(1);
    auto pass1 = pm.add_instruction_stream(0, pass_op{}, one);
    pm.add_instruction_stream(0, record_event{5});
    auto pass2 = pm.add_instruction_stream(1, pass_op{}, one);
    pm.add_instruction_stream(1, record_event{1});
    pm.add_instruction_stream(2, wait_event{1});
    auto pass3 = pm.add_instruction_stream(2, pass_op{}, one, pass2);
    pm.add_instruction_stream(2, record_event{2});
    pm.add_instruction_stream(3, wait_event{5});
    auto pass4 = pm.add_instruction_stream(3, pass_op{}, one, pass1);
    pm.add_instruction_stream(3, record_event{3});
    pm.add_instruction_stream(0, wait_event{2});
    auto pass5 = pm.add_instruction_stream(0, pass_op{}, pass3, pass1);
    pm.add_instruction_stream(0, record_event{4});
    pm.add_instruction_stream(1, wait_event{3});
    auto pass6 = pm.add_instruction_stream(1, pass_op{}, pass4, pass5);

    auto races = pm.analyze();

    EXPECT(races.size() == 1);
    EXPECT(bool{races.front().ins == pass6});
    EXPECT(bool{races.front().before == pass5});
}

TEST_CASE(race_multi_wait2)
{
    program_model pm;
    auto one   = pm.add_literal(1);
    auto pass1 = pm.add_instruction_stream(0, pass_op{}, one);
    pm.add_instruction_stream(0, record_event{5});
    auto pass2 = pm.add_instruction_stream(1, pass_op{}, one);
    pm.add_instruction_stream(1, record_event{1});
    pm.add_instruction_stream(2, wait_event{1});
    auto pass3 = pm.add_instruction_stream(2, pass_op{}, one, pass2);
    pm.add_instruction_stream(2, record_event{2});
    pm.add_instruction_stream(3, wait_event{5});
    auto pass4 = pm.add_instruction_stream(3, pass_op{}, one, pass1);
    pm.add_instruction_stream(3, record_event{3});
    pm.add_instruction_stream(0, wait_event{2});
    auto pass5 = pm.add_instruction_stream(0, pass_op{}, pass3, pass1);
    pm.add_instruction_stream(0, record_event{4});
    pm.add_instruction_stream(1, wait_event{4});
    auto pass6 = pm.add_instruction_stream(1, pass_op{}, pass4, pass5);

    auto races = pm.analyze();

    EXPECT(races.size() == 1);
    EXPECT(bool{races.front().ins == pass6});
    EXPECT(bool{races.front().before == pass4});
}

TEST_CASE(race_multi_wait3)
{
    program_model pm;
    auto one   = pm.add_literal(1);
    auto pass1 = pm.add_instruction_stream(0, pass_op{}, one);
    auto pass2 = pm.add_instruction_stream(1, pass_op{}, one);
    pm.add_instruction_stream(1, record_event{1});
    pm.add_instruction_stream(2, wait_event{1});
    auto pass3 = pm.add_instruction_stream(2, pass_op{}, one, pass2);
    pm.add_instruction_stream(2, record_event{2});
    auto pass4 = pm.add_instruction_stream(3, pass_op{}, one, pass1);
    pm.add_instruction_stream(3, record_event{3});
    pm.add_instruction_stream(0, wait_event{2});
    auto pass5 = pm.add_instruction_stream(0, pass_op{}, pass3, pass1);
    pm.add_instruction_stream(0, record_event{4});
    pm.add_instruction_stream(1, wait_event{3});
    pm.add_instruction_stream(1, wait_event{4});
    pm.add_instruction_stream(1, pass_op{}, pass4, pass5);

    auto races = pm.analyze();

    EXPECT(races.size() == 1);
    EXPECT(bool{races.front().ins == pass4});
    EXPECT(bool{races.front().before == pass1});
}

TEST_CASE(race_multi_wait_sync)
{
    program_model pm;
    auto one   = pm.add_literal(1);
    auto pass1 = pm.add_instruction_stream(0, pass_op{}, one);
    pm.add_instruction_stream(0, record_event{5});
    auto pass2 = pm.add_instruction_stream(1, pass_op{}, one);
    pm.add_instruction_stream(1, record_event{1});
    pm.add_instruction_stream(2, wait_event{1});
    auto pass3 = pm.add_instruction_stream(2, pass_op{}, one, pass2);
    pm.add_instruction_stream(2, record_event{2});
    pm.add_instruction_stream(3, wait_event{5});
    auto pass4 = pm.add_instruction_stream(3, pass_op{}, one, pass1);
    pm.add_instruction_stream(3, record_event{3});
    pm.add_instruction_stream(0, wait_event{2});
    auto pass5 = pm.add_instruction_stream(0, pass_op{}, pass3, pass1);
    pm.add_instruction_stream(0, record_event{4});
    pm.add_instruction_stream(1, wait_event{3});
    pm.add_instruction_stream(1, wait_event{4});
    pm.add_instruction_stream(1, pass_op{}, pass4, pass5);

    auto races = pm.analyze();

    EXPECT(races.empty());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
