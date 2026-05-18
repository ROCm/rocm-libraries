#include "run_swiglu_example.hpp"

#include <cstdlib>

[[nodiscard]]
inline auto create_arg_parser() -> ck_tile::ArgParser
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "3840", "m dimension")
        .insert("n", "4096", "n dimension")
        .insert("k", "2048", "k dimension")
        .insert("a_layout", "R", "A tensor data layout - Row by default")
        .insert("b_layout", "C", "B tensor data layout - Column by default")
        .insert("c_layout", "R", "C tensor data layout - Row by default")
        .insert("stride_a", "0", "Tensor A stride")
        .insert("stride_b", "0", "Tensor B stride")
        .insert("stride_c", "0", "Tensor C stride")
        .insert("v", "1", "0. No validation, 1. Validation on CPU, 2. Validation on GPU")
        .insert("prec", "fp16", "data type. fp16/bf16/fp8/bf8/pk_int4/tf32 (tf32 only on gfx950)")
        .insert("warmup", "50", "number of iterations before benchmark the kernel")
        .insert("repeat", "100", "number of iterations to benchmark the kernel")
        .insert("timer", "gpu", "gpu:gpu timer, cpu:cpu timer")
        .insert("split_k", "1", "splitK value")
        .insert("init", "0", "0:random, 1:linear, 2:constant(1)")
        .insert("persistent", "0", "0:non-persistent, 1:persistent")
        .insert("pipeline", "v3", "")
        .insert("json", "0", "0: No Json, 1: Dump Results in Json format")
        .insert("jsonfile", "gemm.json", "json file name to dump results")
        .insert("flush_cache", "true", "flush cache before running the kernel, defaults to true")
        .insert("rotating_count", "1000", "rotating count, defaults to 1000")
        .insert("test_async", "0", "0: normal gemm, 1: test async input scheduler");
    return arg_parser;
}

int main(int argc, char* argv[])
{
    auto arg_parser = create_arg_parser();
    if(!arg_parser.parse(argc, argv))
        return -1;

    try
    {
        return ck_tile::swiglu_example::run_swiglu_example(arg_parser);
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "Caught runtime error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
