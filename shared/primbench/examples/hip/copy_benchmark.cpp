#include "primbench.hpp"

// All benchmarked types must be declared
// This allows you to for example format `long long` as `int64_t`, or `i64`
PRIMBENCH_REGISTER_TYPE(char, "char")
PRIMBENCH_REGISTER_TYPE(long long, "long long")

// Simple copy kernel
template<typename T, unsigned int BlockSize, unsigned int ItemsPerThread>
__global__ __launch_bounds__(BlockSize)
void copy_kernel(const T* input, T* output)
{
    unsigned int idx = threadIdx.x + blockIdx.x * BlockSize * ItemsPerThread;
#pragma unroll
    for(unsigned int i = 0; i < ItemsPerThread; ++i)
        output[idx + i * BlockSize] = input[idx + i * BlockSize];
}

template<typename T>
struct copy_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        // This meta() method must return an `algo` key
        // The `algo` name must be identical across all queued specializations
        // primbench::json objects can be nested
        return primbench::json{}.add("algo", "copy").add("type", primbench::name<T>());
    }

    // This contains both the setup and running of the benchmark
    void run(primbench::state& state) override
    {
        const auto& stream = state.stream;
        const auto& bytes  = state.size;

        // primbench::log() calls report progress in gray
        // They help with discovering slow setup steps
        primbench::log("Calculating items");
        size_t                 N              = bytes / sizeof(T);
        constexpr unsigned int BlockSize      = 256;
        constexpr unsigned int ItemsPerThread = 4;

        const size_t items_per_block = BlockSize * ItemsPerThread;
        const size_t items = items_per_block * ((N + items_per_block - 1) / items_per_block);

        primbench::log("Generating input data");
        std::vector<T> h_input(items);
        for(size_t i = 0; i < items; ++i)
            h_input[i] = T(i);

        primbench::log("Allocating output vector");
        std::vector<T> h_output(items);

        primbench::log("Allocating device memory");
        T* d_input;
        T* d_output;
        PRIMBENCH_CHECK(hipMalloc(&d_input, items * sizeof(T)));
        PRIMBENCH_CHECK(hipMalloc(&d_output, items * sizeof(T)));

        primbench::log("Copying to device");
        PRIMBENCH_CHECK(hipMemcpyAsync(d_input,
                                       h_input.data(),
                                       items * sizeof(T),
                                       hipMemcpyHostToDevice,
                                       stream));

        dim3 grid(items / items_per_block);
        dim3 block(BlockSize);

        // primbench uses this to calculates the items/sec and bytes/sec
        state.set_items(items);
        state.add_reads<T>(items);
        state.add_writes<T>(items);

        // This passes a lambda to primbench, which calls it many times
        // primbench completely handles synchronization
        state.run(
            [&] {
                copy_kernel<T, BlockSize, ItemsPerThread>
                    <<<grid, block, 0, stream>>>(d_input, d_output);
            });

        PRIMBENCH_CHECK(hipFree(d_input));
        PRIMBENCH_CHECK(hipFree(d_output));
    }
};

int main(int argc, char* argv[])
{
    primbench::executor executor(argc, argv);

    executor.queue<copy_benchmark<char>>();
    executor.queue<copy_benchmark<long long>>();

    executor.run();
}
