/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2021-2024, Advanced Micro Devices, Inc.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#ifndef EXAMPLES_EXAMPLE_UTILS_HPP
#define EXAMPLES_EXAMPLE_UTILS_HPP
#include "mersenne.h"
#include <vector>
#include <sstream>
#include <iostream>

#include <hipcub/util_type.hpp>
#include <hipcub/util_allocator.hpp>
#include <hipcub/iterator/discard_output_iterator.hpp>

#define AssertEquals(a, b) if ((a) != (b)) { std::cerr << "\n(" << __FILE__ << ": " << __LINE__ << ")\n"; exit(1);}

#define HIP_CHECK(condition)                                                           \
    do                                                                                 \
    {                                                                                  \
        hipError_t error = condition;                                                  \
        if(error != hipSuccess)                                                        \
        {                                                                              \
            std::cout << "HIP error: " << error << " line: " << __LINE__ << std::endl; \
            exit(error);                                                               \
        }                                                                              \
    }                                                                                  \
    while(0);

template <typename T>
T CoutCast(T val) { return val; }

int CoutCast(char val) { return val; }

int CoutCast(unsigned char val) { return val; }

int CoutCast(signed char val) { return val; }
/******************************************************************************
 * Command-line parsing functionality
 ******************************************************************************/

/**
 * Utility for parsing command line arguments
 */
struct CommandLineArgs
{

    std::vector<std::string>    keys;
    std::vector<std::string>    values;
    std::vector<std::string>    args;
    hipDeviceProp_t             deviceProp;
    float                       device_giga_bandwidth;
    std::size_t                 device_free_physmem;
    std::size_t                 device_total_physmem;

    /**
     * Constructor
     */
    CommandLineArgs(int argc, char **argv) :
        keys(10),
        values(10)
    {
        using namespace std;

        // Initialize mersenne generator
        unsigned int mersenne_init[4]=  {0x123, 0x234, 0x345, 0x456};
        mersenne::init_by_array(mersenne_init, 4);

        for (int i = 1; i < argc; i++)
        {
            string arg = argv[i];

            if ((arg[0] != '-') || (arg[1] != '-'))
            {
                args.push_back(arg);
                continue;
            }

            string::size_type pos;
            string key, val;
            if ((pos = arg.find('=')) == string::npos) {
                key = string(arg, 2, arg.length() - 2);
                val = "";
            } else {
                key = string(arg, 2, pos - 2);
                val = string(arg, pos + 1, arg.length() - 1);
            }

            keys.push_back(key);
            values.push_back(val);
        }
    }


    /**
     * Checks whether a flag "--<flag>" is present in the commandline
     */
    bool CheckCmdLineFlag(const char* arg_name)
    {
        using namespace std;

        for (std::size_t i = 0; i < keys.size(); ++i)
        {
            if (keys[i] == string(arg_name))
                return true;
        }
        return false;
    }


    /**
     * Returns number of naked (non-flag and non-key-value) commandline parameters
     */
    template <typename T>
    int NumNakedArgs()
    {
        return args.size();
    }


    /**
     * Returns the commandline parameter for a given index (not including flags)
     */
    template <typename T>
    void GetCmdLineArgument(std::size_t index, T &val)
    {
        using namespace std;
        if (index < args.size()) {
            std::istringstream str_stream(args[index]);
            str_stream >> val;
        }
    }

    /**
     * Returns the value specified for a given commandline parameter --<flag>=<value>
     */
    template <typename T>
    void GetCmdLineArgument(const char *arg_name, T &val)
    {
        using namespace std;

        for (std::size_t i = 0; i < keys.size(); ++i)
        {
            if (keys[i] == string(arg_name))
            {
                std::istringstream str_stream(values[i]);
                str_stream >> val;
            }
        }
    }


    /**
     * Returns the values specified for a given commandline parameter --<flag>=<value>,<value>*
     */
    template <typename T>
    void GetCmdLineArguments(const char *arg_name, std::vector<T> &vals)
    {
        using namespace std;

        if (CheckCmdLineFlag(arg_name))
        {
            // Clear any default values
            vals.clear();

            // Recover from multi-value string
            for (std::size_t i = 0; i < keys.size(); ++i)
            {
                if (keys[i] == string(arg_name))
                {
                    string val_string(values[i]);
                    std::istringstream str_stream(val_string);
                    string::size_type old_pos = 0;
                    string::size_type new_pos = 0;

                    // Iterate comma-separated values
                    T val;
                    while ((new_pos = val_string.find(',', old_pos)) != string::npos)
                    {
                        if (new_pos != old_pos)
                        {
                            str_stream.width(new_pos - old_pos);
                            str_stream >> val;
                            vals.push_back(val);
                        }

                        // skip over comma
                        str_stream.ignore(1);
                        old_pos = new_pos + 1;
                    }

                    // Read last value
                    str_stream >> val;
                    vals.push_back(val);
                }
            }
        }
    }


    /**
     * The number of pairs parsed
     */
    int ParsedArgc()
    {
        return (int) keys.size();
    }

    /**
     * Initialize device
     */
    hipError_t DeviceInit(int dev = -1)
    {
        hipError_t error = hipSuccess;

        do
        {
            int deviceCount;
            error = hipGetDeviceCount(&deviceCount);
            if (error) break;

            if (deviceCount == 0) {
                fprintf(stderr, "No devices supporting CUDA.\n");
                exit(1);
            }
            if (dev < 0)
            {
                GetCmdLineArgument("device", dev);
            }
            if ((dev > deviceCount - 1) || (dev < 0))
            {
                dev = 0;
            }

            error = hipSetDevice(dev);
            if (error) break;

            HIP_CHECK(hipMemGetInfo(&device_free_physmem, &device_total_physmem));

            // int ptx_version = 0;
            // error = hipcub::PtxVersion(ptx_version);
            // if (error) break;

            error = hipGetDeviceProperties(&deviceProp, dev);
            if (error) break;

            if (deviceProp.major < 1) {
                fprintf(stderr, "Device does not support Hip.\n");
                exit(1);
            }

            device_giga_bandwidth = float(deviceProp.memoryBusWidth) * deviceProp.memoryClockRate * 2 / 8 / 1000 / 1000;

            if (!CheckCmdLineFlag("quiet"))
            {
                printf(
                        "Using device %d: %s ( SM%d, %d SMs, "
                        "%lld free / %lld total MB physmem, "
                        "%.3f GB/s @ %d kHz mem clock, ECC %s)\n",
                    dev,
                    deviceProp.name,
                    deviceProp.major * 100 + deviceProp.minor * 10,
                    deviceProp.multiProcessorCount,
                    (unsigned long long) device_free_physmem / 1024 / 1024,
                    (unsigned long long) device_total_physmem / 1024 / 1024,
                    device_giga_bandwidth,
                    deviceProp.memoryClockRate,
                    (deviceProp.ECCEnabled) ? "on" : "off");
                fflush(stdout);
            }

        } while (0);

        return error;
    }
};
/******************************************************************************
 * Helper routines for list comparison and display
 ******************************************************************************/


/**
 * Compares the equivalence of two arrays
 */
template <typename S, typename T, typename OffsetT>
int CompareResults(T* computed, S* reference, OffsetT len, bool verbose = true)
{
    for (OffsetT i = 0; i < len; i++)
    {
        if (computed[i] != reference[i])
        {
            if (verbose) std::cout << "INCORRECT: [" << i << "]: "
                << CoutCast(computed[i]) << " != "
                << CoutCast(reference[i]);
            return 1;
        }
    }
    return 0;
}


/**
 * Compares the equivalence of two arrays
 */
template <typename OffsetT>
int CompareResults(float* computed, float* reference, OffsetT len, bool verbose = true)
{
    for (OffsetT i = 0; i < len; i++)
    {
        if (computed[i] != reference[i])
        {
            float difference = std::abs(computed[i]-reference[i]);
            float fraction = difference / std::abs(reference[i]);

            if (fraction > 0.0001)
            {
                if (verbose) std::cout << "INCORRECT: [" << i << "]: "
                    << "(computed) " << CoutCast(computed[i]) << " != "
                    << CoutCast(reference[i]) << " (difference:" << difference << ", fraction: " << fraction << ")";
                return 1;
            }
        }
    }
    return 0;
}


/**
 * Compares the equivalence of two arrays
 */
// template <typename OffsetT>
// int CompareResults(hipcub::NullType* computed, hipcub::NullType* reference, OffsetT len, bool verbose = true)
// {
//     return 0;
// }

/**
 * Compares the equivalence of two arrays
 */
template <typename OffsetT>
int CompareResults(double* computed, double* reference, OffsetT len, bool verbose = true)
{
    for (OffsetT i = 0; i < len; i++)
    {
        if (computed[i] != reference[i])
        {
            double difference = std::abs(computed[i]-reference[i]);
            double fraction = difference / std::abs(reference[i]);

            if (fraction > 0.0001)
            {
                if (verbose) std::cout << "INCORRECT: [" << i << "]: "
                    << CoutCast(computed[i]) << " != "
                    << CoutCast(reference[i]) << " (difference:" << difference << ", fraction: " << fraction << ")";
                return 1;
            }
        }
    }
    return 0;
}


// /**
//  * Verify the contents of a device array match those
//  * of a host array
//  */
// int CompareDeviceResults(
//     hipcub::NullType */* h_reference */,
//     hipcub::NullType */* d_data */,
//     std::size_t /* num_items */,
//     bool /* verbose */ = true,
//     bool /* display_data */ = false)
// {
//     return 0;
// }

/**
 * Verify the contents of a device array match those
 * of a host array
 */
// template <typename S, typename OffsetT>
// int CompareDeviceResults(
//     S *h_reference,
//     rocprim::discard_iterator d_data,
//     std::size_t num_items,
//     bool verbose = true,
//     bool display_data = false)
// {
//     return 0;
// }

/**
 * Verify the contents of a device array match those
 * of a host array
 */
template <typename S, typename T>
int CompareDeviceResults(
    S *h_reference,
    T *d_data,
    std::size_t num_items,
    bool verbose = true,
    bool display_data = false)
{
    // Allocate array on host
    T *h_data = (T*) malloc(num_items * sizeof(T));

    // Copy data back
    HIP_CHECK(hipMemcpy(h_data, d_data, sizeof(T) * num_items, hipMemcpyDeviceToHost));

    // Display data
    if (display_data)
    {
        printf("Reference:\n");
        for (std::size_t i = 0; i < num_items; i++)
        {
            std::cout << CoutCast(h_reference[i]) << ", ";
        }
        printf("\n\nComputed:\n");
        for (std::size_t i = 0; i < num_items; i++)
        {
            std::cout << CoutCast(h_data[i]) << ", ";
        }
        printf("\n\n");
    }

    // Check
    int retval = CompareResults(h_data, h_reference, num_items, verbose);

    // Cleanup
    if (h_data) free(h_data);

    return retval;
}


/**
 * Verify the contents of a device array match those
 * of a device array
 */
template <typename T>
int CompareDeviceDeviceResults(
    T *d_reference,
    T *d_data,
    std::size_t num_items,
    bool verbose = true,
    bool display_data = false)
{
    // Allocate array on host
    T *h_reference = (T*) malloc(num_items * sizeof(T));
    T *h_data = (T*) malloc(num_items * sizeof(T));

    // Copy data back
    HIP_CHECK(hipMemcpy(h_reference, d_reference, sizeof(T) * num_items, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_data, d_data, sizeof(T) * num_items, hipMemcpyDeviceToHost));

    // Display data
    if (display_data) {
        printf("Reference:\n");
        for (std::size_t i = 0; i < num_items; i++)
        {
            std::cout << CoutCast(h_reference[i]) << ", ";
        }
        printf("\n\nComputed:\n");
        for (std::size_t i = 0; i < num_items; i++)
        {
            std::cout << CoutCast(h_data[i]) << ", ";
        }
        printf("\n\n");
    }

    // Check
    int retval = CompareResults(h_data, h_reference, num_items, verbose);

    // Cleanup
    if (h_reference) free(h_reference);
    if (h_data) free(h_data);

    return retval;
}

/**
 * Print the contents of a host array
 */
template <typename InputIteratorT>
void DisplayResults(
    InputIteratorT h_data,
    std::size_t num_items)
{
    // Display data
    for (std::size_t i = 0; i < num_items; i++)
    {
        std::cout << CoutCast(h_data[i]) << ", ";
    }
    printf("\n");
}


int g_num_rand_samples = 0;
/**
 * Generates random keys.
 *
 * We always take the second-order byte from rand() because the higher-order
 * bits returned by rand() are commonly considered more uniformly distributed
 * than the lower-order bits.
 *
 * We can decrease the entropy level of keys by adopting the technique
 * of Thearling and Smith in which keys are computed from the bitwise AND of
 * multiple random samples:
 *
 * entropy_reduction    | Effectively-unique bits per key
 * -----------------------------------------------------
 * -1                   | 0
 * 0                    | 32
 * 1                    | 25.95 (81%)
 * 2                    | 17.41 (54%)
 * 3                    | 10.78 (34%)
 * 4                    | 6.42 (20%)
 * ...                  | ...
 *
 */
template <typename K>
void RandomBits(
    K &key,
    int entropy_reduction = 0,
    int begin_bit = 0,
    int end_bit = sizeof(K) * 8)
{
    const int NUM_BYTES = sizeof(K);
    const int WORD_BYTES = sizeof(unsigned int);
    const int NUM_WORDS = (NUM_BYTES + WORD_BYTES - 1) / WORD_BYTES;

    unsigned int word_buff[NUM_WORDS];

    if (entropy_reduction == -1)
    {
        memset((void *) &key, 0, sizeof(key));
        return;
    }

    if (end_bit < 0)
        end_bit = sizeof(K) * 8;

    while (true)
    {
        // Generate random word_buff
        for (int j = 0; j < NUM_WORDS; j++)
        {
            int current_bit = j * WORD_BYTES * 8;

            unsigned int word = 0xffffffff;
            word &= 0xffffffff << std::max(0, begin_bit - current_bit);
            word &= 0xffffffff >> std::max(0, (current_bit + (WORD_BYTES * 8)) - end_bit);

            for (int i = 0; i <= entropy_reduction; i++)
            {
                // Grab some of the higher bits from rand (better entropy, supposedly)
                word &= mersenne::genrand_int32();
                g_num_rand_samples++;
            }

            word_buff[j] = word;
        }

        memcpy(&key, word_buff, sizeof(K));

        K copy = key;
        if HIPCUB_IF_CONSTEXPR(std::is_floating_point<K>::value)
#ifndef _WIN32
            if(!std::isnan(copy))
#else
            // MSVC STL is missing the integral overload of std::isnan
            // https://github.com/microsoft/STL/issues/3400
            // They're right, the integral overloads were removed from the standard
            // after national body comments. std::isnan is supposed to behave _identically_
            // to the isnan CRT macro, where ISO C 7.12.3 states
            // > In the synopses in this subclause, real-floating indicates that the
            //   argument shall be an expression of real floating type.
            if(!std::isnan(static_cast<double>(copy)))
#endif
                break; // avoids NaNs when generating random floating point numbers
    }
}

/// Randomly select number between [0:max)
template <typename T>
T RandomValue(T max)
{
    unsigned int bits;
    unsigned int max_int = (unsigned int) -1;
    do {
        RandomBits(bits);
    } while (bits == max_int);

    return (T) ((double(bits) / double(max_int)) * double(max));
}

struct GpuTimer
{
    hipEvent_t start;
    hipEvent_t stop;

    GpuTimer()
    {
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
    }

    ~GpuTimer()
    {
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }

    void Start()
    {
        HIP_CHECK(hipEventRecord(start, 0));
    }

    void Stop()
    {
        HIP_CHECK(hipEventRecord(stop, 0));
    }

    float ElapsedMillis()
    {
        float elapsed;
        HIP_CHECK(hipEventSynchronize(stop));
        HIP_CHECK(hipEventElapsedTime(&elapsed, start, stop));
        return elapsed;
    }
};

#endif
