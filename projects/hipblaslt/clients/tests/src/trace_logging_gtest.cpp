// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "hipblaslt_test.hpp"
#include "hipBuffer.hpp"
#include "utility.hpp"

#include <vector>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <string>

#ifndef _WIN32
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace
{
    // Each producer is run in its own process by test_trace_logging.py. The
    // library's real mask-64/128 aggregation is flushed at normal process exit.
    class TraceLoggingProducer : public ::testing::Test
    {
    protected:
        static constexpr int64_t size = 128;
        static constexpr size_t workspace_bytes = 32 * 1024 * 1024;
        hipblaslt_local_handle handle;
        hipblaslt_local_matmul_descr desc{HIPBLAS_OP_N, HIPBLAS_OP_N,
                                         HIPBLAS_COMPUTE_32F, HIP_R_32F};
        hipblaslt_local_matrix_layout layout{size, size, size, HIP_R_32F};
        hipblaslt_local_preference pref;
        HipDeviceBuffer a{HIP_R_32F, size * size}, b{HIP_R_32F, size * size};
        HipDeviceBuffer c{HIP_R_32F, size * size}, d{HIP_R_32F, size * size};
        HipDeviceBuffer aux{HIP_R_32F, size * size};
        HipDeviceBuffer device_alpha{HIP_R_32F, size}, device_beta{HIP_R_32F, 1};
        HipDeviceBuffer workspace{HIP_R_8I, workspace_bytes};
        float alpha = 1, beta = 1;
        const void* alpha_ptr = &alpha;
        const void* beta_ptr = &beta;
        hipblasLtMatmulHeuristicResult_t result{};

        void SetUp() override
        {
            ASSERT_EQ(desc.status(), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(layout.status(), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(pref.status(), HIPBLAS_STATUS_SUCCESS);
            for(auto* buffer : {&a, &b, &c, &d, &aux, &device_alpha, &device_beta, &workspace})
                ASSERT_EQ(buffer->memcheck(), hipSuccess);
            std::vector<float> matrix(size * size, 0.25f), alphas(size, alpha);
            for(auto* buffer : {&a, &b, &c, &aux})
                ASSERT_EQ(hipMemcpy(buffer->buf(), matrix.data(), buffer->getNumBytes(),
                                    hipMemcpyHostToDevice), hipSuccess);
            ASSERT_EQ(hipMemcpy(device_alpha.buf(), alphas.data(), device_alpha.getNumBytes(),
                                hipMemcpyHostToDevice), hipSuccess);
            ASSERT_EQ(hipMemcpy(device_beta.buf(), &beta, sizeof(beta), hipMemcpyHostToDevice),
                      hipSuccess);
            ASSERT_EQ(hipblasLtMatmulPreferenceSetAttribute(pref,
                          HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                          &workspace_bytes, sizeof(workspace_bytes)), HIPBLAS_STATUS_SUCCESS);
        }

        void epilogue(hipblasLtEpilogue_t value)
        {
            ASSERT_EQ(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                                    &value, sizeof(value)), HIPBLAS_STATUS_SUCCESS);
            if(value == HIPBLASLT_EPILOGUE_DRELU || value == HIPBLASLT_EPILOGUE_DGELU)
            {
                void* ptr = aux.buf();
                ASSERT_EQ(hipblasLtMatmulDescSetAttribute(desc,
                              HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
                              &ptr, sizeof(ptr)), HIPBLAS_STATUS_SUCCESS);
                ASSERT_EQ(hipblasLtMatmulDescSetAttribute(desc,
                              HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,
                              &size, sizeof(size)), HIPBLAS_STATUS_SUCCESS);
                const int64_t stride = size * size;
                ASSERT_EQ(hipblasLtMatmulDescSetAttribute(desc,
                              HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE,
                              &stride, sizeof(stride)), HIPBLAS_STATUS_SUCCESS);
            }
        }

        void pointer_mode(hipblasLtPointerMode_t value)
        {
            ASSERT_EQ(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_POINTER_MODE,
                                                    &value, sizeof(value)), HIPBLAS_STATUS_SUCCESS);
            alpha_ptr = value == HIPBLASLT_POINTER_MODE_HOST ? &alpha : device_alpha.buf();
            beta_ptr = value == HIPBLASLT_POINTER_MODE_DEVICE ? device_beta.buf() : &beta;
        }

        void clamp(float lower, float upper)
        {
            ASSERT_EQ(hipblasLtMatmulDescSetAttribute(desc,
                          HIPBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG0_EXT,
                          &lower, sizeof(lower)), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatmulDescSetAttribute(desc,
                          HIPBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG1_EXT,
                          &upper, sizeof(upper)), HIPBLAS_STATUS_SUCCESS);
        }

        void select()
        {
            int count = 0;
            ASSERT_EQ(hipblasLtMatmulAlgoGetHeuristic(handle, desc, layout, layout,
                          layout, layout, pref, 1, &result, &count), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(count, 1);
            ASSERT_EQ(result.state, HIPBLAS_STATUS_SUCCESS);
        }

        void launch(int count)
        {
            for(int i = 0; i < count; ++i)
                ASSERT_EQ(hipblasLtMatmul(handle, desc, alpha_ptr, a.buf(), layout,
                              b.buf(), layout, beta_ptr, c.buf(), layout, d.buf(), layout,
                              &result.algo, workspace.buf(), workspace_bytes, nullptr),
                          HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        }

        void argument_pair(int argument, float first, float second, int first_count = 1,
                           int second_count = 1)
        {
            epilogue(HIPBLASLT_EPILOGUE_CLAMP_EXT);
            clamp(-1.5f, 2.25f);
            ASSERT_NO_FATAL_FAILURE(select());
            clamp(argument == 0 ? first : -1.5f, argument == 1 ? first : 2.25f);
            ASSERT_NO_FATAL_FAILURE(launch(first_count));
            // Keep the same solution, shape and other argument for the second call.
            clamp(argument == 0 ? second : -1.5f, argument == 1 ? second : 2.25f);
            launch(second_count);
        }
    };

    TEST_F(TraceLoggingProducer, Sigmoid)
    {
        epilogue(HIPBLASLT_EPILOGUE_SIGMOID);
        ASSERT_NO_FATAL_FAILURE(select());
        launch(1);
    }
    TEST_F(TraceLoggingProducer, Default)
    {
        // Intentionally do not set POINTER_MODE: the descriptor default is host.
        ASSERT_NO_FATAL_FAILURE(select());
        launch(1);
    }
    TEST_F(TraceLoggingProducer, Device)
    {
        pointer_mode(HIPBLASLT_POINTER_MODE_DEVICE);
        ASSERT_NO_FATAL_FAILURE(select());
        launch(1);
    }
    TEST_F(TraceLoggingProducer, Vector)
    {
        pointer_mode(HIPBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST);
        ASSERT_NO_FATAL_FAILURE(select());
        launch(1);
    }
    TEST_F(TraceLoggingProducer, Clamp)
    {
        epilogue(HIPBLASLT_EPILOGUE_CLAMP_EXT);
        clamp(-1.5f, 2.25f);
        ASSERT_NO_FATAL_FAILURE(select());
        launch(1);
    }
    TEST_F(TraceLoggingProducer, PointerPair)
    {
        pointer_mode(HIPBLASLT_POINTER_MODE_DEVICE);
        ASSERT_NO_FATAL_FAILURE(select());
        ASSERT_NO_FATAL_FAILURE(launch(2));
        pointer_mode(HIPBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST);
        // Reuse the very same algo to isolate the pointer-mode aggregation key.
        launch(3);
    }
    TEST_F(TraceLoggingProducer, ClampArg0Pair)
    {
        epilogue(HIPBLASLT_EPILOGUE_CLAMP_EXT);
        clamp(-1.5f, 2.25f);
        ASSERT_NO_FATAL_FAILURE(select());
        ASSERT_NO_FATAL_FAILURE(launch(2));
        clamp(-0.5f, 2.25f);
        launch(3);
    }
    TEST_F(TraceLoggingProducer, ClampArg1Pair)
    {
        epilogue(HIPBLASLT_EPILOGUE_CLAMP_EXT);
        clamp(-1.5f, 2.25f);
        ASSERT_NO_FATAL_FAILURE(select());
        ASSERT_NO_FATAL_FAILURE(launch(2));
        clamp(-1.5f, 3.5f);
        launch(3);
    }
    TEST_F(TraceLoggingProducer, Relu)
    {
        epilogue(HIPBLASLT_EPILOGUE_RELU);
        ASSERT_NO_FATAL_FAILURE(select());
        launch(1);
    }
    TEST_F(TraceLoggingProducer, Gelu)
    {
        epilogue(HIPBLASLT_EPILOGUE_GELU);
        ASSERT_NO_FATAL_FAILURE(select());
        launch(1);
    }
    TEST_F(TraceLoggingProducer, Silu)
    {
        epilogue(HIPBLASLT_EPILOGUE_SWISH_EXT);
        ASSERT_NO_FATAL_FAILURE(select());
        launch(1);
    }
    TEST_F(TraceLoggingProducer, DRelu)
    {
        epilogue(HIPBLASLT_EPILOGUE_DRELU);
        ASSERT_NO_FATAL_FAILURE(select());
        launch(1);
    }
    TEST_F(TraceLoggingProducer, DGelu)
    {
        epilogue(HIPBLASLT_EPILOGUE_DGELU);
        ASSERT_NO_FATAL_FAILURE(select());
        launch(1);
    }
    // Arguments one binary32 ulp apart, and values whose shortest
    // round-trip form needs an exponent.
    TEST_F(TraceLoggingProducer, ClampArg0Ulp)
    {
        epilogue(HIPBLASLT_EPILOGUE_CLAMP_EXT);
        clamp(1.0f, 2.25f);
        ASSERT_NO_FATAL_FAILURE(select());
        ASSERT_NO_FATAL_FAILURE(launch(2));
        clamp(std::nextafterf(1.0f, 2.0f), 2.25f);
        launch(3);
    }
    TEST_F(TraceLoggingProducer, ClampArg1Ulp)
    {
        epilogue(HIPBLASLT_EPILOGUE_CLAMP_EXT);
        clamp(-1.5f, 1.0f);
        ASSERT_NO_FATAL_FAILURE(select());
        ASSERT_NO_FATAL_FAILURE(launch(2));
        clamp(-1.5f, std::nextafterf(1.0f, 2.0f));
        launch(3);
    }
    TEST_F(TraceLoggingProducer, ClampExponent)
    {
        epilogue(HIPBLASLT_EPILOGUE_CLAMP_EXT);
        clamp(-2e10f, 1e10f);
        ASSERT_NO_FATAL_FAILURE(select());
        launch(1);
    }

    TEST_F(TraceLoggingProducer, ClampArg0NaN)
    {
        const float nan = std::numeric_limits<float>::quiet_NaN();
        argument_pair(0, nan, nan);
    }
    TEST_F(TraceLoggingProducer, ClampArg1NaN)
    {
        const float nan = std::numeric_limits<float>::quiet_NaN();
        argument_pair(1, nan, nan);
    }
    TEST_F(TraceLoggingProducer, ClampArg0NaNSignPair)
    {
        const float nan = std::numeric_limits<float>::quiet_NaN();
        argument_pair(0, nan, -nan);
    }
    TEST_F(TraceLoggingProducer, ClampArg1NaNSignPair)
    {
        const float nan = std::numeric_limits<float>::quiet_NaN();
        argument_pair(1, nan, -nan);
    }
    // Two quiet NaNs with different payload bits (same sign).
    float nan_payload(uint32_t bits)
    {
        float value;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }
    TEST_F(TraceLoggingProducer, ClampArg0NaNPayloadPair)
    {
        argument_pair(0, nan_payload(0x7fc00000u), nan_payload(0x7fc12345u));
    }
    TEST_F(TraceLoggingProducer, ClampArg1NaNPayloadPair)
    {
        argument_pair(1, nan_payload(0x7fc00000u), nan_payload(0x7fc12345u));
    }
    TEST_F(TraceLoggingProducer, ClampArg0InfinityPair)
    {
        const float inf = std::numeric_limits<float>::infinity();
        argument_pair(0, inf, -inf, 2, 3);
    }
    TEST_F(TraceLoggingProducer, ClampArg1InfinityPair)
    {
        const float inf = std::numeric_limits<float>::infinity();
        argument_pair(1, inf, -inf, 2, 3);
    }

#ifndef _WIN32
    using TraceRecord = std::map<std::string, std::string>;

    class TraceLogFile
    {
    public:
        TraceLogFile()
        {
            const char* tmp = std::getenv("TMPDIR");
            path = std::string(tmp ? tmp : "/tmp") + "/hipblaslt-trace-XXXXXX";
            int fd = mkstemp(path.data());
            if(fd >= 0)
            {
                valid = true;
                close(fd);
            }
        }
        ~TraceLogFile()
        {
            if(valid)
                unlink(path.c_str());
        }
        std::string path;
        bool valid = false;
    };

    void read_trace(const char* scenario, std::vector<TraceRecord>& records)
    {
        TraceLogFile log;
        ASSERT_TRUE(log.valid);
        // The logger is process-global and flushes at exit. Exec a fresh producer,
        // inheriting the allocated GPU but replacing only the logging settings.
        std::vector<std::string> environment;
        for(char** entry = environ; *entry; ++entry)
        {
            const std::string value(*entry);
            const auto name = value.substr(0, value.find('='));
            if(name != "HIPBLASLT_LOG_LEVEL" && name != "HIPBLASLT_LOG_MASK"
               && name != "HIPBLASLT_LOG_FILE" && name != "GTEST_OUTPUT")
                environment.push_back(value);
        }
        environment.emplace_back("HIPBLASLT_LOG_MASK=128");
        environment.emplace_back("HIPBLASLT_LOG_FILE=" + log.path);
        std::vector<char*> envp;
        for(auto& entry : environment)
            envp.push_back(entry.data());
        envp.push_back(nullptr);
        const std::string executable = hipblaslt_exepath() + "hipblaslt-test";
        const std::string filter = std::string("--gtest_filter=TraceLoggingProducer.") + scenario;
        char* argv[] = {const_cast<char*>(executable.c_str()), const_cast<char*>(filter.c_str()),
                        const_cast<char*>("--gtest_repeat=1"),
                        const_cast<char*>("--gtest_color=no"), nullptr};
        pid_t pid;
        ASSERT_EQ(posix_spawn(&pid, executable.c_str(), nullptr, nullptr, argv, envp.data()), 0);
        int status = 0;
        ASSERT_EQ(waitpid(pid, &status, 0), pid);
        ASSERT_TRUE(WIFEXITED(status));
        ASSERT_EQ(WEXITSTATUS(status), 0) << scenario;

        std::ifstream stream(log.path);
        ASSERT_TRUE(stream.is_open());
        std::string line;
        while(std::getline(stream, line))
        {
            if(line.empty())
                continue;
            ASSERT_EQ(line.substr(0, 3), "- {") << line;
            ASSERT_EQ(line.back(), '}') << line;
            std::istringstream fields(line.substr(3, line.size() - 4));
            TraceRecord record;
            std::string field;
            while(std::getline(fields, field, ','))
            {
                const auto colon = field.find(':');
                ASSERT_NE(colon, std::string::npos) << field;
                const auto key_start = field.find_first_not_of(' ');
                const auto value_start = field.find_first_not_of(' ', colon + 1);
                const auto value_end = field.find_last_not_of(' ');
                ASSERT_NE(value_start, std::string::npos) << field;
                ASSERT_TRUE(record.emplace(field.substr(key_start, colon - key_start),
                                           field.substr(value_start, value_end - value_start + 1))
                                .second) << field;
            }
            EXPECT_EQ(record["function"], "matmul");
            EXPECT_FALSE(record["kernel_name"].empty());
            EXPECT_FALSE(record["solution_Name"].empty());
            records.push_back(std::move(record));
        }
        ASSERT_FALSE(records.empty()) << scenario << " produced no mask-128 records";
    }

    void expect_float(const std::string& text, float expected)
    {
        ASSERT_FALSE(text.empty()) << "mask-128 record omitted an activation argument";
        char* end = nullptr;
        const float parsed = std::strtof(text.c_str(), &end);
        EXPECT_EQ(end, text.c_str() + text.size()) << text;
        EXPECT_EQ(parsed, expected) << text;
    }

    struct NaNCase
    {
        const char* scenario;
        const char* argument;
    };
    class TraceLoggingNaN : public ::testing::TestWithParam<NaNCase> {};
    TEST_P(TraceLoggingNaN, pre_checkin_NaNsAggregateWithCanonicalSpelling)
    {
        // Fix: NaN-equal keys with a matching canonical hash and printed "nan".
        // Regression: float ==, sign-dependent NaN hashes, or signed NaN text.
        const auto test = GetParam();
        std::vector<TraceRecord> records;
        ASSERT_NO_FATAL_FAILURE(read_trace(test.scenario, records));
        EXPECT_EQ(records.size(), 1u) << "NaN calls must form one mask-128 identity";
        for(auto& record : records)
        {
            EXPECT_EQ(record["call_count"], "2");
            EXPECT_EQ(record[test.argument], "nan");
            EXPECT_EQ(record["activation_type"], "clamp");
            EXPECT_EQ(record["pointer_mode"], "0");
            expect_float(record[test.argument == std::string("activation_arg0")
                                    ? "activation_arg1" : "activation_arg0"],
                         test.argument == std::string("activation_arg0") ? 2.25f : -1.5f);
        }
        // Verify the producer varied no other logged fact, including recipe identity.
        auto identity = records.front();
        identity.erase(test.argument);
        identity.erase("call_count");
        for(auto record : records)
        {
            record.erase(test.argument);
            record.erase("call_count");
            EXPECT_EQ(record, identity);
        }
    }
    INSTANTIATE_TEST_SUITE_P(Mask128, TraceLoggingNaN,
        ::testing::Values(NaNCase{"ClampArg0NaN", "activation_arg0"},
                          NaNCase{"ClampArg1NaN", "activation_arg1"},
                          NaNCase{"ClampArg0NaNSignPair", "activation_arg0"},
                          NaNCase{"ClampArg1NaNSignPair", "activation_arg1"},
                          NaNCase{"ClampArg0NaNPayloadPair", "activation_arg0"},
                          NaNCase{"ClampArg1NaNPayloadPair", "activation_arg1"}),
        [](const ::testing::TestParamInfo<NaNCase>& info) { return info.param.scenario; });

    class TraceLoggingRecord : public ::testing::TestWithParam<const char*> {};
    TEST_P(TraceLoggingRecord, pre_checkin_RecordContent)
    {
        // Preserve the raw enum, sigmoid mapping and max_digits10 argument output.
        // Omitting/defaulting a field, mapping sigmoid to none, or six-digit
        // formatting must fail this host test without any geko test installed.
        const std::string scenario = GetParam();
        std::vector<TraceRecord> records;
        ASSERT_NO_FATAL_FAILURE(read_trace(GetParam(), records));
        if(scenario == "PointerPair")
        {
            ASSERT_EQ(records.size(), 2u);
            std::map<std::string, std::string> counts;
            for(auto& record : records)
                counts.emplace(record["pointer_mode"], record["call_count"]);
            const std::map<std::string, std::string> expected{{"1", "2"}, {"4", "3"}};
            EXPECT_EQ(counts, expected);
            auto left = records[0], right = records[1];
            left.erase("pointer_mode"); right.erase("pointer_mode");
            left.erase("call_count"); right.erase("call_count");
            EXPECT_EQ(left, right);
        }
        else if(scenario == "ClampArg0Ulp" || scenario == "ClampArg1Ulp")
        {
            ASSERT_EQ(records.size(), 2u);
            const auto arg = scenario == "ClampArg0Ulp" ? "activation_arg0" : "activation_arg1";
            std::map<std::string, TraceRecord> by_count;
            for(auto& record : records)
            {
                EXPECT_EQ(record["activation_type"], "clamp");
                EXPECT_EQ(record["pointer_mode"], "0");
                by_count.emplace(record["call_count"], record);
            }
            ASSERT_EQ(by_count.count("2"), 1u);
            ASSERT_EQ(by_count.count("3"), 1u);
            expect_float(by_count.at("2")[arg], 0x1p0f);
            expect_float(by_count.at("3")[arg], 0x1.000002p0f);
            const auto other = scenario == "ClampArg0Ulp" ? "activation_arg1" : "activation_arg0";
            for(auto& record : records)
                expect_float(record[other], scenario == "ClampArg0Ulp" ? 2.25f : -1.5f);
            auto left = records[0], right = records[1];
            left.erase(arg); right.erase(arg);
            left.erase("call_count"); right.erase("call_count");
            EXPECT_EQ(left, right);
        }
        else
        {
            ASSERT_EQ(records.size(), 1u);
            auto& record = records.front();
            EXPECT_EQ(record["call_count"], "1");
            EXPECT_EQ(record["pointer_mode"], scenario == "Device" ? "1"
                                             : scenario == "Vector" ? "4" : "0");
            EXPECT_EQ(record["activation_type"], scenario == "Sigmoid" ? "sigmoid"
                                                : scenario == "ClampExponent" ? "clamp" : "none");
            expect_float(record["activation_arg0"], scenario == "ClampExponent" ? -2e10f : 0.0f);
            expect_float(record["activation_arg1"], scenario == "ClampExponent" ? 1e10f : 0.0f);
        }
    }
    INSTANTIATE_TEST_SUITE_P(Mask128, TraceLoggingRecord,
        ::testing::Values("Default", "Device", "Vector", "PointerPair", "Sigmoid",
                          "ClampArg0Ulp", "ClampArg1Ulp", "ClampExponent"),
        [](const ::testing::TestParamInfo<const char*>& info) { return info.param; });

    class TraceLoggingInfinity : public ::testing::TestWithParam<const char*> {};
    TEST_P(TraceLoggingInfinity, pre_checkin_InfinitySignsRemainDistinct)
    {
        // Preserve non-NaN equality: treating all nonfinite arguments as NaN
        // would merge these real producer calls and lose their separate counts.
        std::vector<TraceRecord> records;
        ASSERT_NO_FATAL_FAILURE(read_trace(GetParam(), records));
        ASSERT_EQ(records.size(), 2u);
        const auto arg = std::string(GetParam()) == "ClampArg0InfinityPair"
                             ? "activation_arg0" : "activation_arg1";
        std::map<std::string, std::string> counts;
        for(auto& record : records)
            counts.emplace(record[arg], record["call_count"]);
        const std::map<std::string, std::string> expected{{"inf", "2"}, {"-inf", "3"}};
        EXPECT_EQ(counts, expected);
        auto left = records[0], right = records[1];
        left.erase(arg); right.erase(arg);
        left.erase("call_count"); right.erase("call_count");
        EXPECT_EQ(left, right);
    }
    INSTANTIATE_TEST_SUITE_P(Mask128, TraceLoggingInfinity,
        ::testing::Values("ClampArg0InfinityPair", "ClampArg1InfinityPair"),
        [](const ::testing::TestParamInfo<const char*>& info) { return info.param; });
#endif
}
