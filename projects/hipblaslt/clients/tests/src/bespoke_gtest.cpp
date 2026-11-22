// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt_test.hpp"
#include "hipBuffer.hpp"
#include "norm.hpp"
#include "sigmoid_test_observer.hpp"
#include "utility.hpp"
#include <hipblaslt/hipblaslt-ext.hpp>
#include <spawn.h>
#include <sys/wait.h>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <sstream>
#include <unistd.h>
#include <algorithm>
#include <cstdio>
#include <optional>
#include <regex>

// testing_matmul.hpp defines non-inline functions; its sole test translation
// unit is matmul_gtest.cpp. Reuse that real client and its test-only observer.
void testing_matmul(const Arguments& arg);

extern char** environ;
extern "C" void hipblaslt_tuning_reset_for_test();

namespace
{
    struct FixtureCase
    {
        const char* name;
        int64_t m, n, k;
        float beta;
        hipblaslt_activation_type activation;
    };
    const FixtureCase features[] = {
        {"beta_off_streamk", 64, 128, 385, 0, hipblaslt_activation_type::none},
        {"single_gelu", 64, 128, 385, 1, hipblaslt_activation_type::gelu},
        {"sigmoid_client", 64, 128, 385, 1, hipblaslt_activation_type::sigmoid},
        {"exact_size160", 320, 64, 64, 1, hipblaslt_activation_type::none},
        {"edge_removal", 64, 128, 385, 1, hipblaslt_activation_type::none},
        {"loop_exits", 64, 128, 512, 1, hipblaslt_activation_type::none},
    };

    std::string architecture()
    {
        int device = 0;
        hipDeviceProp_t props{};
        if(hipGetDevice(&device) != hipSuccess || hipGetDeviceProperties(&props, device) != hipSuccess)
            return {};
        const std::string arch = props.gcnArchName;
        return arch.substr(0, arch.find(':'));
    }

    int fixture_index(const std::string& root, const std::string& name,
                      std::string* kernel_name = nullptr)
    {
        std::ifstream input(root + "/indices.txt");
        std::string key, kernel;
        int index;
        while(input >> key >> index >> kernel)
            if(key == name)
            {
                if(kernel_name)
                    *kernel_name = kernel;
                return index;
            }
        ADD_FAILURE() << "Missing build-local fixture index: " << root << '/' << name;
        return -1;
    }

    void numerical_client(const FixtureCase& test, int index)
    {
        Arguments arg{};
        arg.init();
        arg.a_type = arg.b_type = arg.c_type = arg.d_type = HIP_R_16BF;
        arg.compute_type = HIPBLAS_COMPUTE_32F;
        arg.scale_type = HIP_R_32F;
        arg.transA = 'T';
        arg.transB = 'N';
        arg.M[0] = test.m;
        arg.N[0] = test.n;
        arg.K[0] = test.k;
        arg.lda[0] = arg.ldb[0] = test.k;
        arg.ldc[0] = arg.ldd[0] = test.m;
        arg.batch_count = 1;
        arg.alpha = 1;
        arg.beta = test.beta;
        arg.bias_vector = 0;
        arg.scaleAlpha_vector = false;
        arg.activation_type = test.activation;
        arg.api_method = 0;
        arg.algo_method = 2;
        arg.solution_index = index;
        arg.requested_solution_num = 1;
        arg.unit_check = 0;
        arg.norm_check = 1;
        arg.norm_check_assert = true;
        arg.timing = 0;
        arg.cold_iters = 0;
        arg.iters = 1;
        SigmoidEpilogueObservation observation;
        SigmoidObservationScope scope(observation);
        testing_matmul(arg);
        hipblaslt_cout << std::endl;
        if(test.activation == hipblaslt_activation_type::sigmoid)
        {
            ASSERT_TRUE(observation.sent);
            ASSERT_EQ(observation.epilogue, HIPBLASLT_EPILOGUE_SIGMOID);
        }
    }

    // This path intentionally never asks matmulIsAlgoSupported/isAlgoSupported.
    // The positive and incompatible calls use the exact same build-local index.
    void unchecked(const FixtureCase& test, int index, const std::string& refusal,
                   bool bias_present = false, bool select = false)
    {
        const int64_t m = test.m + (refusal == "size" ? 1 : 0), n = test.n, k = test.k;
        const float alpha = 1, beta = refusal == "beta" ? 1 : test.beta;
        std::vector<hip_bfloat16> a(m * k), b(k * n), c(m * n), d(m * n);
        for(int64_t p = 0; p < k; ++p)
        {
            for(int64_t i = 0; i < m; ++i)
                a[p + i * k] = hip_bfloat16(float((i + p) % 7 - 2) / 16);
            for(int64_t j = 0; j < n; ++j)
                b[p + j * k] = hip_bfloat16(float((j + 2 * p) % 11 - 4)
                                           * (j % 2 ? 1 : -1) / 16);
        }
        for(int64_t i = 0; i < m * n; ++i)
            c[i] = hip_bfloat16(float(i % 5 - 2) / 4);
        HipDeviceBuffer da(HIP_R_16BF, a.size()), db(HIP_R_16BF, b.size());
        HipDeviceBuffer dc(HIP_R_16BF, c.size()), dd(HIP_R_16BF, d.size());
        const size_t workspace_bytes = 32 * 1024 * 1024;
        HipDeviceBuffer workspace(HIP_R_8I, workspace_bytes);
        HipDeviceBuffer extra(HIP_R_32F, m);
        for(auto* buffer : {&da, &db, &dc, &dd, &workspace, &extra})
            ASSERT_EQ(buffer->memcheck(), hipSuccess);
        ASSERT_EQ(hipMemcpy(da.buf(), a.data(), da.getNumBytes(), hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemcpy(db.buf(), b.data(), db.getNumBytes(), hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemcpy(dc.buf(), c.data(), dc.getNumBytes(), hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemset(workspace.buf(), 0, workspace_bytes), hipSuccess);
        std::vector<float> extra_host(m, 1);
        if(bias_present)
            for(int64_t i = 0; i < m; ++i)
                extra_host[i] = float(i % 7 - 3) / 4;
        ASSERT_EQ(hipMemcpy(extra.buf(), extra_host.data(), extra.getNumBytes(), hipMemcpyHostToDevice), hipSuccess);
        hipblaslt_local_handle handle;
        hipblaslt_local_matmul_descr desc(HIPBLAS_OP_T, HIPBLAS_OP_N,
                                         HIPBLAS_COMPUTE_32F, HIP_R_32F);
        hipblaslt_local_matrix_layout la(k, m, k, HIP_R_16BF), lb(k, n, k, HIP_R_16BF);
        hipblaslt_local_matrix_layout ld(m, n, m, HIP_R_16BF);
        ASSERT_EQ(desc.status(), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(la.status(), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(lb.status(), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(ld.status(), HIPBLAS_STATUS_SUCCESS);
        hipblasLtEpilogue_t epilogue = test.activation == hipblaslt_activation_type::gelu
                                          ? HIPBLASLT_EPILOGUE_GELU : HIPBLASLT_EPILOGUE_DEFAULT;
        if(test.activation == hipblaslt_activation_type::sigmoid)
            epilogue = HIPBLASLT_EPILOGUE_SIGMOID;
        if(refusal == "activation")
            epilogue = HIPBLASLT_EPILOGUE_RELU;
        if(refusal == "bias" || bias_present)
        {
            epilogue = HIPBLASLT_EPILOGUE_BIAS;
            void* bias = extra.buf();
            const hipDataType type = HIP_R_32F;
            ASSERT_EQ(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                    &bias, sizeof(bias)), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                                    &type, sizeof(type)), HIPBLAS_STATUS_SUCCESS);
        }
        ASSERT_EQ(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                                &epilogue, sizeof(epilogue)), HIPBLAS_STATUS_SUCCESS);
        const void* alpha_ptr = &alpha;
        if(refusal == "sav")
        {
            const auto mode = HIPBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST;
            ASSERT_EQ(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_POINTER_MODE,
                                                    &mode, sizeof(mode)), HIPBLAS_STATUS_SUCCESS);
            alpha_ptr = extra.buf();
        }
        std::vector<hipblasLtMatmulHeuristicResult_t> algorithms;
        if(select)
        {
            hipblaslt_local_preference preference;
            ASSERT_EQ(preference.status(), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatmulPreferenceSetAttribute(
                          preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                          &workspace_bytes, sizeof(workspace_bytes)), HIPBLAS_STATUS_SUCCESS);
            algorithms.resize(1);
            int count = 0;
            ASSERT_EQ(hipblasLtMatmulAlgoGetHeuristic(handle, desc, la, lb, ld, ld,
                                                    preference, 1, algorithms.data(), &count),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(count, 1);
            ASSERT_EQ(algorithms[0].state, HIPBLAS_STATUS_SUCCESS);
        }
        else
        {
            std::vector<int> indices{index};
            ASSERT_EQ(hipblaslt_ext::getAlgosFromIndex(handle, indices, algorithms), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(algorithms.size(), 1u);
        }
        ASSERT_EQ(hipblaslt_ext::getIndexFromAlgo(algorithms[0].algo), index);
        hipblaslt_cout << "selected index=" << index << " kernel_name="
                      << hipblaslt_ext::getKernelNameFromAlgo(handle, algorithms[0].algo)
                      << " bias_present=" << bias_present << std::endl;
        const auto status = hipblasLtMatmul(handle, desc, alpha_ptr, da.buf(), la, db.buf(), lb,
                                           &beta, dc.buf(), ld, dd.buf(), ld, &algorithms[0].algo,
                                           workspace.buf(), workspace_bytes, nullptr);
        hipblaslt_cout << "unchecked index=" << index << " refusal=" << refusal
                      << " status=" << status << std::endl;
        if(!refusal.empty())
        {
            ASSERT_EQ(status, HIPBLAS_STATUS_INVALID_VALUE);
            return;
        }
        ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        ASSERT_EQ(hipMemcpy(d.data(), dd.buf(), dd.getNumBytes(), hipMemcpyDeviceToHost), hipSuccess);
        double squared_error = 0, squared_reference = 0;
        for(int64_t j = 0; j < n; ++j)
            for(int64_t i = 0; i < m; ++i)
            {
                double sum = 0;
                for(int64_t p = 0; p < k; ++p)
                    sum += double(a[p + i * k]) * double(b[p + j * k]);
                double reference = sum + beta * double(c[i + j * m]);
                if(bias_present)
                    reference += extra_host[i];
                if(test.activation == hipblaslt_activation_type::gelu)
                    reference = 0.5 * reference * (1 + std::tanh(0.7978845608028654
                                * (reference + 0.044715 * reference * reference * reference)));
                if(test.activation == hipblaslt_activation_type::sigmoid)
                    reference = 1 / (1 + std::exp(-reference));
                const double delta = double(d[i + j * m]) - reference;
                squared_error += delta * delta;
                squared_reference += reference * reference;
            }
        const double norm_error = std::sqrt(squared_error / squared_reference);
        hipblaslt_cout << "positive_control index=" << index << " norm_error=" << norm_error
                      << " tolerance=" << norm_tolerance(HIP_R_16BF) << std::endl;
        ASSERT_TRUE(std::isfinite(norm_error));
        ASSERT_TRUE(norm_check(norm_error, HIP_R_16BF, HIPBLAS_COMPUTE_32F, HIP_R_16BF, HIP_R_16BF));
    }
#include "bespoke_bias_test.hpp"


    void run_case(const std::string& name, const std::string& root)
    {
        if(name.find("bias_") == 0 && name != "bias_absent_pinned"
           && name != "bias_absent_selection" && name != "bias_present_selection")
        {
            bias_route_case(name, root);
            return;
        }
        if(name == "bias_absent_pinned" || name == "bias_absent_selection"
           || name == "bias_present_selection")
        {
            const FixtureCase bias_case{"bias_present", 64, 64, 128, 1,
                                        hipblaslt_activation_type::none};
            const int bespoke = fixture_index(root, "bias_present_bespoke");
            const int stock = fixture_index(root, "bias_present_general");
            ASSERT_GE(bespoke, 0);
            ASSERT_GE(stock, 0);
            ASSERT_NE(bespoke, stock);
            // Every route has a compatible, norm-checked pinned positive control.
            unchecked(bias_case, bespoke, "", true);
            if(testing::Test::HasFailure())
                return;
            if(name == "bias_absent_pinned")
                unchecked(bias_case, bespoke, "bias_absent");
            else if(name == "bias_absent_selection")
                unchecked(bias_case, stock, "", false, true);
            else
                unchecked(bias_case, bespoke, "", true, true);
            return;
        }
        for(const auto& test : features)
            if(name == test.name)
            {
                for(const char* variant : {"general", "bespoke"})
                {
                    const bool shared_general = !std::strcmp(variant, "general")
                                                && name != "exact_size160" && name != "loop_exits";
                    const int index = fixture_index(root, shared_general ? "streamk_general"
                                                                         : name + '_' + variant);
                    ASSERT_GE(index, 0);
                    unchecked(test, index, "");
                    if(test.activation == hipblaslt_activation_type::sigmoid
                       && !testing::Test::HasFailure())
                        numerical_client(test, index);
                    if(testing::Test::HasFailure())
                        return;
                }
                return;
            }
        const auto refusal = name.substr(std::strlen("refuse_"));
        const auto& control = refusal == "activation" ? features[1]
                              : refusal == "size" ? features[3] : features[0];
        const int index = fixture_index(root, std::string(control.name) + "_bespoke");
        ASSERT_GE(index, 0);
        unchecked(control, index, "");
        if(testing::Test::HasFailure())
            return;
        unchecked(control, index, refusal);
    }

    struct FixtureLogFile
    {
        std::string path;
        ~FixtureLogFile()
        {
            if(!path.empty())
                unlink(path.c_str());
        }
    };

    void check_bias_launches(const std::string& name, const std::string& root,
                             const std::string& path)
    {
        std::map<int, std::pair<std::string, int>> expected, actual;
        std::string kernel;
        const int bespoke = fixture_index(root, "bias_present_bespoke", &kernel);
        ASSERT_GE(bespoke, 0);
        expected[bespoke] = {kernel, name == "bias_present_selection" ? 2 : 1};
        if(name == "bias_absent_selection")
        {
            const int stock = fixture_index(root, "bias_present_general", &kernel);
            ASSERT_GE(stock, 0);
            expected[stock] = {kernel, 1};
        }
        std::ifstream log(path);
        ASSERT_TRUE(log.is_open());
        std::string line;
        while(std::getline(log, line))
        {
            if(line.empty())
                continue;
            ASSERT_EQ(line.substr(0, 3), "- {");
            ASSERT_EQ(line.back(), '}');
            std::istringstream fields(line.substr(3, line.size() - 4));
            std::map<std::string, std::string> record;
            std::string field;
            while(std::getline(fields, field, ','))
            {
                const auto colon = field.find(':');
                ASSERT_NE(colon, std::string::npos);
                const auto start = field.find_first_not_of(' ');
                const auto value = field.find_first_not_of(' ', colon + 1);
                ASSERT_NE(value, std::string::npos);
                record[field.substr(start, colon - start)]
                    = field.substr(value, field.find_last_not_of(' ') - value + 1);
            }
            ASSERT_EQ(record["function"], "matmul");
            ASSERT_FALSE(record["solution_index"].empty());
            ASSERT_FALSE(record["kernel_name"].empty());
            ASSERT_FALSE(record["call_count"].empty());
            const int index = std::stoi(record["solution_index"]);
            ASSERT_EQ(actual.count(index), 0u);
            actual[index] = {record["kernel_name"], std::stoi(record["call_count"])};
        }
        ASSERT_EQ(actual, expected);
    }

    class BespokeFixture : public testing::TestWithParam<const char*> {};
    TEST_P(BespokeFixture, pre_checkin)
    {
        const auto arch = architecture();
        if(arch != "gfx950" && arch != "gfx942")
            GTEST_SKIP() << "Bespoke fixtures target gfx950/gfx942";
        // A fresh exec, not fork-only HIP work: the process-global lazy library
        // must never inherit a stock library initialized by an earlier test.
        const std::string executable = hipblaslt_exepath() + "hipblaslt-test";
        const std::string name = GetParam();
        const bool bias_case = name.find("bias_absent_") == 0
                               || name == "bias_present_selection";
        FixtureLogFile log;
        std::vector<std::string> environment;
        std::vector<char*> envp;
        if(bias_case)
        {
            const char* tmp = std::getenv("TMPDIR");
            log.path = std::string(tmp ? tmp : "/tmp") + "/hipblaslt-bespoke-XXXXXX";
            const int fd = mkstemp(log.path.data());
            ASSERT_GE(fd, 0);
            close(fd);
            for(char** entry = environ; *entry; ++entry)
            {
                const std::string value(*entry);
                const auto key = value.substr(0, value.find('='));
                if(key != "HIPBLASLT_LOG_LEVEL" && key != "HIPBLASLT_LOG_MASK"
                   && key != "HIPBLASLT_LOG_FILE" && key != "GTEST_OUTPUT")
                    environment.push_back(value);
            }
            environment.emplace_back("HIPBLASLT_LOG_MASK=128");
            environment.emplace_back("HIPBLASLT_LOG_FILE=" + log.path);
            for(auto& entry : environment)
                envp.push_back(entry.data());
            envp.push_back(nullptr);
        }
        char* argv[] = {const_cast<char*>(executable.c_str()),
                        const_cast<char*>("--bespoke-fixture-child"),
                        const_cast<char*>(GetParam()), nullptr};
        std::optional<BiasLaunchCapture> tuning_log;
        if(name == "bias_tuning_cache_replay")
            tuning_log.emplace();
        pid_t pid;
        ASSERT_EQ(posix_spawn(&pid, executable.c_str(), nullptr, nullptr, argv,
                              bias_case ? envp.data() : environ), 0);
        int status = 0;
        ASSERT_EQ(waitpid(pid, &status, 0), pid);
        const auto tuning_output = tuning_log ? tuning_log->finish() : std::string{};
        ASSERT_TRUE(WIFEXITED(status)) << "fixture child terminated by a signal";
        ASSERT_EQ(WEXITSTATUS(status), 0) << GetParam() << tuning_output;
        if(bias_case)
            check_bias_launches(name, hipblaslt_exepath() + "bespoke-fixture/" + arch, log.path);
        if(name == "bias_tuning_cache_replay")
        {
            std::smatch summary;
            ASSERT_TRUE(std::regex_search(tuning_output, summary,
                std::regex("tuning-cache: summary shapes=([0-9]+) matched=([0-9]+) fellback=([0-9]+)")))
                << tuning_output;
            ASSERT_EQ(std::stoull(summary[1].str()), 2u);
            ASSERT_EQ(std::stoull(summary[2].str()), 1u);
            ASSERT_EQ(std::stoull(summary[3].str()), 1u);
            hipblaslt_cout << summary[0].str() << std::endl;
        }
    }
    INSTANTIATE_TEST_SUITE_P(Bespoke, BespokeFixture,
                            testing::Values("beta_off_streamk", "single_gelu", "sigmoid_client",
                                            "exact_size160", "edge_removal", "loop_exits",
                                            "refuse_beta", "refuse_activation", "refuse_bias",
                                            "refuse_sav", "refuse_size", "bias_absent_pinned",
                                            "bias_absent_selection", "bias_present_selection",
                                            "bias_pinned_ext", "bias_pinned_ext_setproblem",
                                            "bias_pinned_grouped", "bias_pinned_device_args",
                                            "bias_initialized_ext_result",
                                            "bias_support_matmul", "bias_support_ext",
                                            "bias_support_grouped", "bias_reuse",
                                            "bias_cached_selection", "bias_tuning_cache_replay",
                                            "bias_tuning_override_replay", "bias_grouped_mixed_first",
                                            "bias_grouped_mixed_last", "bias_null_pointer",
                                            "bias_null_pointer_ext", "bias_type_mismatch_ext",
                                            "bias_grouped_device_args_mixed_first",
                                            "bias_grouped_device_args_mixed_last",
                                            "bias_support_grouped_mixed_first",
                                            "bias_support_grouped_mixed_last",
                                            "bias_stock_shared_carrier",
                                            "bias_type_mismatch", "bias_stock_without_bias",
                                            "bias_stock_grouped_without_bias",
                                            "bias_stock_device_args_without_bias"),
                            [](const testing::TestParamInfo<const char*>& info) { return info.param; });
}

int hipblaslt_bespoke_child(const char* name)
{
    const auto arch = architecture();
    if(arch != "gfx950" && arch != "gfx942")
        return 2;
    const std::string root = hipblaslt_exepath() + "bespoke-fixture/" + arch;
    const std::string library = root + (std::string(name) == "bias_stock_shared_carrier"
                                           ? "/carrier-library/" : "/library/") + arch;
    if(setenv("HIPBLASLT_TENSILE_LIBPATH", library.c_str(), 1) != 0)
        return 2;
    if(std::string(name).find("bias_") == 0 && std::string(name).find("bias_absent_") != 0
       && std::string(name) != "bias_present_selection")
    {
        if(setenv("HIPBLASLT_LOG_MASK", "32", 1) != 0)
            return 2;
        unsetenv("HIPBLASLT_LOG_LEVEL");
        unsetenv("HIPBLASLT_LOG_FILE");
    }
    run_case(name, root);
    return testing::UnitTest::GetInstance()->ad_hoc_test_result().Failed() ? 1 : 0;
}
