// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Included inside BespokeFixture's private namespace. Every route has a real
// matching launch and enforced norm_error verdict before the incompatible probe.
// Pass: retain bias predicates at selection, support and every launch boundary.
// Break: omit the bias-presence conjunct, grouped member checks or replay revalidation.

struct BiasProblem
{
    static constexpr int64_t m = 64, n = 64, k = 128;
    static constexpr size_t workspace_bytes = 32 * 1024 * 1024;
    const float alpha = 1, beta = 0;
    std::vector<hip_bfloat16> a, b, c, d;
    std::vector<float> bias;
    HipDeviceBuffer da, db, dc, dd, extra, workspace;
    hipblaslt_local_handle handle;
    hipblaslt_local_matmul_descr desc;
    hipblaslt_local_matrix_layout la, lb, ld;
    bool present = true;

    BiasProblem()
        : a(m * k), b(k * n), c(m * n), d(m * n), bias(m),
          da(HIP_R_16BF, a.size()), db(HIP_R_16BF, b.size()),
          dc(HIP_R_16BF, c.size()), dd(HIP_R_16BF, d.size()),
          extra(HIP_R_32F, m), workspace(HIP_R_8I, workspace_bytes),
          desc(HIPBLAS_OP_T, HIPBLAS_OP_N, HIPBLAS_COMPUTE_32F, HIP_R_32F),
          la(k, m, k, HIP_R_16BF), lb(k, n, k, HIP_R_16BF), ld(m, n, m, HIP_R_16BF)
    {
    }

    void prepare()
    {
        for(auto* buffer : {&da, &db, &dc, &dd, &extra, &workspace})
            ASSERT_EQ(buffer->memcheck(), hipSuccess);
        ASSERT_EQ(desc.status(), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(la.status(), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(lb.status(), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(ld.status(), HIPBLAS_STATUS_SUCCESS);
        for(int64_t p = 0; p < k; ++p)
        {
            for(int64_t i = 0; i < m; ++i)
                a[p + i * k] = hip_bfloat16(float((i + p) % 7 - 2) / 16);
            for(int64_t j = 0; j < n; ++j)
                b[p + j * k] = hip_bfloat16(float((j + 2 * p) % 11 - 4) / 16);
        }
        for(int64_t i = 0; i < m; ++i)
            bias[i] = float(i % 7 - 3) / 4;
        ASSERT_EQ(hipMemcpy(da.buf(), a.data(), da.getNumBytes(), hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemcpy(db.buf(), b.data(), db.getNumBytes(), hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemcpy(dc.buf(), c.data(), dc.getNumBytes(), hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemcpy(extra.buf(), bias.data(), extra.getNumBytes(), hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemset(workspace.buf(), 0, workspace_bytes), hipSuccess);
        set_bias(true);
        reset_output();
    }

    void set_bias(bool enabled, bool null_pointer = false, hipDataType type = HIP_R_32F)
    {
        present = enabled;
        const auto epilogue = enabled ? HIPBLASLT_EPILOGUE_BIAS : HIPBLASLT_EPILOGUE_DEFAULT;
        void* pointer = enabled && !null_pointer ? extra.buf() : nullptr;
        ASSERT_EQ(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                                &epilogue, sizeof(epilogue)), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                &pointer, sizeof(pointer)), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                                &type, sizeof(type)), HIPBLAS_STATUS_SUCCESS);
    }

    void reset_output()
    {
        ASSERT_EQ(hipMemset(dd.buf(), 0x5a, dd.getNumBytes()), hipSuccess);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    }

    void unchanged()
    {
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        std::vector<unsigned char> bytes(dd.getNumBytes());
        ASSERT_EQ(hipMemcpy(bytes.data(), dd.buf(), bytes.size(), hipMemcpyDeviceToHost), hipSuccess);
        ASSERT_TRUE(std::all_of(bytes.begin(), bytes.end(), [](auto v) { return v == 0x5a; }))
            << "refused route wrote D";
    }

    void numerical(int index, const std::string& route, bool expected_bias)
    {
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        ASSERT_EQ(hipMemcpy(d.data(), dd.buf(), dd.getNumBytes(), hipMemcpyDeviceToHost), hipSuccess);
        double error = 0, reference_norm = 0;
        for(int64_t j = 0; j < n; ++j)
            for(int64_t i = 0; i < m; ++i)
            {
                double reference = expected_bias ? bias[i] : 0;
                for(int64_t p = 0; p < k; ++p)
                    reference += double(a[p + i * k]) * double(b[p + j * k]);
                const double delta = double(d[i + j * m]) - reference;
                error += delta * delta;
                reference_norm += reference * reference;
            }
        const double norm_error = std::sqrt(error / reference_norm);
        hipblaslt_cout << "positive_control route=" << route << " index=" << index
                      << " norm_error=" << norm_error
                      << " tolerance=" << norm_tolerance(HIP_R_16BF) << std::endl;
        ASSERT_TRUE(std::isfinite(norm_error));
        ASSERT_TRUE(norm_check(norm_error, HIP_R_16BF, HIPBLAS_COMPUTE_32F, HIP_R_16BF, HIP_R_16BF));
    }

    void numerical(int index, const std::string& route)
    {
        numerical(index, route, present);
    }

    hipblasStatus_t matmul(const hipblasLtMatmulAlgo_t* algo)
    {
        return hipblasLtMatmul(handle, desc, &alpha, da.buf(), la, db.buf(), lb,
                              &beta, dc.buf(), ld, dd.buf(), ld, algo,
                              workspace.buf(), workspace_bytes, nullptr);
    }

    hipblasStatus_t set_problem(hipblaslt_ext::Gemm& gemm)
    {
        return gemm.setProblem(desc, &alpha, da.buf(), la, db.buf(), lb, &beta,
                               dc.buf(), ld, dd.buf(), ld);
    }
};

struct BiasLaunchCapture
{
    FILE* file = std::tmpfile();
    int saved = -1;
    BiasLaunchCapture()
    {
        if(file)
        {
            std::fflush(nullptr);
            saved = dup(STDERR_FILENO);
            if(saved >= 0 && dup2(fileno(file), STDERR_FILENO) < 0)
            {
                close(saved);
                saved = -1;
            }
        }
    }
    std::string finish()
    {
        if(saved < 0)
            return {};
        hipblaslt_cerr.flush();
        std::fflush(nullptr);
        dup2(saved, STDERR_FILENO);
        close(saved);
        saved = -1;
        std::rewind(file);
        std::string result;
        char buffer[1024];
        while(const auto count = std::fread(buffer, 1, sizeof(buffer), file))
            result.append(buffer, count);
        return result;
    }
    ~BiasLaunchCapture()
    {
        finish();
        if(file)
            std::fclose(file);
    }
};

void bias_launch_identity(const std::string& log, int index)
{
    const auto token = "--solution_index ";
    auto position = log.find(token);
    ASSERT_NE(position, std::string::npos) << "launch has no library-issued identity: " << log;
    do
    {
        ASSERT_EQ(std::atoi(log.c_str() + position + std::strlen(token)), index);
        position = log.find(token, position + std::strlen(token));
    } while(position != std::string::npos);
}

struct BiasTuningFile
{
    std::string path;
    ~BiasTuningFile()
    {
        if(!path.empty())
            unlink(path.c_str());
    }
};

void bias_route_case(const std::string& name, const std::string& root)
{
    const bool grouped = name.find("grouped") != std::string::npos
                         || name.find("device_args") != std::string::npos;
    const bool stock = name.find("stock_") != std::string::npos;
    const bool query = name.find("support_") != std::string::npos;
    const bool mixed = name.find("mixed_") != std::string::npos;
    const bool device_args = name.find("device_args") != std::string::npos;
    const bool initialized = name == "bias_initialized_ext_result";
    const bool replay = name.find("replay") != std::string::npos || name == "bias_cached_selection";
    const std::string stem = grouped ? "bias_grouped" : "bias_present";
    std::string kernel;
    const bool carrier = name == "bias_stock_shared_carrier";
    const int index = fixture_index(root, stem + (stock && !carrier ? "_general" : "_bespoke"), &kernel);
    ASSERT_GE(index, 0);
    BiasTuningFile tuning;
    if(replay)
    {
        ASSERT_EQ(unsetenv("HIPBLASLT_TUNING_MODE"), 0);
        ASSERT_EQ(unsetenv("HIPBLASLT_TUNING_CACHE_PATH"), 0);
        ASSERT_EQ(unsetenv("HIPBLASLT_TUNING_OVERRIDE_FILE"), 0);
    }
    BiasProblem first;
    std::optional<BiasProblem> second;
    first.prepare();
    if(mixed)
    {
        second.emplace();
        second->prepare();
    }
    if(testing::Test::HasFailure())
        return;
    const bool tuning_replay = name == "bias_tuning_cache_replay"
                               || name == "bias_tuning_override_replay";
    auto tuning_launch = [&]() -> hipblasStatus_t {
        if(name == "bias_tuning_cache_replay")
            return first.matmul(nullptr);
        hipblaslt_local_preference preference;
        const uint64_t workspace_limit = BiasProblem::workspace_bytes;
        auto status = hipblasLtMatmulPreferenceSetAttribute(preference,
            HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_limit, sizeof(workspace_limit));
        if(status != HIPBLAS_STATUS_SUCCESS)
            return status;
        hipblasLtMatmulHeuristicResult_t selected{};
        int count = 0;
        status = hipblasLtMatmulAlgoGetHeuristic(first.handle, first.desc,
            first.la, first.lb, first.ld, first.ld, preference, 1, &selected, &count);
        if(status != HIPBLAS_STATUS_SUCCESS)
            return status;
        if(count != 1)
            return HIPBLAS_STATUS_NOT_SUPPORTED;
        return first.matmul(&selected.algo);
    };
    if(tuning_replay)
    {
        const char* tmp = std::getenv("TMPDIR");
        tuning.path = std::string(tmp ? tmp : "/tmp") + "/hipblaslt-bias-tuning-XXXXXX";
        const int fd = mkstemp(tuning.path.data());
        ASSERT_GE(fd, 0);
        close(fd);
        char revision[128]{};
        std::string general_kernel;
        const int general_index = fixture_index(root, "bias_present_general", &general_kernel);
        ASSERT_GE(general_index, 0);
        ASSERT_NE(general_index, index);
        hipblaslt_tuning_reset_for_test();
        BiasLaunchCapture default_capture;
        const auto default_status = tuning_launch();
        const auto default_log = default_capture.finish();
        ASSERT_EQ(default_status, HIPBLAS_STATUS_SUCCESS) << default_log;
        bias_launch_identity(default_log, index);
        if(testing::Test::HasFailure())
            return;
        first.numerical(index, name + "_default");
        if(testing::Test::HasFailure())
            return;
        first.reset_output();
        ASSERT_EQ(hipblasLtGetGitRevision(first.handle, revision), HIPBLAS_STATUS_SUCCESS);
        auto write_entry = [&](int selected_index, const std::string& selected_kernel) {
            std::ofstream file(tuning.path);
            ASSERT_TRUE(file.is_open());
            file << "Git Version: " << revision
                 << "\ntransA,transB,batch_count,m,n,k,a_type,b_type,c_type,compute_type,solution_index,kernel_name\n"
                 << "T,N,1,64,64,128,bf16_r,bf16_r,bf16_r,f32_r,"
                 << selected_index << ',' << selected_kernel << '\n';
            file.close();
            ASSERT_TRUE(file.good());
        };
        write_entry(general_index, general_kernel);
        if(name == "bias_tuning_cache_replay")
        {
            ASSERT_EQ(setenv("HIPBLASLT_TUNING_MODE", "cache", 1), 0);
            ASSERT_EQ(setenv("HIPBLASLT_TUNING_CACHE_PATH", tuning.path.c_str(), 1), 0);
        }
        else
            ASSERT_EQ(setenv("HIPBLASLT_TUNING_OVERRIDE_FILE", tuning.path.c_str(), 1), 0);
        hipblaslt_tuning_reset_for_test();
        BiasLaunchCapture consumed;
        const auto consumed_status = tuning_launch();
        const auto consumed_log = consumed.finish();
        ASSERT_EQ(consumed_status, HIPBLAS_STATUS_SUCCESS) << consumed_log;
        bias_launch_identity(consumed_log, general_index);
        if(testing::Test::HasFailure())
            return;
        first.numerical(general_index, name + "_file_selected");
        if(testing::Test::HasFailure())
            return;
        write_entry(index, kernel);
        hipblaslt_tuning_reset_for_test();
        first.reset_output();
    }
    std::vector<hipblasLtMatmulHeuristicResult_t> algorithms;
    std::vector<int> indices{index};
    ASSERT_EQ(hipblaslt_ext::getAlgosFromIndex(first.handle, indices, algorithms), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(algorithms.size(), 1u);
    auto algo = algorithms[0].algo;
    ASSERT_EQ(hipblaslt_ext::getIndexFromAlgo(algo), index);
    ASSERT_EQ(hipblaslt_ext::getKernelNameFromAlgo(first.handle, algo), kernel);

    std::optional<hipblaslt_ext::GroupedGemm> group;
    if(grouped)
    {
        group.emplace(first.handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
                      HIP_R_16BF, HIP_R_16BF, HIP_R_16BF, HIP_R_16BF, HIPBLAS_COMPUTE_32F);
        group->setMaxWorkspaceBytes(BiasProblem::workspace_bytes);
    }
    auto set_group = [&] {
        std::vector<hipblasLtMatmulDesc_t> descriptions{first.desc};
        std::vector<void*> alphas{const_cast<float*>(&first.alpha)}, betas{const_cast<float*>(&first.beta)};
        std::vector<void*> as{first.da.buf()}, bs{first.db.buf()}, cs{first.dc.buf()}, ds{first.dd.buf()};
        std::vector<hipblasLtMatrixLayout_t> las{first.la}, lbs{first.lb}, lcs{first.ld}, lds{first.ld};
        if(mixed)
        {
            descriptions.push_back(second->desc);
            alphas.push_back(const_cast<float*>(&second->alpha));
            betas.push_back(const_cast<float*>(&second->beta));
            as.push_back(second->da.buf()); bs.push_back(second->db.buf());
            cs.push_back(second->dc.buf()); ds.push_back(second->dd.buf());
            las.push_back(second->la); lbs.push_back(second->lb);
            lcs.push_back(second->ld); lds.push_back(second->ld);
        }
        return group->setProblem(descriptions, alphas, as, las, bs, lbs, betas, cs, lcs, ds, lds);
    };
    std::optional<hipblaslt_ext::Gemm> gemm;
    if(name.find("ext") != std::string::npos)
    {
        gemm.emplace(first.handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
                     HIP_R_16BF, HIP_R_16BF, HIP_R_16BF, HIP_R_16BF, HIPBLAS_COMPUTE_32F);
        gemm->setMaxWorkspaceBytes(BiasProblem::workspace_bytes);
    }
    bool grouped_verified = false;
    auto run_group = [&]() -> hipblasStatus_t {
        auto status = group->initialize(algo, first.workspace.buf(), device_args, nullptr);
        if(status != HIPBLAS_STATUS_SUCCESS)
            return status;
        if(!device_args)
            return group->run(nullptr);
        if(!grouped_verified)
            return HIPBLAS_STATUS_NOT_SUPPORTED;
        std::vector<hipblaslt_ext::UserArguments> arguments(mixed ? 2 : 1);
        status = group->getDefaultValueForDeviceUserArguments(arguments.data());
        if(status != HIPBLAS_STATUS_SUCCESS)
            return status;
        HipDeviceBuffer device_arguments(HIP_R_8I, arguments.size() * sizeof(arguments[0]));
        if(device_arguments.memcheck() != hipSuccess)
            return HIPBLAS_STATUS_ALLOC_FAILED;
        if(hipMemcpy(device_arguments.buf(), arguments.data(), device_arguments.getNumBytes(),
                     hipMemcpyHostToDevice) != hipSuccess)
            return HIPBLAS_STATUS_EXECUTION_FAILED;
        status = group->run(device_arguments.buf(), nullptr);
        if(hipDeviceSynchronize() != hipSuccess)
            return HIPBLAS_STATUS_EXECUTION_FAILED;
        return status;
    };
    if(device_args)
    {
        ASSERT_EQ(set_group(), HIPBLAS_STATUS_SUCCESS);
        std::vector<hipblasLtMatmulHeuristicResult_t> grouped_algorithms;
        ASSERT_EQ(hipblaslt_ext::getAllAlgos(first.handle, hipblaslt_ext::GemmType::HIPBLASLT_GROUPED_GEMM,
                      HIPBLAS_OP_T, HIPBLAS_OP_N, HIP_R_16BF, HIP_R_16BF, HIP_R_16BF, HIP_R_16BF,
                      HIPBLAS_COMPUTE_32F, grouped_algorithms), HIPBLAS_STATUS_SUCCESS);
        ASSERT_TRUE(std::any_of(grouped_algorithms.begin(), grouped_algorithms.end(), [&](auto& candidate) {
            return hipblaslt_ext::getIndexFromAlgo(candidate.algo) == index;
        }));
        auto checked = algo;
        size_t required = 0;
        ASSERT_EQ(group->isAlgoSupported(checked, required), HIPBLAS_STATUS_SUCCESS);
        grouped_verified = true;
    }
    auto launch = [&]() -> hipblasStatus_t {
        if(grouped)
        {
            auto status = set_group();
            return status == HIPBLAS_STATUS_SUCCESS ? run_group() : status;
        }
        if(name.find("ext") != std::string::npos)
        {
            auto status = first.set_problem(*gemm);
            if(status == HIPBLAS_STATUS_SUCCESS)
                status = gemm->initialize(algo, first.workspace.buf(), true, nullptr);
            return status == HIPBLAS_STATUS_SUCCESS ? gemm->run(nullptr) : status;
        }
        if(tuning_replay)
            return tuning_launch();
        return first.matmul(replay ? nullptr : &algo);
    };
    if(stock)
        first.set_bias(false);
    BiasLaunchCapture positive;
    const auto positive_status = launch();
    const auto positive_log = positive.finish();
    ASSERT_EQ(positive_status, HIPBLAS_STATUS_SUCCESS) << positive_log;
    first.numerical(index, name);
    if(mixed)
        second->numerical(index, name);
    if(!grouped)
        bias_launch_identity(positive_log, index);
    else
        ASSERT_EQ(group->getKernelName(), kernel);
    if(testing::Test::HasFailure() || stock)
        return;
    if(query)
    {
        size_t required = 0;
        if(name == "bias_support_matmul")
            ASSERT_EQ(hipblaslt_ext::matmulIsAlgoSupported(first.handle, first.desc, &first.alpha,
                         first.la, first.lb, &first.beta, first.ld, first.ld, algo, required), HIPBLAS_STATUS_SUCCESS);
        else if(grouped)
            ASSERT_EQ(group->isAlgoSupported(algo, required), HIPBLAS_STATUS_SUCCESS);
        else
        {
            ASSERT_EQ(first.set_problem(*gemm), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(gemm->isAlgoSupported(algo, required), HIPBLAS_STATUS_SUCCESS);
        }
    }
    if(mixed && name.find("last") != std::string::npos)
        second->set_bias(false);
    else
        first.set_bias(false);
    if(name == "bias_null_pointer" || name == "bias_null_pointer_ext")
        first.set_bias(true, true);
    if(name == "bias_type_mismatch" || name == "bias_type_mismatch_ext")
        first.set_bias(true, false, HIP_R_16BF);
    first.reset_output();
    if(mixed)
        second->reset_output();
    hipblasStatus_t status;
    BiasLaunchCapture probe;
    if(query)
    {
        size_t required = 0;
        if(name == "bias_support_matmul")
            status = hipblaslt_ext::matmulIsAlgoSupported(first.handle, first.desc, &first.alpha,
                         first.la, first.lb, &first.beta, first.ld, first.ld, algo, required);
        else if(grouped)
        {
            ASSERT_EQ(set_group(), HIPBLAS_STATUS_SUCCESS);
            status = group->isAlgoSupported(algo, required);
        }
        else
        {
            ASSERT_EQ(first.set_problem(*gemm), HIPBLAS_STATUS_SUCCESS);
            status = gemm->isAlgoSupported(algo, required);
        }
    }
    else if(initialized)
    {
        // run() uses the arguments bound by the last initialize(), not setProblem().
        ASSERT_EQ(first.set_problem(*gemm), HIPBLAS_STATUS_SUCCESS);
        status = gemm->run(nullptr);
    }
    else if(name == "bias_pinned_ext_setproblem")
    {
        ASSERT_EQ(first.set_problem(*gemm), HIPBLAS_STATUS_SUCCESS);
        status = gemm->initialize(algo, first.workspace.buf(), true, nullptr);
        if(status == HIPBLAS_STATUS_SUCCESS)
        {
            ASSERT_EQ(first.set_problem(*gemm), HIPBLAS_STATUS_SUCCESS);
            status = gemm->run(nullptr);
        }
    }
    else
        status = launch();
    const auto probe_log = probe.finish();
    hipblaslt_cout << "probe route=" << name << " status=" << status << std::endl;
    if(initialized)
    {
        ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
        bias_launch_identity(probe_log, index);
        first.numerical(index, name, true);
        return;
    }
    if(replay)
    {
        ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
        bias_launch_identity(probe_log, fixture_index(root, "bias_present_general"));
        first.numerical(fixture_index(root, "bias_present_general"), name);
    }
    else
    {
        ASSERT_EQ(status, HIPBLAS_STATUS_INVALID_VALUE);
        first.unchanged();
        if(mixed)
            second->unchanged();
    }
}
