/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

// utils/def.hpp enables assert even with NDEBUG, but only until the next
// inclusion of <assert.h>. Include everything else before the library headers.
#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#ifdef SUPPORT_HIP
#include <hip/hip_runtime_api.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __linux__
#include <sched.h>
#endif

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#include "base/backend_manager.hpp"
#include "base/base_matrix.hpp"
#include "base/base_vector.hpp"
#include "base/host/host_matrix_csr.hpp"
#include "base/host/host_vector.hpp"
#include "base/matrix_formats.hpp"

namespace rocalution_unit
{
    // SIGABRT handler for death tests. Exiting instead of aborting keeps the system
    // crash handler (apport, WER) out of every death test, and runs the exit handlers
    // that write the coverage profile of every module. assert() raises SIGABRT in
    // the failing thread, so this is like calling exit() at the failed assert.
    [[noreturn]] inline void exit_on_abort(int)
    {
        std::exit(1);
    }

    // Death tests have no timeout of their own. A statement that hangs, for example
    // in the exit handlers after an assert, ends with exit code 124, which no death
    // test expects, so the death test fails.
    inline void death_test_watchdog(unsigned timeout_s = 120)
    {
#ifndef _WIN32
        std::signal(SIGALRM, [](int) {
            static const char message[] = "death test timed out\n";
            (void)!write(STDERR_FILENO, message, sizeof(message) - 1);
            _exit(124);
        });
        alarm(timeout_s);
#else
        (void)timeout_s;
#endif
    }

    // Death tests match only stderr, but the library logs to stdout
    inline void stdout_to_stderr()
    {
        std::fflush(stdout);
#ifdef _WIN32
        _dup2(_fileno(stderr), _fileno(stdout));
#else
        dup2(STDERR_FILENO, STDOUT_FILENO);
#endif
    }
    using rocalution::BaseMatrix;
    using rocalution::BaseVector;
    using rocalution::HostMatrix;
    using rocalution::HostMatrixCSR;
    using rocalution::HostVector;

    enum class Backend
    {
        Host,
        Accelerator
    };

    inline std::string backend_name(Backend b)
    {
        return b == Backend::Host ? "Host" : "Accelerator";
    }

    inline const rocalution::Rocalution_Backend_Descriptor& descriptor()
    {
        return *rocalution::_get_backend_descriptor();
    }

    inline bool available(Backend b)
    {
        return b == Backend::Host || rocalution::_rocalution_available_accelerator();
    }

    inline bool hip_device_present()
    {
#ifdef SUPPORT_HIP
        int n = 0;
        return hipGetDeviceCount(&n) == hipSuccess && n > 0;
#else
        return false;
#endif
    }

    // Skips without a HIP build or device, but fails if a device is present and
    // the accelerator is not initialized, so a broken init cannot pass as skipped
#define REQUIRE_BACKEND(backend)                                                         \
    if(!rocalution_unit::available(backend))                                             \
    {                                                                                    \
        if(rocalution_unit::hip_device_present())                                        \
        {                                                                                \
            FAIL() << "a HIP device is present, but the accelerator is not initialized"; \
        }                                                                                \
        GTEST_SKIP() << "no accelerator available";                                      \
    }

    // Library preconditions are asserts. exit_on_abort turns the abort into exit(1),
    // so another exit code, such as the one of the watchdog, fails the test.
#define EXPECT_ASSERT_DEATH(statement, regex)                     \
    EXPECT_EXIT(                                                  \
        {                                                         \
            rocalution_unit::death_test_watchdog();               \
            std::signal(SIGABRT, rocalution_unit::exit_on_abort); \
            statement;                                            \
        },                                                        \
        testing::ExitedWithCode(1),                               \
        regex)

    // The statement ends the program with exit(1), and the log matches regex
#define EXPECT_LOGGED_EXIT(statement, regex)        \
    EXPECT_EXIT(                                    \
        {                                           \
            rocalution_unit::death_test_watchdog(); \
            rocalution_unit::stdout_to_stderr();    \
            statement;                              \
        },                                          \
        testing::ExitedWithCode(1),                 \
        regex)

    // Unsupported configurations end the program through FATAL_ERROR. file is a
    // regex for the source file that raises it, such as "hip_vector\\.cpp".
#define EXPECT_FATAL_ERROR(statement, file) \
    EXPECT_LOGGED_EXIT(statement,           \
                       "Fatal error - the program will be terminated.*File: [^;]*" file ";")

    // ASSERT_TRUE for one case of a table-driven loop: fails the test, but ends only
    // the current case, so that the remaining cases still run
#define ASSERT_IN_LOOP(condition)                                     \
    if(!(condition))                                                  \
    {                                                                 \
        ADD_FAILURE() << "Value of: " #condition "\n  Actual: false"; \
        continue;                                                     \
    }                                                                 \
    else                                                              \
        static_cast<void>(0)

    // A file name of the current test and parameter, so that tests can run in parallel
    inline std::string temp_file(const std::string& extension)
    {
        const testing::TestInfo* info = testing::UnitTest::GetInstance()->current_test_info();
        std::string              name
            = std::string(info->test_suite_name()) + "." + info->name() + "." + extension;
        std::replace(name.begin(), name.end(), '/', '_');
        return testing::TempDir() + "rocalution_unit_" + name;
    }

    inline bool run_known_bugs()
    {
        const char* run = std::getenv("ROCALUTION_UNIT_TEST_RUN_KNOWN_BUGS");
        return run != nullptr && std::string(run) == "1";
    }

    // A case broken by a bug of rocALUTION or of a library it calls. Returns from
    // the enclosing function, which is the rest of the test, or only the current
    // call of a lambda (for example one value type of for_each_type). So it comes
    // after the checks that the bug does not affect.
    // ROCALUTION_UNIT_TEST_RUN_KNOWN_BUGS=1 runs the case, to check whether the bug
    // is fixed.
#define KNOWN_BUG(reason)                 \
    if(rocalution_unit::run_known_bugs()) \
    {                                     \
    }                                     \
    else                                  \
        GTEST_SKIP() << "known bug: " << reason

    // Restores the backend descriptor, the OpenMP thread count and the CPU
    // affinity on destruction, also after fatal failures and exceptions. Setting
    // the thread count pins the calling thread to cores, and the threads it
    // creates inherit that.
    class DescriptorGuard
    {
    public:
        DescriptorGuard()
            : saved_(descriptor())
        {
#ifdef _OPENMP
            omp_threads_ = omp_get_max_threads();
#endif
#ifdef __linux__
            affinity_saved_ = sched_getaffinity(0, sizeof(affinity_), &affinity_) == 0;
#endif
        }

        ~DescriptorGuard()
        {
            rocalution::_set_backend_descriptor(saved_);
#ifdef _OPENMP
            omp_set_num_threads(omp_threads_);
#endif
#ifdef __linux__
            if(affinity_saved_)
            {
                // In every thread of the OpenMP pool, as some were created pinned
#ifdef _OPENMP
#pragma omp parallel
#endif
                sched_setaffinity(0, sizeof(affinity_), &affinity_);
            }
#endif
        }

        DescriptorGuard(const DescriptorGuard&)            = delete;
        DescriptorGuard& operator=(const DescriptorGuard&) = delete;

    private:
        rocalution::Rocalution_Backend_Descriptor saved_;
        int                                       omp_threads_ = 1;
#ifdef __linux__
        cpu_set_t affinity_{};
        bool      affinity_saved_ = false;
#endif
    };

    struct Format
    {
        unsigned int id;
        int          blockdim;
    };

    inline std::string format_name(Format f)
    {
        return rocalution::_matrix_format_names[f.id];
    }

    // The formats in which a backend implements an operation, for the operations
    // that are not implemented everywhere, and the formats the tests use. In the
    // other formats the operation returns false and leaves its outputs untouched,
    // so that LocalMatrix can convert to CSR or move to the host and retry.
    inline bool implements(Backend b, unsigned int format, const std::string& operation)
    {
        using rocalution::BCSR;
        using rocalution::COO;
        using rocalution::CSR;
        using rocalution::DENSE;
        using rocalution::DIA;
        using rocalution::ELL;
        using rocalution::HYB;
        using rocalution::MCSR;

        struct Formats
        {
            std::vector<unsigned int> host;
            std::vector<unsigned int> accelerator;
        };

        const std::vector<unsigned int> all    = {CSR, COO, ELL, DIA, HYB, MCSR, BCSR, DENSE};
        const Formats                   values = {{CSR, COO}, {CSR}};
        const Formats                   order  = {{CSR, COO}, {CSR, COO}};
        const Formats                   csr_host_only   = {{CSR}, {}};
        const Formats                   dense_host_only = {{DENSE}, {}};

        static const std::map<std::string, Formats> table = {
            // ConvertFrom of a CSR matrix into the format, and of the format into CSR
            {"ConvertFromCSR", {all, {CSR, COO, ELL, DIA, HYB, BCSR, DENSE}}},
            {"ConvertToCSR", {all, {CSR, COO, ELL, BCSR, DENSE}}},

            {"ExtractDiagonal", {{CSR}, {CSR}}},

            {"Scale", values},
            {"ScaleDiagonal", values},
            {"ScaleOffDiagonal", values},
            {"AddScalar", values},
            {"AddScalarDiagonal", values},
            {"AddScalarOffDiagonal", values},

            {"Permute", order},
            {"PermuteBackward", {{COO}, {COO}}},
            {"Sort", order},

            {"ILU0Factorize", {{CSR, MCSR}, {CSR, BCSR}}},
            {"LUSolve", {{CSR, MCSR, DENSE}, {CSR, BCSR}}},
            {"LSolve", {{CSR}, {CSR, BCSR}}},
            {"USolve", {{CSR}, {CSR, BCSR}}},
            {"LLSolve", {{CSR}, {CSR, BCSR}}},

            {"LUFactorize", dense_host_only},
            {"QRDecompose", dense_host_only},
            {"QRSolve", dense_host_only},
            {"Invert", dense_host_only},
            {"Householder", dense_host_only},

            {"Gershgorin", csr_host_only},
            {"ReplaceRowVector", {{CSR, DENSE}, {DENSE}}},
            {"ILUTFactorize", csr_host_only},
            {"ILUpFactorizeNumeric", csr_host_only},
            {"CMK", csr_host_only},
            {"RCMK", csr_host_only},
            {"ConnectivityOrder", csr_host_only},
            {"ZeroBlockPermutation", csr_host_only},
            {"SymbolicPower", csr_host_only},
            {"SymbolicMatMatMult", csr_host_only},
            {"CreateFromMap", csr_host_only},
            {"Key", csr_host_only},
            {"ReadFileCSR", csr_host_only},
            {"WriteFileCSR", csr_host_only},

            {"ReadFileRSIO", {all, {}}},
            {"WriteFileRSIO", {all, {}}},
        };

        auto entry = table.find(operation);
        if(entry == table.end())
        {
            ADD_FAILURE() << "no capability entry for " << operation;
            return false;
        }

        const std::vector<unsigned int>& formats
            = b == Backend::Host ? entry->second.host : entry->second.accelerator;
        return std::find(formats.begin(), formats.end(), format) != formats.end();
    }

    // Tests of the behaviour of an operation skip where the backend does not
    // implement it. There the fallback tests check that it returns false and
    // leaves its outputs untouched.
#define REQUIRE_NATIVE(backend, format, operation)                                             \
    if(!rocalution_unit::implements(backend, format, operation))                               \
    {                                                                                          \
        GTEST_SKIP() << operation << " is not implemented for "                                \
                     << rocalution_unit::backend_name(backend) << " "                          \
                     << rocalution::_matrix_format_names[format] << ", see the fallback test"; \
    }

    template <typename T = double>
    std::unique_ptr<BaseMatrix<T>> empty_matrix(Backend b, Format f)
    {
        if(b == Backend::Host)
        {
            return std::unique_ptr<BaseMatrix<T>>(
                rocalution::_rocalution_init_base_host_matrix<T>(descriptor(), f.id, f.blockdim));
        }
        return std::unique_ptr<BaseMatrix<T>>(
            rocalution::_rocalution_init_base_backend_matrix<T>(descriptor(), f.id, f.blockdim));
    }

    template <typename T = double>
    std::unique_ptr<HostMatrix<T>> empty_host_matrix(Format f)
    {
        return std::unique_ptr<HostMatrix<T>>(
            rocalution::_rocalution_init_base_host_matrix<T>(descriptor(), f.id, f.blockdim));
    }

    // Host copy of a matrix of any backend, in the same format
    template <typename T>
    std::unique_ptr<HostMatrix<T>> host_copy(const BaseMatrix<T>& A)
    {
        auto host = empty_host_matrix<T>(Format{A.GetMatFormat(), A.GetMatBlockDimension()});
        if(auto accel = dynamic_cast<const rocalution::AcceleratorMatrix<T>*>(&A))
        {
            accel->CopyToHost(host.get());
        }
        else
        {
            host->CopyFrom(A);
        }
        return host;
    }

    // Host copy of a matrix of any backend and format, converted to CSR
    template <typename T>
    std::unique_ptr<HostMatrixCSR<T>> host_csr_of(const BaseMatrix<T>& A)
    {
        auto host = host_copy(A);
        auto csr  = std::make_unique<HostMatrixCSR<T>>(descriptor());
        EXPECT_TRUE(csr->ConvertFrom(*host));
        return csr;
    }

    template <typename T>
    std::unique_ptr<HostVector<T>> host_vector(const std::vector<T>& values)
    {
        auto v = std::make_unique<HostVector<T>>(descriptor());
        v->Allocate(values.size());
        if(!values.empty())
        {
            v->CopyFromData(values.data());
        }
        return v;
    }

    // Backend vector holding the given values
    template <typename T>
    std::unique_ptr<BaseVector<T>> make_vector(Backend b, const std::vector<T>& values)
    {
        auto host = host_vector(values);
        if(b == Backend::Host)
        {
            return host;
        }

        std::unique_ptr<rocalution::AcceleratorVector<T>> accel(
            rocalution::_rocalution_init_base_backend_vector<T>(descriptor()));
        accel->Allocate(values.size());
        accel->CopyFromHost(*host);
        return accel;
    }

    template <typename T>
    std::vector<T> values_of(const BaseVector<T>& v)
    {
        HostVector<T> host(descriptor());
        host.Allocate(v.GetSize());

        if(auto accel = dynamic_cast<const rocalution::AcceleratorVector<T>*>(&v))
        {
            accel->CopyToHost(&host);
        }
        else
        {
            host.CopyFrom(v);
        }

        std::vector<T> values(v.GetSize());
        if(!values.empty())
        {
            host.CopyToData(values.data());
        }
        return values;
    }

    // Backend matrix in format f, converted on the host from a CSR matrix. Fails
    // without aborting the test, so that rejected conversions can be tested.
    template <typename T>
    testing::AssertionResult try_make_matrix(Backend                         b,
                                             Format                          f,
                                             const HostMatrixCSR<T>&         csr,
                                             std::unique_ptr<BaseMatrix<T>>* out)
    {
        std::unique_ptr<HostMatrix<T>> host(
            rocalution::_rocalution_init_base_host_matrix<T>(descriptor(), f.id, f.blockdim));
        if(!host->ConvertFrom(csr))
        {
            return testing::AssertionFailure()
                   << "host conversion from CSR to " << format_name(f) << " failed";
        }

        if(b == Backend::Host)
        {
            *out = std::move(host);
            return testing::AssertionSuccess();
        }

        std::unique_ptr<rocalution::AcceleratorMatrix<T>> accel(
            rocalution::_rocalution_init_base_backend_matrix<T>(descriptor(), f.id, f.blockdim));
        accel->CopyFromHost(*host);
        if(accel->GetMatFormat() != f.id || accel->GetNnz() != host->GetNnz())
        {
            return testing::AssertionFailure()
                   << "copy of " << format_name(f) << " to the accelerator has format "
                   << rocalution::_matrix_format_names[accel->GetMatFormat()] << " and "
                   << accel->GetNnz() << " instead of " << host->GetNnz() << " non-zeros";
        }

        *out = std::move(accel);
        return testing::AssertionSuccess();
    }

    // As try_make_matrix, but a failure is a test failure. Returns nullptr then.
    template <typename T>
    std::unique_ptr<BaseMatrix<T>> make_matrix(Backend b, Format f, const HostMatrixCSR<T>& csr)
    {
        std::unique_ptr<BaseMatrix<T>> A;
        EXPECT_TRUE(try_make_matrix(b, f, csr, &A));
        return A;
    }

    // n x m matrix in CSR format, square unless m is given, with a reference
    // matrix-vector product
    template <typename T>
    struct Csr
    {
        explicit Csr(int n, int m = -1)
            : n(n)
            , m(m < 0 ? n : m)
        {
            row_offset.push_back(0);
        }

        void add(int j, T v)
        {
            col.push_back(j);
            val.push_back(v);
        }

        void end_row()
        {
            row_offset.push_back(static_cast<PtrType>(col.size()));
        }

        int64_t nnz() const
        {
            return static_cast<int64_t>(col.size());
        }

        std::unique_ptr<HostMatrixCSR<T>> host_csr() const
        {
            auto A = std::make_unique<HostMatrixCSR<T>>(descriptor());
            A->AllocateCSR(nnz(), n, m);
            A->CopyFromCSR(row_offset.data(), col.data(), val.data());
            return A;
        }

        std::vector<T> apply(const std::vector<T>& x) const
        {
            std::vector<T> y(n, static_cast<T>(0));
            for(int i = 0; i < n; ++i)
            {
                for(PtrType j = row_offset[i]; j < row_offset[i + 1]; ++j)
                {
                    y[i] += val[j] * x[col[j]];
                }
            }
            return y;
        }

        int                  n;
        int                  m;
        std::vector<PtrType> row_offset;
        std::vector<int>     col;
        std::vector<T>       val;
    };

    // n x n tridiagonal matrix with off-diagonals -1 and diagonal 2 + i
    template <typename T>
    struct Tridiagonal : Csr<T>
    {
        explicit Tridiagonal(int n)
            : Csr<T>(n)
        {
            for(int i = 0; i < n; ++i)
            {
                for(int j = i - 1; j <= i + 1; ++j)
                {
                    if(j >= 0 && j < n)
                    {
                        this->add(j, j == i ? diagonal(i) : static_cast<T>(-1));
                    }
                }
                this->end_row();
            }
        }

        T diagonal(int i) const
        {
            return static_cast<T>(2 + i);
        }
    };

    // Tridiagonal with a second upper diagonal in the even rows, so that DIA
    // stores zeros inside the matrix and ELL pads every other row
    template <typename T>
    Csr<T> banded_with_gaps(int n)
    {
        Csr<T> A(n);
        for(int i = 0; i < n; ++i)
        {
            for(int j = i - 1; j <= i + 2; ++j)
            {
                if(j >= 0 && j < n && (j < i + 2 || i % 2 == 0))
                {
                    A.add(j, static_cast<T>(10 * i + j + 1));
                }
            }
            A.end_row();
        }
        return A;
    }

    // Dense first and last row with a diagonal in between. The long rows are
    // stored in the COO part of HYB.
    template <typename T>
    Csr<T> arrow(int n)
    {
        Csr<T> A(n);
        for(int i = 0; i < n; ++i)
        {
            for(int j = 0; j < n; ++j)
            {
                if(i == 0 || i == n - 1 || i == j)
                {
                    A.add(j, static_cast<T>(i + j + 1));
                }
            }
            A.end_row();
        }
        return A;
    }

    // ELL and DIA reject the arrow, see ell_and_dia_reject_oversized_matrices
    template <typename T = double>
    std::vector<std::pair<std::string, Csr<T>>> matrices(Format f)
    {
        std::vector<std::pair<std::string, Csr<T>>> m
            = {{"tridiagonal", Tridiagonal<T>(12)}, {"banded with gaps", banded_with_gaps<T>(12)}};
        if(f.id != rocalution::ELL && f.id != rocalution::DIA)
        {
            m.push_back({"arrow", arrow<T>(12)});
        }
        return m;
    }

    // Host copy of a matrix of any backend and format, as Csr
    template <typename T>
    Csr<T> csr_of(const BaseMatrix<T>& A)
    {
        auto   host = host_csr_of(A);
        Csr<T> B(host->GetM(), host->GetN());
        B.row_offset.resize(B.n + 1);
        B.col.resize(host->GetNnz());
        B.val.resize(host->GetNnz());
        host->CopyToCSR(B.row_offset.data(), B.col.data(), B.val.data());
        return B;
    }

    // With zero_fill, A may hold explicit zeros in addition to the entries of ref
    template <typename T>
    void expect_csr_eq(const HostMatrixCSR<T>& A, const Csr<T>& ref, bool zero_fill)
    {
        ASSERT_EQ(A.GetM(), ref.n);
        ASSERT_EQ(A.GetN(), ref.m);
        if(!zero_fill)
        {
            ASSERT_EQ(A.GetNnz(), ref.nnz());
        }

        std::vector<PtrType> row_offset(ref.n + 1);
        std::vector<int>     col(A.GetNnz());
        std::vector<T>       val(A.GetNnz());
        A.CopyToCSR(row_offset.data(), col.data(), val.data());

        Csr<T> result(ref.n, ref.m);
        for(int i = 0; i < ref.n; ++i)
        {
            for(PtrType j = row_offset[i]; j < row_offset[i + 1]; ++j)
            {
                if(!zero_fill || val[j] != static_cast<T>(0))
                {
                    result.add(col[j], val[j]);
                }
            }
            result.end_row();
        }

        EXPECT_EQ(result.row_offset, ref.row_offset);
        EXPECT_EQ(result.col, ref.col);
        EXPECT_EQ(result.val, ref.val);
    }

    // What f writes to the log
    template <typename F>
    std::string logged(F&& f)
    {
        testing::internal::CaptureStdout();
        f();
        return testing::internal::GetCapturedStdout();
    }

    // Runs the checks of a known bug that ends the process in a child process, so
    // that the bug fails the test instead of ending the suite. The checks build
    // all their objects. A skip inside the checks fails the test, as the child
    // cannot report it. A nonzero timeout_s ends a child that hangs with SIGALRM
    // (not on Windows).
    // gtest does not report the results of a death test child, so the child
    // writes them to stderr, which the failure message of the parent shows.
    inline void expect_in_child(const std::function<void()>& checks, unsigned timeout_s = 0)
    {
        EXPECT_EXIT(
            {
#ifndef _WIN32
                if(timeout_s != 0)
                {
                    alarm(timeout_s);
                }
#endif
                testing::TestPartResultArray results;
                {
                    testing::ScopedFakeTestPartResultReporter reporter(
                        testing::ScopedFakeTestPartResultReporter::INTERCEPT_ALL_THREADS, &results);
                    checks();
                }

                int code = 0;
                for(int i = 0; i < results.size(); ++i)
                {
                    const testing::TestPartResult& r = results.GetTestPartResult(i);
                    std::fprintf(stderr,
                                 "%s:%d: %s\n",
                                 r.file_name() != nullptr ? r.file_name() : "unknown file",
                                 r.line_number(),
                                 r.message());
                    if(r.failed())
                    {
                        code = 1;
                    }
                    else if(r.skipped() && code == 0)
                    {
                        code = 3;
                    }
                }
                std::exit(code);
            },
            testing::ExitedWithCode(0),
            "");
    }

    using BackendFormat = std::tuple<Backend, Format>;

    // Test names such as Accelerator_BCSR2
    inline std::string backend_format_name(const testing::TestParamInfo<BackendFormat>& info)
    {
        const Format f = std::get<1>(info.param);
        return backend_name(std::get<0>(info.param)) + "_" + format_name(f)
               + (f.id == rocalution::BCSR ? std::to_string(f.blockdim) : "");
    }
}
