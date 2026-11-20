/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNEC-
 * TION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

// Loads the Tensile library from a directory whose paths reach the Windows MAX_PATH
// limit (260 characters). rocBLAS reads ROCBLAS_TENSILE_LIBPATH once per process, so
// each case copies the installed library to a deep directory and runs a GEMM in a
// fresh rocblas-test process pointed at it. Off Windows there is no path limit, and
// the cases only check that loading still works.

// System headers come before the client headers, which poison stdio identifiers.
#ifdef WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <process.h>
#include <windows.h>
#else
#include <link.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "client_utility.hpp"
#include "testing_kernel_select.hpp"

#include <gtest/gtest.h>

namespace
{
    namespace stdfs = std::filesystem;

// On Windows these tests are the regression check for the long-path fix, so a missing
// precondition fails there. Elsewhere it only skips.
#ifdef WIN32
#define LONG_PATH_UNAVAILABLE FAIL()
#else
#define LONG_PATH_UNAVAILABLE GTEST_SKIP()
#endif

    constexpr size_t      c_max_path    = 260;
    constexpr const char* c_child_env   = "ROCBLAS_LONG_PATH_CHILD";
    constexpr const char* c_child_test  = "rocblas_long_path_child.DISABLED_gemm";
    constexpr const char* c_root_prefix = "rocblas_long_path_";

    int process_id()
    {
#ifdef WIN32
        return _getpid();
#else
        return getpid();
#endif
    }

    // The \\?\ form of `p`, which the fixture uses to create and remove the deep tree.
    // This is independent of the longPathSafe() code under test.
    stdfs::path verbatim(const stdfs::path& p)
    {
#ifdef WIN32
        std::error_code ec;
        stdfs::path     full = stdfs::absolute(p, ec);
        if(ec)
            return p;
        std::wstring native = full.lexically_normal().native();
        if(native.rfind(L"\\\\?\\", 0) == 0)
            return native;
        if(native.rfind(L"\\\\", 0) == 0)
            return L"\\\\?\\UNC\\" + native.substr(2);
        return L"\\\\?\\" + native;
#else
        return p;
#endif
    }

    // Length of the absolute, normalized form of `p`, as Windows measures it.
    size_t path_length(const stdfs::path& p)
    {
        return stdfs::absolute(p).lexically_normal().native().size();
    }

    // Directory containing the loaded rocBLAS library, or "" if unknown.
    std::string rocblas_module_dir()
    {
#ifdef WIN32
        HMODULE module = GetModuleHandleA("rocblas.dll");
        if(!module)
            return "";
        std::vector<wchar_t> buffer(MAX_PATH + 1);
        for(;;)
        {
            DWORD length = GetModuleFileNameW(module, buffer.data(), DWORD(buffer.size()));
            if(length == 0)
                return "";
            if(length < buffer.size() - 1)
                break;
            buffer.resize(buffer.size() * 2);
        }
        stdfs::path dll(std::wstring(buffer.data()));
        return dll.parent_path().string();
#else
        // Find librocblas the way tensile_host.cpp does.
        std::string so_path;
        dl_iterate_phdr(
            [](dl_phdr_info* info, size_t, void* data) {
                if(info->dlpi_name && strstr(info->dlpi_name, "librocblas."))
                {
                    *static_cast<std::string*>(data) = info->dlpi_name;
                    return 1;
                }
                return 0;
            },
            &so_path);
        return so_path.empty() ? "" : stdfs::path(so_path).parent_path().string();
#endif
    }

    bool has_tensile_library(const stdfs::path& dir)
    {
        std::error_code ec;
        if(!stdfs::is_directory(dir, ec))
            return false;
        for(auto it = stdfs::recursive_directory_iterator(dir, ec);
            !ec && it != stdfs::recursive_directory_iterator();
            it.increment(ec))
        {
            if(it.depth() > 1)
            {
                it.disable_recursion_pending();
                continue;
            }
            const std::string name = it->path().filename().string();
            if(name.rfind("TensileLibrary", 0) == 0 && it->path().extension() == ".dat")
                return true;
        }
        return false;
    }

    // The Tensile library directory rocBLAS loads, found the way tensile_host.cpp
    // searches for it.
    stdfs::path find_tensile_library()
    {
        if(const char* env = getenv("ROCBLAS_TENSILE_LIBPATH"))
            return has_tensile_library(env) ? stdfs::path(env) : stdfs::path();

        std::vector<stdfs::path> candidates;
        const std::string        module_dir = rocblas_module_dir();
        if(!module_dir.empty())
        {
            const stdfs::path base(module_dir);
            for(const char* suffix : {"../../Tensile/library",
                                      "library",
                                      "../rocblas/library",
                                      "../Tensile/library",
                                      "rocblas/library"})
                candidates.push_back(base / suffix);
        }
        candidates.push_back(stdfs::path(rocblas_exepath()) / "rocblas/library");
        candidates.push_back(stdfs::path(rocblas_exepath()) / "../../Tensile/library");

        for(const auto& dir : candidates)
            if(has_tensile_library(dir))
                return stdfs::absolute(dir).lexically_normal();
        return {};
    }

    // Whether `source` holds a Tensile library for `arch`, in any of the layouts
    // tensile_host.cpp searches: an <arch>[-<xnack>] directory or the flat directory.
    bool has_device_library(const stdfs::path& source, const std::string& arch)
    {
        const std::vector<std::string> names
            = {"TensileLibrary_lazy_" + arch + ".dat", "TensileLibrary_" + arch + ".dat"};
        std::error_code ec;
        for(const auto& entry : stdfs::directory_iterator(source, ec))
        {
            if(!entry.is_directory() || entry.path().filename().string().rfind(arch, 0) != 0)
                continue;
            for(const auto& name : names)
                if(stdfs::exists(entry.path() / name, ec))
                    return true;
        }
        for(const auto& name : names)
            if(stdfs::exists(source / name, ec))
                return true;
        return stdfs::exists(source / "TensileLibrary.dat", ec);
    }

    // Base name of the current device's architecture, such as gfx942.
    std::string device_arch()
    {
        int device = 0;
        if(hipGetDevice(&device) != hipSuccess)
            return "";
        hipDeviceProp_t props;
        if(hipGetDeviceProperties(&props, device) != hipSuccess)
            return "";
        std::string arch = props.gcnArchName;
        return arch.substr(0, arch.find(':'));
    }

    std::string read_file(const stdfs::path& p)
    {
        std::ifstream      ifs(p, std::ios::binary);
        std::ostringstream ss;
        ss << ifs.rdbuf();
        return ss.str();
    }

    std::string tail(const std::string& s, size_t n = 4000)
    {
        return s.size() <= n ? s : "..." + s.substr(s.size() - n);
    }

    // Copies the Tensile library at `source` to a new directory whose own path is
    // `root_length` characters long, using hard links where the volume allows. The
    // directory goes next to the executable, which shares its volume with an installed
    // library, or into the temporary directory if that is read-only.
    class deep_library
    {
        stdfs::path m_unique; // the directory this object created and removes
        stdfs::path m_root; // the copy of the library

        static stdfs::path create_unique_directory()
        {
            static int      count = 0;
            std::error_code ec;
            for(const stdfs::path& base :
                {stdfs::path(rocblas_exepath()), stdfs::temp_directory_path()})
            {
                // A directory left by an earlier run with the same process id is skipped.
                for(int attempt = 0; attempt < 100; ++attempt)
                {
                    const stdfs::path dir = base
                                            / (c_root_prefix + std::to_string(process_id()) + "_"
                                               + std::to_string(count++));
                    if(stdfs::create_directory(verbatim(dir), ec))
                        return dir;
                    if(ec)
                        break; // this base is not writable
                }
            }
            throw std::runtime_error("cannot create a " + std::string(c_root_prefix)
                                     + " directory: " + ec.message());
        }

    public:
        deep_library(const stdfs::path& source, const std::string& arch, size_t root_length)
            : m_unique(create_unique_directory())
            , m_root(m_unique)
        {
            try
            {
                // Pad with directories of at most 100 characters up to root_length.
                const size_t base = path_length(m_unique);
                if(root_length > base)
                {
                    // Each level adds a separator plus its name, so it needs at least 2.
                    size_t missing = std::max<size_t>(root_length - base, 2);
                    while(missing > 0)
                    {
                        size_t name = std::min<size_t>(100, missing - 1);
                        if(missing - (name + 1) == 1)
                            --name; // leave room for one more level
                        m_root /= std::string(name, 'd');
                        missing -= name + 1;
                    }
                }
                stdfs::create_directories(verbatim(m_root));

                // Mirror the top-level files and the directories for this architecture.
                for(const auto& entry : stdfs::directory_iterator(source))
                {
                    const std::string name = entry.path().filename().string();
                    if(entry.is_regular_file())
                        link_or_copy(entry.path(), m_root / name);
                    else if(entry.is_directory() && name.rfind(arch, 0) == 0)
                    {
                        stdfs::create_directories(verbatim(m_root / name));
                        for(const auto& file : stdfs::directory_iterator(entry.path()))
                            if(file.is_regular_file())
                                link_or_copy(file.path(), m_root / name / file.path().filename());
                    }
                }
            }
            catch(...)
            {
                remove();
                throw;
            }
        }

        ~deep_library()
        {
            remove();
        }

        deep_library(const deep_library&)            = delete;
        deep_library& operator=(const deep_library&) = delete;

        const stdfs::path& root() const
        {
            return m_root;
        }

    private:
        void remove()
        {
            if(m_unique.filename().string().rfind(c_root_prefix, 0) != 0)
                return;
            std::error_code ec;
            stdfs::remove_all(verbatim(m_unique), ec);
        }

        static void link_or_copy(const stdfs::path& from, const stdfs::path& to)
        {
            std::error_code ec;
            stdfs::create_hard_link(from, verbatim(to), ec);
            if(ec)
            {
                ec.clear();
                stdfs::copy_file(from, verbatim(to), ec);
                if(ec)
                    throw std::runtime_error("cannot copy " + from.string() + ": " + ec.message());
            }
        }
    };

    // Shortest name of a library file the f32 NN GEMM can open in `arch_dir`.
    size_t shortest_sgemm_library_name(const stdfs::path& arch_dir)
    {
        size_t shortest = 0;
        for(const auto& file : stdfs::directory_iterator(arch_dir))
        {
            const std::string name = file.path().filename().string();
            if(name.find("_Type_SS_Contraction_l_Ailk_Bljk_Cijk_Dijk") != std::string::npos
               && file.path().extension() == ".dat")
                shortest = shortest ? std::min<size_t>(shortest, name.size()) : name.size();
        }
        return shortest;
    }

    std::string quote_arg(const std::string& s)
    {
        return "\"" + s + "\"";
    }

    // Writes the header of the default test data, with no test cases, to `path`. The
    // child gets this instead of the full data file, which takes minutes to parse.
    void write_empty_test_data(const stdfs::path& path)
    {
        // "rocBLAS", one signature Arguments record, "ROCblas"; see Arguments::validate.
        constexpr size_t  header_size = 8 + sizeof(Arguments) + 8;
        std::vector<char> header(header_size);
        std::ifstream     in(rocblas_exepath() + "rocblas_gtest.data", std::ios::binary);
        if(!in.read(header.data(), header.size()))
            throw std::runtime_error("cannot read the rocblas_gtest.data header");
        std::ofstream out(path, std::ios::binary);
        if(!out.write(header.data(), header.size()))
            throw std::runtime_error("cannot write " + path.string());
    }

    struct child_result
    {
        int         status;
        std::string output;
        std::string trace;
        std::string report; // the child's gtest XML report
    };

    // Runs c_child_test in a new rocblas-test process with ROCBLAS_TENSILE_LIBPATH
    // pointing at `library`.
    child_result run_child(const stdfs::path& library, const char* mode)
    {
        static int        count = 0;
        const std::string stem  = std::string(c_root_prefix) + std::to_string(process_id()) + "_"
                                 + std::to_string(count++);
        const stdfs::path out   = stdfs::temp_directory_path() / (stem + ".out");
        const stdfs::path trace = stdfs::temp_directory_path() / (stem + ".trace");
        const stdfs::path data  = stdfs::temp_directory_path() / (stem + ".data");
        const stdfs::path shard = stdfs::temp_directory_path() / (stem + ".shard");
        const stdfs::path xml   = stdfs::temp_directory_path() / (stem + ".xml");
        write_empty_test_data(data);

        std::string exe = rocblas_exepath() + "rocblas-test";
#ifdef WIN32
        exe += ".exe";
#endif
        std::string command = quote_arg(exe) + " --data " + quote_arg(data.string())
                              + " --gtest_also_run_disabled_tests --gtest_filter=" + c_child_test
                              + " > " + quote_arg(out.string()) + " 2>&1";
#ifdef WIN32
        // cmd /c strips the outer pair of quotes, so wrap the whole command.
        command = quote_arg(command);
#endif

        child_result result;
        {
            kernel_select::scoped_env libpath("ROCBLAS_TENSILE_LIBPATH", library.string());
            kernel_select::scoped_env child(c_child_env, mode);
            kernel_select::scoped_env layer("ROCBLAS_LAYER", "0x10");
            kernel_select::scoped_env trace_path("ROCBLAS_LOG_TRACE_PATH", trace.string());
            kernel_select::scoped_env tensile("ROCBLAS_USE_HIPBLASLT", "0");
            // CI shards rocblas-test through the environment. The child's one test is
            // shard 0 of its own run, so turn sharding off, and keep the child's
            // reports away from the parent's.
            kernel_select::scoped_env total_shards("GTEST_TOTAL_SHARDS", "1");
            kernel_select::scoped_env shard_index("GTEST_SHARD_INDEX", "0");
            kernel_select::scoped_env shard_status("GTEST_SHARD_STATUS_FILE", shard.string());
            kernel_select::scoped_env output("GTEST_OUTPUT", "xml:" + xml.string());
            kernel_select::scoped_env repeat("GTEST_REPEAT", "1");
            kernel_select::scoped_env color("GTEST_COLOR", "no");
            kernel_select::scoped_env brief("GTEST_BRIEF", "0");
            result.status = std::system(command.c_str());
        }
        result.output = read_file(out);
        result.trace  = read_file(trace);
        result.report = read_file(xml);

        std::error_code ec;
        stdfs::remove(out, ec);
        stdfs::remove(trace, ec);
        stdfs::remove(data, ec);
        stdfs::remove(shard, ec);
        stdfs::remove(xml, ec);
        return result;
    }

    // Checks that the child ran the GEMM through Tensile with no library error.
    void expect_child_passed(const child_result& r)
    {
        EXPECT_EQ(r.status, 0) << tail(r.output);

        // The child's report must show its one test run to completion, not skipped.
        const size_t      root_end = r.report.find('>', r.report.find("<testsuites"));
        const std::string root = root_end == std::string::npos ? "" : r.report.substr(0, root_end);
        const bool        ran  = root.find(" tests=\"1\"") != std::string::npos
                         && root.find(" failures=\"0\"") != std::string::npos
                         && root.find(" errors=\"0\"") != std::string::npos
                         && r.report.find("result=\"completed\"") != std::string::npos
                         && r.report.find("<skipped") == std::string::npos;
        EXPECT_TRUE(ran) << "the child test did not run and pass:\n"
                         << tail(r.report) << "\n"
                         << tail(r.output);
        EXPECT_EQ(r.output.find("Cannot read"), std::string::npos) << tail(r.output);
        EXPECT_EQ(r.output.find("No paths matched"), std::string::npos) << tail(r.output);
        EXPECT_NE(r.trace.find("# source=tensile"), std::string::npos)
            << "the GEMM did not run a Tensile kernel:\n"
            << tail(r.trace) << "\n"
            << tail(r.output);
    }

    class rocblas_long_path : public ::testing::Test
    {
    protected:
        stdfs::path m_source;
        std::string m_arch;

        void SetUp() override
        {
            m_source = find_tensile_library();
            m_arch   = device_arch();
            if(m_source.empty() || m_arch.empty() || !has_device_library(m_source, m_arch))
                LONG_PATH_UNAVAILABLE << "no Tensile library for the device architecture '"
                                      << m_arch << "' in '" << m_source.string() << "'";
        }
    };

    // The case reported in issue 15596: the library directory and its lazy catalog stay
    // under MAX_PATH, and only the per-type library files reach it.
    TEST_F(rocblas_long_path, quick_tensile_libpath_longest_files)
    {
        const stdfs::path arch_dir = m_source / m_arch;
        const size_t      shortest
            = stdfs::is_directory(arch_dir) ? shortest_sgemm_library_name(arch_dir) : 0;
        const std::string lazy = "TensileLibrary_lazy_" + m_arch + ".dat";
        if(!shortest || !stdfs::exists(arch_dir / lazy))
            LONG_PATH_UNAVAILABLE << "no lazy " << m_arch << " library in " << m_source.string();

        // Put the shortest sgemm library file a few characters past the limit.
        const size_t root = c_max_path + 4 - (2 + m_arch.size() + shortest);
        deep_library library(m_source, m_arch, root);
        if(path_length(library.root() / m_arch) + 1 + shortest < c_max_path
           || path_length(library.root() / m_arch / lazy) >= c_max_path)
            LONG_PATH_UNAVAILABLE << "cannot build the layout: the base directory of "
                                  << library.root().string() << " is too long";

        expect_child_passed(run_child(library.root(), "lazy"));
    }

    // The whole library directory past MAX_PATH, loaded lazily.
    TEST_F(rocblas_long_path, quick_tensile_libpath_lazy)
    {
        deep_library library(m_source, m_arch, c_max_path + 40);
        expect_child_passed(run_child(library.root(), "lazy"));
    }

    // The whole library directory past MAX_PATH, loaded eagerly by rocblas_initialize().
    TEST_F(rocblas_long_path, quick_tensile_libpath_eager)
    {
        deep_library library(m_source, m_arch, c_max_path + 40);
        expect_child_passed(run_child(library.root(), "eager"));
    }
} // namespace

// Runs only inside the processes started above, which pass --gtest_also_run_disabled_tests
// and set c_child_env.
TEST(rocblas_long_path_child, DISABLED_gemm)
{
    const char* mode = getenv(c_child_env);
    if(!mode)
        GTEST_SKIP() << "started only by the rocblas_long_path tests";
    if(std::string(mode) == "eager")
        rocblas_initialize();

    constexpr rocblas_int n = 64;
    host_matrix<float>    hA(n, n, n), hB(n, n, n), hC(n, n, n);
    device_matrix<float>  dA(n, n, n), dB(n, n, n), dC(n, n, n);
    ASSERT_EQ(dA.memcheck(), hipSuccess);
    ASSERT_EQ(dB.memcheck(), hipSuccess);
    ASSERT_EQ(dC.memcheck(), hipSuccess);

    // Small integers keep the float products exact.
    float* A = hA[0];
    float* B = hB[0];
    for(rocblas_int i = 0; i < n * n; ++i)
    {
        A[i] = float(i % 7 - 3);
        B[i] = float(i % 5 - 2);
    }
    std::vector<float> expected(n * n, 0.0f);
    for(rocblas_int j = 0; j < n; ++j)
        for(rocblas_int k = 0; k < n; ++k)
            for(rocblas_int i = 0; i < n; ++i)
                expected[i + j * n] += A[i + k * n] * B[k + j * n];
    ASSERT_EQ(dA.transfer_from(hA), hipSuccess);
    ASSERT_EQ(dB.transfer_from(hB), hipSuccess);

    rocblas_local_handle handle;
    const float          alpha = 1.0f, beta = 0.0f;
    ASSERT_EQ(rocblas_sgemm(handle,
                            rocblas_operation_none,
                            rocblas_operation_none,
                            n,
                            n,
                            n,
                            &alpha,
                            dA,
                            n,
                            dB,
                            n,
                            &beta,
                            dC,
                            n),
              rocblas_status_success);
    ASSERT_EQ(hC.transfer_from(dC), hipSuccess);
    EXPECT_EQ(std::vector<float>(hC[0], hC[0] + n * n), expected);
}
