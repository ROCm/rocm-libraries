/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#ifndef GUARD_MIOPEN_FIND_DB_HPP_
#define GUARD_MIOPEN_FIND_DB_HPP_

#include <miopen/config.h>
#include <miopen/db.hpp>
#include <miopen/db_record.hpp>
#include <miopen/env.hpp>
#include <miopen/perf_field.hpp>
#include <miopen/ramdb.hpp>
#include <miopen/readonlyramdb.hpp>
#include <miopen/solution.hpp>
#include <miopen/conv/solver_finders.hpp>

#include <algorithm>
#include <functional>
#include <optional>
#include <vector>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_DISABLE_FIND_DB)

namespace miopen {

struct Handle;
struct NetworkConfig;

template <class TDb>
class FindDbRecord_t;

#if MIOPEN_DEBUG_FIND_DB_CACHING
using SystemFindDb = ReadonlyRamDb;
using UserFindDb   = RamDb;
#else
using SystemFindDb = PlainTextDb;
using UserFindDb   = PlainTextDb;
#endif

using FindDb           = MultiFileDb<SystemFindDb, UserFindDb, false>;
using FindDbRecord     = FindDbRecord_t<FindDb>;
using UserFindDbRecord = FindDbRecord_t<UserFindDb>;

namespace debug {

// For unit tests.
MIOPEN_EXPORT extern bool
    testing_find_db_enabled; // NOLINT (cppcoreguidelines-avoid-non-const-global-variables)
MIOPEN_EXPORT extern std::optional<fs::path>&
testing_find_db_path_override(); /// \todo Remove when #1723 is resolved.

} // namespace debug

template <class TDb>
class FindDbRecord_t
{
private:
    template <class TTestDb>
    using is_find_t = std::enable_if_t<std::is_same<TTestDb, UserFindDb>::value, int>;

    template <class TTestDb>
    using is_immediate_t = std::enable_if_t<std::is_same<TTestDb, FindDb>::value, int>;

public:
    FindDbRecord_t(const FindDbRecord_t&)            = delete;
    FindDbRecord_t& operator=(const FindDbRecord_t&) = delete;

    template <class TProblemDescription, class TTestDb = TDb>
    FindDbRecord_t(const Handle& handle,
                   const TProblemDescription& problem,
                   const std::string& path_suffix = "",
                   is_immediate_t<TTestDb>        = 0)
        // NOLINTBEGIN (bugprone-unchecked-optional-access)
        : path(debug::testing_find_db_path_override().has_value()
                   ? debug::testing_find_db_path_override().value()
                   : GetUserPath(handle, path_suffix)),
          installed_path(debug::testing_find_db_path_override().has_value()
                             ? debug::testing_find_db_path_override().value()
                             : GetInstalledPath(handle, path_suffix)),
          db(construct_db(debug::testing_find_db_enabled &&
                              !env::enabled(MIOPEN_DEBUG_DISABLE_FIND_DB),
                          DbTimer<TDb>{DbKinds::FindDb, installed_path, path}))
    // NOLINTEND (bugprone-unchecked-optional-access)
    {
        if(!db)
            return;

        content = db->FindRecord(problem);
        in_sync = content.has_value();
    }

    template <class TProblemDescription, class TTestDb = TDb>
    FindDbRecord_t(const Handle& handle,
                   const TProblemDescription& problem,
                   const std::string& path_suffix = "",
                   is_find_t<TTestDb>             = 0)
        // NOLINTBEGIN (bugprone-unchecked-optional-access)
        : path(debug::testing_find_db_path_override().has_value()
                   ? debug::testing_find_db_path_override().value()
                   : GetUserPath(handle, path_suffix)),
#if MIOPEN_DISABLE_USERDB
          db(std::optional<DbTimer<TDb>>{DbKinds::FindDb})
#else
          db(construct_db(debug::testing_find_db_enabled &&
                              !env::enabled(MIOPEN_DEBUG_DISABLE_FIND_DB),
                          DbTimer<TDb>{DbKinds::FindDb, path, false}))
#endif
    // NOLINTEND (bugprone-unchecked-optional-access)
    {
        if(!db)
            return;

        content = db->FindRecord(problem);
        in_sync = content.has_value();
    }

    ~FindDbRecord_t()
    {
        if(dont_store || !db || !content || in_sync)
            return;
        if(!db->StoreRecord(content.value()))
            MIOPEN_LOG_E("Failed to store record to find-db at <" << path << ">");
    }

    // NOLINTBEGIN (bugprone-unchecked-optional-access)
    auto begin() const { return content.value().As<FindDbData>().begin(); }
    auto begin() { return content.value().As<FindDbData>().begin(); }
    auto end() const { return content.value().As<FindDbData>().end(); }
    auto end() { return content.value().As<FindDbData>().end(); }
    // NOLINTEND (bugprone-unchecked-optional-access)
    bool empty() const { return !content.has_value(); }

    template <class TProblemDescription>
    static std::vector<Solution> TryLoad(const Handle& handle,
                                         const TProblemDescription& problem,
                                         const std::function<FindCoreResult()>& regenerator,
                                         const std::string& path_suffix            = "",
                                         const AnyInvokeParams* const invokeParams = nullptr)
    {
        FindDbRecord_t<TDb> record{handle, problem, path_suffix};

        const auto network_config = problem.MakeNetworkConfig();

        if(record.in_sync && !record.Validate(handle, network_config))
        {
            auto solutions = std::vector<Solution>{};
            record.CopyTo(solutions);

            // The cached record may contain solvers that no longer fit the
            // caller's workspace (e.g. when a workspace-range walker invokes
            // Find with a smaller buffer than the one used to populate the
            // record). Drop those before returning, so the dispatch can't
            // pick a solver whose invoker will throw "Not enough workspace".
            if(invokeParams != nullptr)
            {
                solutions.erase(std::remove_if(solutions.begin(),
                                               solutions.end(),
                                               [&](const Solution& s) {
                                                   return !conv::IsEnoughWorkspace(
                                                       "TryLoad",
                                                       s.GetSolver(),
                                                       s.GetWorkspaceSize(),
                                                       invokeParams);
                                               }),
                                solutions.end());
            }

            if(!solutions.empty())
                return solutions;
            // All cached entries filtered out: fall through and let the
            // regenerator re-evaluate against the current workspace.
        }

        MIOPEN_LOG_I("Find-db regenerating.");
        record.in_sync = false;
        record.content.emplace(DbKinds::FindDb, problem);

        const auto result = regenerator();
        record.dont_store = !result.is_optimal;

        if(record.dont_store)
            return result.solutions;

        for(const auto& solution : result.solutions)
        {
            const auto algo = solution.GetSolver().GetAlgo(problem.GetDirection());
            record.content->SetValues(
                solution.GetSolver().ToString(),
                FindDbData{solution.GetTime(), solution.GetWorkspaceSize(), algo});
        }

        return result.solutions;
    }

private:
    fs::path path;
    fs::path installed_path;
    std::optional<DbTimer<TDb>> db;
    std::optional<DbRecord> content;
    bool in_sync    = false;
    bool dont_store = false; // E.g. to skip writing sub-optimal find-db records to disk.

    std::optional<DbTimer<TDb>> construct_db(bool cond, DbTimer<TDb>&& timer)
    {
        if(cond)
        {
            return std::optional<DbTimer<TDb>>(std::move(timer));
        }
        return std::nullopt;
    }

    static fs::path GetInstalledPath(const Handle& handle, const std::string& path_suffix);
    static fs::path GetInstalledPathEmbed(const Handle& handle, const std::string& path_suffix);
    static fs::path GetInstalledPathFile(const Handle& handle, const std::string& path_suffix);
    static fs::path GetUserPath(const Handle& handle, const std::string& path_suffix);

    // Returns true if rebuild is required
    bool Validate(const Handle& handle, const NetworkConfig& config) const;
    void CopyTo(std::vector<Solution>& to) const;

    void LogFindDbItem(const std::pair<std::string, FindDbData>& item) const;
};

extern template class FindDbRecord_t<FindDb>;
extern template class FindDbRecord_t<UserFindDb>;

} // namespace miopen

#endif
