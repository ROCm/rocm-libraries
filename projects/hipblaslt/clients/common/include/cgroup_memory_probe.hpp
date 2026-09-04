// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Host-only cgroup memory headroom probe; no GPU/HIP calls.

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <regex>
#include <string>
#include <vector>

namespace hipblaslt_client
{
    struct cgroup_paths
    {
        std::string v2; // relative path from /proc/self/cgroup (cgroup2)
        std::string v1; // relative path from /proc/self/cgroup (v1 memory line)
    };

    struct cgroup_mount
    {
        std::string mountpoint;
        std::string root; // path inside the cgroup fs covered by this mount
        std::string superopts;
        bool        is_v2 = false;
    };

    namespace detail
    {
        // Held in a non-template accessor so there is a single instance:
        // for_each_nonempty_line is a template, so a function-local static there
        // would be constructed once per instantiation.
        inline std::regex const& line_delimiter()
        {
            static std::regex const pattern{"\n"};
            return pattern;
        }
    } // namespace detail

    // Calls fn for each line of content, dropping trailing carriage returns so
    // CRLF input parses like LF, and skipping lines that are then empty.
    template <typename Fn>
    inline void for_each_nonempty_line(std::string const& content, Fn fn)
    {
        std::for_each(std::sregex_token_iterator(
                          content.begin(), content.end(), detail::line_delimiter(), -1),
                      std::sregex_token_iterator(),
                      [&fn](std::string const& raw) {
                          // npos + 1 == 0, so an all-CR line correctly truncates to empty.
                          std::string const line = raw.substr(0, raw.find_last_not_of('\r') + 1);
                          if(!line.empty())
                              fn(line);
                      });
    }

    inline std::string unescape_mount_path(std::string path)
    {
        for(size_t at = 0; at < path.size();)
        {
            if(at + 3 < path.size() && path[at] == '\\' && path[at + 1] == '0'
               && path[at + 2] == '4' && path[at + 3] == '0')
            {
                path.replace(at, 4, " ");
                ++at;
            }
            else
            {
                ++at;
            }
        }
        return path;
    }

    inline bool controller_list_has_memory(std::string const& list)
    {
        static std::regex const comma{","};
        return std::any_of(std::sregex_token_iterator(list.begin(), list.end(), comma, -1),
                           std::sregex_token_iterator(),
                           [](std::string const& name) { return name == "memory"; });
    }

    inline bool mount_has_memory_controller(cgroup_mount const& mount)
    {
        if(mount.is_v2)
            return false;
        auto        slash = mount.mountpoint.rfind('/');
        std::string base
            = slash == std::string::npos ? mount.mountpoint : mount.mountpoint.substr(slash + 1);
        return controller_list_has_memory(base) || controller_list_has_memory(mount.superopts);
    }

    // Each line is "<id>:<controllers>:<path>". The cgroup2 line has id 0 and an
    // empty controller list; v1 controllers may be co-mounted and comma-joined,
    // so match a whole token rather than a ":memory:" substring.
    inline cgroup_paths parse_proc_self_cgroup(std::string const& content)
    {
        cgroup_paths paths;
        for_each_nonempty_line(content, [&paths](std::string const& entry) {
            auto const controllers_at = entry.find(':');
            if(controllers_at == std::string::npos)
                return;
            auto const path_at = entry.find(':', controllers_at + 1);
            if(path_at == std::string::npos)
                return;

            std::string const id = entry.substr(0, controllers_at);
            std::string const controllers
                = entry.substr(controllers_at + 1, path_at - controllers_at - 1);
            std::string const path = entry.substr(path_at + 1);

            if(id == "0" && controllers.empty())
                paths.v2 = path;
            else if(controller_list_has_memory(controllers))
                paths.v1 = path;
        });
        return paths;
    }

    inline std::vector<cgroup_mount> parse_mountinfo(std::string const& content)
    {
        std::vector<cgroup_mount> mounts;
        for_each_nonempty_line(content, [&mounts](std::string const& line) {
            auto const sep = line.find(" - ");
            if(sep == std::string::npos)
                return;

            std::string const tail       = line.substr(sep + 3);
            auto const        fstype_end = tail.find(' ');
            if(fstype_end == std::string::npos)
                return;
            std::string const fstype = tail.substr(0, fstype_end);
            if(fstype != "cgroup" && fstype != "cgroup2")
                return;

            unsigned mount_id, parent_id, major, minor;
            char     root_raw[512]       = {};
            char     mountpoint_raw[512] = {};
            if(sscanf(line.c_str(),
                      "%u %u %u:%u %511s %511s",
                      &mount_id,
                      &parent_id,
                      &major,
                      &minor,
                      root_raw,
                      mountpoint_raw)
               < 6)
                return;

            cgroup_mount mount;
            mount.root       = unescape_mount_path(root_raw);
            mount.mountpoint = unescape_mount_path(mountpoint_raw);
            mount.is_v2      = (fstype == "cgroup2");
            auto super_start = tail.find(' ', fstype_end + 1);
            if(super_start != std::string::npos)
            {
                super_start = tail.find(' ', super_start + 1);
                if(super_start != std::string::npos)
                    mount.superopts = tail.substr(super_start + 1);
            }
            mounts.push_back(std::move(mount));
        });
        return mounts;
    }

    inline std::string normalize_cgroup_path(std::string path)
    {
        if(path.empty() || path.front() != '/')
            path.insert(path.begin(), '/');
        while(path.size() > 1 && path.back() == '/')
            path.pop_back();
        return path;
    }

    inline std::string resolve_cgroup_directory(cgroup_mount const& mount, std::string cgroup_rel)
    {
        if(cgroup_rel.empty())
            return {};

        std::string cgroup_path = normalize_cgroup_path(std::move(cgroup_rel));
        std::string mount_root  = normalize_cgroup_path(mount.root);
        if(mount_root == "/")
            return mount.mountpoint + cgroup_path;

        if(cgroup_path.size() < mount_root.size()
           || cgroup_path.compare(0, mount_root.size(), mount_root) != 0)
            return {};
        if(cgroup_path.size() > mount_root.size() && cgroup_path[mount_root.size()] != '/')
            return {};

        std::string suffix = cgroup_path.substr(mount_root.size());
        return mount.mountpoint + suffix;
    }

    inline cgroup_mount const* pick_cgroup_mount(std::vector<cgroup_mount> const& mounts,
                                                 bool                             want_v2,
                                                 std::string const&               cgroup_rel)
    {
        cgroup_mount const* best     = nullptr;
        size_t              best_len = 0;
        for(auto const& mount : mounts)
        {
            if(mount.is_v2 != want_v2)
                continue;
            if(!want_v2 && !mount_has_memory_controller(mount))
                continue;
            if(resolve_cgroup_directory(mount, cgroup_rel).empty())
                continue;
            size_t root_len = normalize_cgroup_path(mount.root).size();
            if(!best || root_len >= best_len)
            {
                best     = &mount;
                best_len = root_len;
            }
        }
        return best;
    }

    inline bool parse_cgroup_size_token(std::string const& token,
                                        size_t&            out,
                                        size_t             unlimited_sentinel)
    {
        if(token == "max")
        {
            out = std::numeric_limits<size_t>::max();
            return true;
        }
        char const*        cstr  = token.c_str();
        char*              end   = nullptr;
        unsigned long long value = strtoull(cstr, &end, 10);
        if(end == cstr)
            return false;
        out = static_cast<size_t>(value);
        if(out >= unlimited_sentinel)
            out = std::numeric_limits<size_t>::max();
        return true;
    }

    inline bool read_cgroup_size_file(std::string const& path, size_t& out, bool log_errors = false)
    {
        size_t const unlimited = static_cast<size_t>(1) << 62;

        FILE* f = fopen(path.c_str(), "r");
        if(!f)
        {
            if(log_errors)
            {
                std::fprintf(stderr, "hipBLASLt cgroup probe: cannot read %s\n", path.c_str());
            }
            return false;
        }
        char token[64] = {};
        bool ok        = std::fscanf(f, "%63s", token) == 1;
        std::fclose(f);
        if(!ok)
        {
            if(log_errors)
            {
                std::fprintf(
                    stderr, "hipBLASLt cgroup probe: empty or unreadable %s\n", path.c_str());
            }
            return false;
        }
        if(!parse_cgroup_size_token(token, out, unlimited))
        {
            if(log_errors)
            {
                std::fprintf(stderr,
                             "hipBLASLt cgroup probe: malformed value '%s' in %s\n",
                             token,
                             path.c_str());
            }
            return false;
        }
        return true;
    }

    namespace detail
    {
        using read_size_fn = bool (*)(std::string const& path, size_t& out, void* ctx);

        inline void log_probe_error(char const* message, std::string const& path)
        {
            std::fprintf(stderr, "hipBLASLt cgroup probe: %s %s\n", message, path.c_str());
        }

        inline bool read_cgroup_size_file_ctx(std::string const& path, size_t& out, void*)
        {
            return read_cgroup_size_file(path, out, false);
        }

        inline bool read_fake_sysfs_ctx(std::string const& path, size_t& out, void* ctx)
        {
            size_t const unlimited = static_cast<size_t>(1) << 62;
            auto const&  fake      = *static_cast<std::map<std::string, std::string> const*>(ctx);
            auto         it        = fake.find(path);
            if(it == fake.end())
                return false;
            return parse_cgroup_size_token(it->second, out, unlimited);
        }

        inline size_t cgroup_headroom(cgroup_paths const&              paths,
                                      std::vector<cgroup_mount> const& mounts,
                                      read_size_fn                     read,
                                      void*                            ctx,
                                      bool                             log_errors)
        {
            size_t const unlimited = static_cast<size_t>(1) << 62;
            size_t       available = std::numeric_limits<size_t>::max();
            bool const   live_read = log_errors;

            auto read_at = [&](std::string const& path, size_t& out, bool log_failure) -> bool {
                if(live_read)
                    return read_cgroup_size_file(path, out, log_failure);
                return read(path, out, ctx);
            };

            auto consider = [&](std::string const& dir, char const* limit, char const* usage) {
                std::string const limit_path = dir + "/" + limit;
                std::string const usage_path = dir + "/" + usage;
                size_t            cap = 0, used = 0;
                if(!read_at(limit_path, cap, true) || cap >= unlimited)
                    return;
                if(!read_at(usage_path, used, false))
                {
                    if(log_errors)
                    {
                        log_probe_error("usage missing or malformed, assuming 0:", usage_path);
                    }
                    used = 0;
                }
                available = std::min(available, cap > used ? cap - used : static_cast<size_t>(0));
            };

            auto walk_resolved = [&](std::string        dir,
                                     std::string const& mountpoint,
                                     char const*        limit,
                                     char const*        usage) {
                if(dir.empty())
                    return;
                for(;;)
                {
                    consider(dir, limit, usage);
                    if(dir == mountpoint)
                        break;
                    auto slash = dir.rfind('/');
                    if(slash == std::string::npos || slash < mountpoint.size())
                        break;
                    dir.erase(slash);
                }
            };

            if(!paths.v2.empty())
            {
                if(cgroup_mount const* mount = pick_cgroup_mount(mounts, true, paths.v2))
                {
                    std::string const dir = resolve_cgroup_directory(*mount, paths.v2);
                    if(dir.empty())
                    {
                        if(log_errors)
                        {
                            std::fprintf(
                                stderr,
                                "hipBLASLt cgroup probe: cannot resolve cgroup2 path '%s'\n",
                                paths.v2.c_str());
                        }
                    }
                    else
                    {
                        walk_resolved(dir, mount->mountpoint, "memory.max", "memory.current");
                    }
                }
                else if(log_errors)
                {
                    std::fprintf(stderr,
                                 "hipBLASLt cgroup probe: no cgroup2 mount covers '%s'\n",
                                 paths.v2.c_str());
                }
            }

            if(!paths.v1.empty())
            {
                if(cgroup_mount const* mount = pick_cgroup_mount(mounts, false, paths.v1))
                {
                    std::string const dir = resolve_cgroup_directory(*mount, paths.v1);
                    if(dir.empty())
                    {
                        if(log_errors)
                        {
                            std::fprintf(
                                stderr,
                                "hipBLASLt cgroup probe: cannot resolve cgroup v1 path '%s'\n",
                                paths.v1.c_str());
                        }
                    }
                    else
                    {
                        walk_resolved(dir,
                                      mount->mountpoint,
                                      "memory.limit_in_bytes",
                                      "memory.usage_in_bytes");
                    }
                }
                else if(log_errors)
                {
                    std::fprintf(stderr,
                                 "hipBLASLt cgroup probe: no v1 memory mount covers '%s'\n",
                                 paths.v1.c_str());
                }
            }

            return available;
        }
    } // namespace detail

    inline size_t cgroup_available_memory_from(cgroup_paths const& paths,
                                               std::string const&  mountinfo)
    {
        return detail::cgroup_headroom(
            paths, parse_mountinfo(mountinfo), detail::read_cgroup_size_file_ctx, nullptr, true);
    }

    inline size_t cgroup_available_memory_from(cgroup_paths const&                       paths,
                                               std::string const&                        mountinfo,
                                               std::map<std::string, std::string> const& fake_sysfs)
    {
        return detail::cgroup_headroom(paths,
                                       parse_mountinfo(mountinfo),
                                       detail::read_fake_sysfs_ctx,
                                       const_cast<void*>(static_cast<void const*>(&fake_sysfs)),
                                       false);
    }

#ifdef __linux__
    inline std::string read_proc_file(char const* path)
    {
        std::string content;
        if(FILE* f = fopen(path, "r"))
        {
            char line[512];
            while(fgets(line, sizeof(line), f))
                content += line;
            fclose(f);
        }
        return content;
    }

    inline size_t cgroup_available_memory_live()
    {
        return cgroup_available_memory_from(
            parse_proc_self_cgroup(read_proc_file("/proc/self/cgroup")),
            read_proc_file("/proc/self/mountinfo"));
    }
#endif

} // namespace hipblaslt_client
