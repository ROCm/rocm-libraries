// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "include/local_kernel_store.hpp"
#include "include/rocblaslt_secure_env.hpp"

#include <hipblaslt/hipblaslt-version.h>

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <random>
#include <sstream>
#include <thread>
#include <vector>

#ifndef TO_STR2
#define TO_STR2(x) #x
#define TO_STR(x) TO_STR2(x)
#endif

#ifndef HIPBLASLT_VERSION_TWEAK
#error "HIPBLASLT_VERSION_TWEAK (hipblaslt-version.h) decides which local kernel journal is read"
#endif

namespace fs = std::filesystem;

namespace rocblaslt
{
    namespace local_kernel_store
    {
        namespace
        {
            constexpr const char* kEnvName       = "HIPBLASLT_LOCAL_KERNEL_LIBRARY";
            constexpr const char* kObjectsDir    = "objects";
            constexpr const char* kJournalPrefix = "journal-";
            constexpr const char* kZlibSuffix    = ".zlib";
            constexpr const char* kTempPrefix    = ".tmp.";
            constexpr int         kFormat        = 1;
            // No copy takes this long, so an older temporary file is a crashed writer's.
            constexpr time_t kStaleTempSeconds = 60 * 60;
            constexpr auto   kLockTimeout      = std::chrono::seconds(10);

            void setError(std::string* error, std::string const& message)
            {
                if(error)
                    *error = message;
            }

            std::string errnoText(std::string const& what, std::string const& path)
            {
                return what + " " + path + ": " + std::strerror(errno);
            }

            bool endsWith(std::string const& text, std::string const& suffix)
            {
                return text.size() >= suffix.size()
                       && text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
            }

            bool isNameChar(char c)
            {
                return std::isalnum(static_cast<unsigned char>(c)) || c == '.' || c == '_'
                       || c == '-';
            }

            /**
             * The version and build stamp, e.g. 1.5.0-3c1c5d13b19, with any
             * character not safe in a file name replaced. A build from sources
             * without git history has no stamp, so the version is what keeps
             * one release's journal from another's.
             */
            std::string const& buildStamp()
            {
                static const std::string stamp = [] {
                    std::string text = std::to_string(HIPBLASLT_VERSION_MAJOR) + "."
                                       + std::to_string(HIPBLASLT_VERSION_MINOR) + "."
                                       + std::to_string(HIPBLASLT_VERSION_PATCH);
                    const std::string tweak = TO_STR(HIPBLASLT_VERSION_TWEAK);
                    if(!tweak.empty())
                        text += "-" + tweak;
                    for(char& c : text)
                        if(!isNameChar(c))
                            c = '_';
                    return text;
                }();
                return stamp;
            }

            // The V line up to its journal id.
            std::string versionPrefix()
            {
                return "V " + std::to_string(kFormat) + " " + buildStamp() + " ";
            }

            std::string newVersionLine()
            {
                uint64_t id = 0;
                try
                {
                    std::random_device random;
                    id = (uint64_t(random()) << 32) ^ random();
                }
                catch(std::exception const&)
                {
                }
                id ^= uint64_t(std::chrono::steady_clock::now().time_since_epoch().count())
                      ^ (uint64_t(::getpid()) << 40);
                char text[17];
                std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(id));
                return versionPrefix() + text;
            }

            bool isVersionLine(std::string const& line)
            {
                const std::string prefix = versionPrefix();
                if(line.size() != prefix.size() + 16 || line.compare(0, prefix.size(), prefix) != 0)
                    return false;
                for(size_t i = prefix.size(); i < line.size(); ++i)
                    if(!std::isxdigit(static_cast<unsigned char>(line[i])))
                        return false;
                return true;
            }

            class Fd
            {
            public:
                explicit Fd(int fd = -1)
                    : m_fd(fd)
                {
                }
                ~Fd()
                {
                    if(m_fd >= 0)
                        static_cast<void>(::close(m_fd));
                }
                Fd(Fd const&)            = delete;
                Fd& operator=(Fd const&) = delete;

                int get() const
                {
                    return m_fd;
                }

                void reset(int fd)
                {
                    if(m_fd >= 0)
                        static_cast<void>(::close(m_fd));
                    m_fd = fd;
                }

            private:
                int m_fd;
            };

            void syncDirectory(fs::path const& dir)
            {
                Fd fd(::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
                if(fd.get() >= 0)
                    static_cast<void>(::fsync(fd.get()));
            }

            bool writeAll(int fd, char const* data, size_t size)
            {
                while(size > 0)
                {
                    const ssize_t wrote = ::write(fd, data, size);
                    if(wrote < 0)
                    {
                        if(errno == EINTR)
                            continue;
                        return false;
                    }
                    data += wrote;
                    size -= static_cast<size_t>(wrote);
                }
                return true;
            }

            bool readFrom(int fd, uint64_t offset, std::string* out)
            {
                out->clear();
                char buffer[64 * 1024];
                for(;;)
                {
                    const ssize_t got = ::pread(fd, buffer, sizeof(buffer), offset);
                    if(got < 0)
                    {
                        if(errno == EINTR)
                            continue;
                        return false;
                    }
                    if(got == 0)
                        return true;
                    out->append(buffer, static_cast<size_t>(got));
                    offset += static_cast<uint64_t>(got);
                }
            }

            /**
             * 128-bit FNV-1a over two independently seeded lanes. It names
             * objects and detects damaged ones. It does not detect tampering;
             * the ownership and permission checks keep other users from
             * writing the library.
             */
            class ObjectHash
            {
            public:
                void update(unsigned char const* data, size_t size)
                {
                    for(size_t i = 0; i < size; ++i)
                    {
                        m_a = (m_a ^ data[i]) * 0x100000001b3ull;
                        m_b = (m_b ^ data[i]) * 0xff51afd7ed558ccdull;
                    }
                }

                std::string hex() const
                {
                    char text[33];
                    std::snprintf(text,
                                  sizeof(text),
                                  "%016llx%016llx",
                                  static_cast<unsigned long long>(m_a),
                                  static_cast<unsigned long long>(m_b));
                    return text;
                }

            private:
                uint64_t m_a = 0xcbf29ce484222325ull;
                uint64_t m_b = 0x9e3779b97f4a7c15ull;
            };

            bool hashFile(std::string const& path, std::string* out, std::string* error)
            {
                Fd fd(::open(path.c_str(), O_RDONLY | O_CLOEXEC));
                if(fd.get() < 0)
                {
                    setError(error, errnoText("cannot read", path));
                    return false;
                }

                ObjectHash                 hash;
                std::vector<unsigned char> buffer(1 << 20);
                for(;;)
                {
                    const ssize_t got = ::read(fd.get(), buffer.data(), buffer.size());
                    if(got < 0)
                    {
                        if(errno == EINTR)
                            continue;
                        setError(error, errnoText("cannot read", path));
                        return false;
                    }
                    if(got == 0)
                        break;
                    hash.update(buffer.data(), static_cast<size_t>(got));
                }
                *out = hash.hex();
                return true;
            }

            /** 32-bit FNV-1a of a record's other fields, its last field. */
            std::string lineChecksum(std::string const& body)
            {
                uint32_t sum = 0x811c9dc5u;
                for(unsigned char c : body)
                    sum = (sum ^ c) * 0x01000193u;
                char text[9];
                std::snprintf(text, sizeof(text), "%08x", sum);
                return text;
            }

            bool isHash(std::string const& text)
            {
                if(text.size() != 32)
                    return false;
                for(char c : text)
                    if(!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
                        return false;
                return true;
            }

            bool checkStat(std::string const& path, struct stat const& st, std::string* error)
            {
                if(st.st_uid != ::geteuid())
                {
                    setError(error, path + " is owned by another user");
                    return false;
                }
                if(st.st_mode & (S_IWGRP | S_IWOTH))
                {
                    setError(error, path + " is group- or world-writable");
                    return false;
                }
                return true;
            }

            bool checkPrivate(fs::path const& path, std::string* error)
            {
                struct stat st
                {
                };
                if(::stat(path.c_str(), &st) != 0)
                {
                    setError(error, errnoText("cannot stat", path.string()));
                    return false;
                }
                return checkStat(path.string(), st, error);
            }

            /**
             * Creates `dir` if needed. A directory created here drops the group
             * and other write bits, so a umask such as 0002 does not make it
             * fail checkPrivate. An existing directory is not modified.
             */
            bool makePrivateDirectory(fs::path const& dir, std::string* error)
            {
                if(::mkdir(dir.c_str(), 0755) == 0)
                {
                    struct stat st
                    {
                    };
                    if(::stat(dir.c_str(), &st) == 0)
                        static_cast<void>(
                            ::chmod(dir.c_str(), st.st_mode & 07777 & ~(S_IWGRP | S_IWOTH)));
                    syncDirectory(dir.parent_path());
                }
                else if(errno != EEXIST)
                {
                    setError(error, errnoText("cannot create", dir.string()));
                    return false;
                }
                return checkPrivate(dir, error);
            }

            /**
             * Copies `from` to a new file in `dir` whose name starts with
             * kTempPrefix, has a random part, and ends in `suffix`, hashing the
             * bytes as they are written. Writers in other processes, including
             * ones in other containers sharing the directory, never pick the
             * same name.
             */
            bool copyToTemp(std::string const& from,
                            fs::path const&    dir,
                            std::string const& suffix,
                            std::string*       temp,
                            std::string*       hash,
                            std::string*       error)
            {
                Fd in(::open(from.c_str(), O_RDONLY | O_CLOEXEC));
                if(in.get() < 0)
                {
                    setError(error, errnoText("cannot read", from));
                    return false;
                }

                std::string name = (dir / (kTempPrefix + std::string("XXXXXX") + suffix)).string();
                Fd out(::mkostemps(name.data(), static_cast<int>(suffix.size()), O_CLOEXEC));
                if(out.get() < 0)
                {
                    setError(error, errnoText("cannot create a temporary file in", dir.string()));
                    return false;
                }
                static_cast<void>(::fchmod(out.get(), 0644));

                ObjectHash                 written;
                std::vector<unsigned char> buffer(1 << 20);
                for(;;)
                {
                    const ssize_t got = ::read(in.get(), buffer.data(), buffer.size());
                    if(got < 0 && errno == EINTR)
                        continue;
                    if(got < 0
                       || (got > 0
                           && !writeAll(out.get(),
                                        reinterpret_cast<char const*>(buffer.data()),
                                        static_cast<size_t>(got))))
                    {
                        setError(error, errnoText("cannot copy to", name));
                        static_cast<void>(::unlink(name.c_str()));
                        return false;
                    }
                    if(got == 0)
                        break;
                    written.update(buffer.data(), static_cast<size_t>(got));
                }

                if(::fsync(out.get()) != 0)
                {
                    setError(error, errnoText("cannot sync", name));
                    static_cast<void>(::unlink(name.c_str()));
                    return false;
                }
                *temp = name;
                *hash = written.hex();
                return true;
            }

            /** Removes temporary files older than any copy takes. */
            void removeStaleTemps(fs::path const& dir)
            {
                const time_t    cutoff = ::time(nullptr) - kStaleTempSeconds;
                std::error_code ec;
                for(auto const& entry : fs::directory_iterator(dir, ec))
                {
                    if(entry.path().filename().string().rfind(kTempPrefix, 0) != 0)
                        continue;
                    struct stat st
                    {
                    };
                    if(::lstat(entry.path().c_str(), &st) == 0 && S_ISREG(st.st_mode)
                       && st.st_mtime < cutoff)
                        static_cast<void>(::unlink(entry.path().c_str()));
                }
            }

            /**
             * Sets `line` to the file's first line and `complete` to whether it
             * ends in a newline. A first line too long to be a V line counts as
             * complete, so that it is rejected rather than taken for an
             * unfinished one.
             */
            bool readFirstLine(int fd, std::string* line, bool* complete)
            {
                char    buffer[512];
                ssize_t got;
                do
                    got = ::pread(fd, buffer, sizeof(buffer), 0);
                while(got < 0 && errno == EINTR);
                if(got < 0)
                    return false;

                auto const* newline = static_cast<char const*>(std::memchr(buffer, '\n', got));
                *line     = std::string(buffer, newline ? size_t(newline - buffer) : size_t(got));
                *complete = newline || got == ssize_t(sizeof(buffer));
                return true;
            }

            bool parseLine(std::string const& line, JournalEntry* entry)
            {
                const size_t split = line.rfind(' ');
                if(split == std::string::npos
                   || line.compare(
                          split + 1, std::string::npos, lineChecksum(line.substr(0, split)))
                          != 0)
                    return false;

                std::istringstream fields(line.substr(0, split));
                std::string        kind, extra;
                if(!(fields >> kind))
                    return false;

                if(kind == "R")
                {
                    std::string format;
                    if(!(fields >> entry->payload.id.datHash >> entry->payload.id.coHash >> format)
                       || (fields >> extra))
                        return false;
                    if(format != "dat" && format != "dat.zlib")
                        return false;
                    entry->kind               = JournalEntry::Kind::Payload;
                    entry->payload.compressed = format == "dat.zlib";
                    return isHash(entry->payload.id.datHash) && isHash(entry->payload.id.coHash);
                }

                if(kind == "E")
                {
                    auto& e = entry->exact;
                    if(!(fields >> e.id.datHash >> e.id.coHash >> e.payloadIndex >> e.m >> e.n
                         >> e.batch >> e.k)
                       || (fields >> extra))
                        return false;
                    entry->kind = JournalEntry::Kind::Exact;
                    return isHash(e.id.datHash) && isHash(e.id.coHash) && e.payloadIndex >= 0;
                }

                return false;
            }

            /**
             * Parses the complete lines of `text`, which starts at byte
             * `offset` of the journal at `path`; a trailing partial line is
             * left unread. The journal's first line must be this build's V
             * line. A journal without a complete first line holds no records.
             */
            bool parseJournal(std::string const&         text,
                              uint64_t                   offset,
                              std::string const&         path,
                              std::vector<JournalEntry>* entries,
                              uint64_t*                  end,
                              std::string*               error)
            {
                size_t start = 0;
                if(offset == 0)
                {
                    const size_t newline = text.find('\n');
                    if(newline == std::string::npos)
                    {
                        *end = 0;
                        return true;
                    }
                    if(!isVersionLine(text.substr(0, newline)))
                    {
                        setError(error,
                                 path + " does not start with \"" + versionPrefix()
                                     + "<id>\"; it was not written by this build of hipBLASLt");
                        return false;
                    }
                    start = newline + 1;
                }

                for(;;)
                {
                    const size_t newline = text.find('\n', start);
                    if(newline == std::string::npos)
                        break;

                    JournalEntry entry;
                    entry.offset = offset + start;
                    if(parseLine(text.substr(start, newline - start), &entry))
                        entries->push_back(std::move(entry));
                    start = newline + 1;
                }
                *end = offset + start;
                return true;
            }

            /**
             * Appends text ending in one record to a journal opened with
             * O_APPEND and locked exclusively. A partial last line, left by a
             * writer that crashed, is terminated first so the new record starts
             * on its own line; its checksum keeps the partial line from being
             * read as a record.
             *
             * A failed append is truncated away. Its bytes would otherwise stay,
             * and the next append would complete them, which for a record
             * missing only its newline is the record reported as not saved.
             */
            bool appendLine(int fd, std::string const& path, std::string line, std::string* error)
            {
                struct stat st
                {
                };
                if(::fstat(fd, &st) != 0)
                {
                    setError(error, errnoText("cannot stat", path));
                    return false;
                }
                if(st.st_size > 0)
                {
                    char last = '\n';
                    if(::pread(fd, &last, 1, st.st_size - 1) == 1 && last != '\n')
                        line.insert(line.begin(), '\n');
                }

                line.push_back('\n');
                if(!writeAll(fd, line.data(), line.size()) || ::fsync(fd) != 0)
                {
                    setError(error, errnoText("cannot append to", path));
                    static_cast<void>(::ftruncate(fd, st.st_size));
                    return false;
                }
                return true;
            }

            /**
             * Opens and flocks a journal: exclusively for appending, creating it
             * if absent, or shared for reading. A missing journal opened for
             * reading succeeds and leaves `out` empty. The lock is on the file
             * at `path` when this returns. Gives up after kLockTimeout, so a
             * process stopped while holding the lock makes this one fail
             * instead of waiting for it.
             */
            bool openJournal(std::string const& path, bool exclusive, Fd* out, std::string* error)
            {
                const int  flags    = exclusive ? (O_RDWR | O_CREAT | O_APPEND | O_CLOEXEC)
                                                : (O_RDONLY | O_CLOEXEC);
                const auto deadline = std::chrono::steady_clock::now() + kLockTimeout;
                auto       pause    = std::chrono::milliseconds(1);
                auto       timedOut = [&] {
                    if(std::chrono::steady_clock::now() < deadline)
                        return false;
                    setError(error,
                             path + " stayed locked by another process for "
                                 + std::to_string(kLockTimeout.count()) + " s");
                    return true;
                };

                for(;;)
                {
                    out->reset(-1);
                    const int fd = ::open(path.c_str(), flags, 0644);
                    if(fd < 0)
                    {
                        if(!exclusive && errno == ENOENT)
                            return true;
                        setError(error, errnoText("cannot open", path));
                        return false;
                    }
                    out->reset(fd);

                    struct stat st
                    {
                    };
                    if(::fstat(fd, &st) != 0)
                    {
                        setError(error, errnoText("cannot stat", path));
                        return false;
                    }
                    if(!checkStat(path, st, error))
                        return false;

                    while(::flock(fd, (exclusive ? LOCK_EX : LOCK_SH) | LOCK_NB) != 0)
                    {
                        if(errno == EINTR)
                            continue;
                        if(errno != EWOULDBLOCK)
                        {
                            setError(error, errnoText("cannot lock", path));
                            return false;
                        }
                        if(timedOut())
                            return false;
                        std::this_thread::sleep_for(pause);
                        pause = std::min(pause * 2, std::chrono::milliseconds(50));
                    }

                    // The journal may have been deleted or replaced while this
                    // waited, leaving the lock on a file no one reads: open the
                    // one at `path` again.
                    struct stat held
                    {
                    }, named{};
                    if(::fstat(fd, &held) == 0 && ::stat(path.c_str(), &named) == 0
                       && held.st_dev == named.st_dev && held.st_ino == named.st_ino
                       && held.st_nlink > 0)
                        return true;
                    if(timedOut())
                        return false;
                }
            }

            std::string withChecksum(std::string const& body)
            {
                return body + " " + lineChecksum(body);
            }

            std::string payloadLine(PayloadRecord const& record)
            {
                return withChecksum("R " + record.id.datHash + " " + record.id.coHash + " "
                                    + (record.compressed ? "dat.zlib" : "dat"));
            }

            std::string exactLine(ExactRecord const& record)
            {
                std::ostringstream line;
                line << "E " << record.id.datHash << ' ' << record.id.coHash << ' '
                     << record.payloadIndex << ' ' << record.m << ' ' << record.n << ' '
                     << record.batch << ' ' << record.k;
                return withChecksum(line.str());
            }

            std::array<uint64_t, 4> shapeOf(ExactRecord const& record)
            {
                return {record.m, record.n, record.batch, record.k};
            }
        } // namespace

        std::string resolveShardFile(std::string const& datPath)
        {
            std::error_code ec;
            if(!endsWith(datPath, kZlibSuffix) && fs::exists(datPath + kZlibSuffix, ec))
                return datPath + kZlibSuffix;
            return datPath;
        }

        bool identifyPayload(std::string const& shardFile,
                             std::string const& coPath,
                             PayloadRecord*     out,
                             std::string*       error)
        {
            out->compressed = endsWith(shardFile, kZlibSuffix);
            return hashFile(shardFile, &out->id.datHash, error)
                   && hashFile(coPath, &out->id.coHash, error);
        }

        Store const* Store::configured()
        {
            static Store const* const store = []() -> Store const* {
                const bool privileged = rocblaslt_process_is_privileged();
                if(rocblaslt_env_suppressed_for_security_impl(kEnvName, privileged))
                {
                    std::cerr << "\nrocblaslt warning: ignoring " << kEnvName
                              << " because the process is running in a secure execution "
                                 "context; local kernels will not be loaded or saved."
                              << std::endl;
                    return nullptr;
                }

                const char* env = rocblaslt_secure_getenv_impl(kEnvName, privileged);
                if(!env || !*env)
                    return nullptr;

                std::string error;
                Store*      opened = open(env, &error);
                if(!opened)
                    std::cerr << "\nrocblaslt warning: not using local kernel library " << env
                              << ": " << error << ". Local kernels will not be loaded or saved."
                              << std::endl;
                return opened;
            }();
            return store;
        }

        Store* Store::open(std::string const& root, std::string* error)
        {
            if(root.empty() || !fs::path(root).is_absolute())
            {
                setError(error, "the path must be absolute");
                return nullptr;
            }

            const fs::path  rootPath = fs::path(root).lexically_normal();
            std::error_code ec;
            fs::create_directories(rootPath.parent_path(), ec);
            if(!makePrivateDirectory(rootPath, error)
               || !makePrivateDirectory(rootPath / kObjectsDir, error))
                return nullptr;
            removeStaleTemps(rootPath / kObjectsDir);

            return new Store(rootPath.string());
        }

        bool Store::checkLayout(std::string* error) const
        {
            return checkPrivate(m_root, error)
                   && checkPrivate(fs::path(m_root) / kObjectsDir, error);
        }

        std::string Store::shardPath(PayloadRecord const& record) const
        {
            return (fs::path(m_root) / kObjectsDir
                    / (record.id.datHash + ".dat" + (record.compressed ? kZlibSuffix : "")))
                .string();
        }

        std::string Store::codeObjectPath(PayloadId const& id) const
        {
            return (fs::path(m_root) / kObjectsDir / (id.coHash + ".co")).string();
        }

        std::string Store::journalPath(std::string const& scope, std::string* error) const
        {
            bool safe = !scope.empty() && scope != "." && scope != ".." && scope != kObjectsDir;
            for(char c : scope)
                safe = safe && isNameChar(c);
            if(!safe)
            {
                setError(error, "invalid journal name " + scope);
                return {};
            }
            return (fs::path(m_root) / scope / (kJournalPrefix + buildStamp())).string();
        }

        StagedObjects::~StagedObjects()
        {
            for(auto const* object : {&m_shard, &m_co})
                if(!object->temp.empty())
                    static_cast<void>(::unlink(object->temp.c_str()));
        }

        bool Store::stageObjects(std::string const& shardFile,
                                 std::string const& coPath,
                                 PayloadRecord*     record,
                                 StagedObjects*     staged,
                                 std::string*       error) const
        {
            if(!checkLayout(error))
                return false;

            const fs::path objects = fs::path(m_root) / kObjectsDir;
            auto           stage   = [&](std::string const&     from,
                             std::string const&     suffix,
                             std::string&           hash,
                             StagedObjects::Object& object) {
                const fs::path saved = objects / (hash + suffix);
                struct stat st
                {
                };
                if(::lstat(saved.c_str(), &st) == 0)
                {
                    std::string actual;
                    if(S_ISREG(st.st_mode) && checkStat(saved.string(), st, nullptr)
                       && hashFile(saved.string(), &actual, nullptr) && actual == hash)
                    {
                        object.final = saved.string();
                        return true;
                    }
                    object.replace = true;
                }

                std::string copied;
                if(!copyToTemp(from, objects, suffix, &object.temp, &copied, error))
                    return false;
                hash         = copied;
                object.final = (objects / (copied + suffix)).string();
                return true;
            };
            return stage(shardFile,
                         std::string(".dat") + (record->compressed ? kZlibSuffix : ""),
                         record->id.datHash,
                         staged->m_shard)
                   && stage(coPath, ".co", record->id.coHash, staged->m_co);
        }

        bool Store::publishObjects(StagedObjects* staged, std::string* error) const
        {
            // Both objects are durable before any record names them.
            for(auto* object : {&staged->m_shard, &staged->m_co})
            {
                if(object->temp.empty())
                    continue;
                // A hard link fails if another writer published the object
                // first, which keeps theirs; rename replaces a damaged one.
                if(object->replace)
                {
                    if(::rename(object->temp.c_str(), object->final.c_str()) != 0)
                    {
                        setError(error, errnoText("cannot publish", object->final));
                        return false;
                    }
                }
                else if(::link(object->temp.c_str(), object->final.c_str()) != 0 && errno != EEXIST
                        && ::rename(object->temp.c_str(), object->final.c_str()) != 0)
                {
                    setError(error, errnoText("cannot publish", object->final));
                    return false;
                }
                static_cast<void>(::unlink(object->temp.c_str()));
                object->temp.clear();
            }
            syncDirectory(fs::path(m_root) / kObjectsDir);
            return true;
        }

        std::string Store::objectState(PayloadRecord const& record) const
        {
            // Everything verifyObjects checks. For the objects: their contents,
            // through size and mtime, and their owner and mode, which ctime
            // also covers. For the directories checkLayout checks: only owner
            // and mode, since their times move whenever any object is saved.
            std::ostringstream state;
            for(auto const& dir : {fs::path(m_root), fs::path(m_root) / kObjectsDir})
            {
                struct stat st
                {
                };
                if(::stat(dir.c_str(), &st) != 0)
                    state << "missing;";
                else
                    state << st.st_mode << ':' << st.st_uid << ';';
            }
            for(auto const& path : {shardPath(record), codeObjectPath(record.id)})
            {
                struct stat st
                {
                };
                if(::stat(path.c_str(), &st) != 0)
                    state << "missing;";
                else
                    state << st.st_dev << ':' << st.st_ino << ':' << st.st_size << ':' << st.st_mode
                          << ':' << st.st_uid << ':' << st.st_mtim.tv_sec << '.'
                          << st.st_mtim.tv_nsec << ':' << st.st_ctim.tv_sec << '.'
                          << st.st_ctim.tv_nsec << ';';
            }
            return state.str();
        }

        bool Store::verifyObjects(PayloadRecord const& record, std::string* error) const
        {
            if(!checkLayout(error))
                return false;

            for(auto const& [path, expected] :
                {std::make_pair(shardPath(record), record.id.datHash),
                 std::make_pair(codeObjectPath(record.id), record.id.coHash)})
            {
                std::string actual;
                if(!checkPrivate(path, error) || !hashFile(path, &actual, error))
                    return false;
                if(actual != expected)
                {
                    setError(error, path + " is damaged: its contents do not match its name");
                    return false;
                }
            }
            return true;
        }

        bool Store::appendUnlessKnown(
            std::string const&                                     scope,
            std::function<std::string(JournalState const&)> const& makeLine,
            JournalCursor*                                         from,
            std::string*                                           error) const
        {
            const std::string path = journalPath(scope, error);
            if(path.empty() || !checkLayout(error)
               || !makePrivateDirectory(fs::path(path).parent_path(), error))
                return false;

            ScopeJournal* journal = nullptr;
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                journal = &m_journals[scope];
            }
            std::lock_guard<std::mutex> lock(journal->mutex);

            Fd fd;
            if(!openJournal(path, true, &fd, error))
                return false;

            struct stat st
            {
            };
            std::string first;
            bool        complete = false;
            if(::fstat(fd.get(), &st) != 0 || !readFirstLine(fd.get(), &first, &complete))
            {
                setError(error, errnoText("cannot read", path));
                return false;
            }

            // Re-read under the lock: another process may have appended, or
            // replaced the journal.
            JournalState& state = journal->state;
            std::string   prefix;
            if(!complete)
            {
                // New, or left without a complete V line by a writer that
                // crashed creating it. Either way it holds no records.
                if(st.st_size > 0 && ::ftruncate(fd.get(), 0) != 0)
                {
                    setError(error, errnoText("cannot truncate", path));
                    return false;
                }
                state  = JournalState{};
                prefix = newVersionLine();
                if(from)
                    *from = JournalCursor{prefix, 0};
                prefix += "\n";
            }
            else if(first != state.cursor.journal || uint64_t(st.st_size) < state.cursor.offset)
            {
                state                = JournalState{};
                state.cursor.journal = first;
            }

            std::string text;
            if(complete && !readFrom(fd.get(), state.cursor.offset, &text))
            {
                setError(error, errnoText("cannot read", path));
                return false;
            }

            std::vector<JournalEntry> entries;
            uint64_t                  end = state.cursor.offset;
            if(!parseJournal(text, state.cursor.offset, path, &entries, &end, error))
                return false;
            for(auto const& entry : entries)
            {
                if(entry.kind == JournalEntry::Kind::Payload)
                    state.payloads.insert(entry.payload.id);
                else
                    state.lastExact[shapeOf(entry.exact)] = entry;
            }
            state.cursor.offset = end;
            // A line appended now starts at or after `end`, which is where the
            // last complete line ends.
            if(from && complete)
                *from = JournalCursor{first, end};

            const std::string line = makeLine(state);
            if(line.empty())
                return true;
            return appendLine(fd.get(), path, prefix + line, error);
        }

        bool Store::appendPayload(std::string const&   scope,
                                  PayloadRecord const& record,
                                  std::string*         error) const
        {
            return appendUnlessKnown(
                scope,
                [&](JournalState const& state) {
                    return state.payloads.count(record.id) ? std::string() : payloadLine(record);
                },
                nullptr,
                error);
        }

        bool Store::appendExact(std::string const&   scope,
                                PayloadRecord const& payload,
                                ExactRecord const&   record,
                                JournalCursor*       from,
                                std::string*         error) const
        {
            return appendUnlessKnown(
                scope,
                [&](JournalState const& state) {
                    std::string lines;
                    if(!state.payloads.count(payload.id))
                        lines = payloadLine(payload) + "\n";
                    auto last = state.lastExact.find(shapeOf(record));
                    if(lines.empty() && last != state.lastExact.end()
                       && last->second.exact.id == record.id
                       && last->second.exact.payloadIndex == record.payloadIndex)
                    {
                        from->offset = last->second.offset;
                        return std::string();
                    }
                    return lines + exactLine(record);
                },
                from,
                error);
        }

        bool Store::read(std::string const&         scope,
                         JournalCursor*             cursor,
                         std::vector<JournalEntry>* entries,
                         std::string*               error) const
        {
            entries->clear();

            const std::string path = journalPath(scope, error);
            if(path.empty() || !checkLayout(error))
                return false;

            std::error_code ec;
            const fs::path  dir = fs::path(path).parent_path();
            if(!fs::exists(dir, ec))
            {
                *cursor = JournalCursor{};
                return true;
            }
            if(!checkPrivate(dir, error))
                return false;

            Fd fd;
            if(!openJournal(path, false, &fd, error))
                return false;
            if(fd.get() < 0)
            {
                *cursor = JournalCursor{};
                return true;
            }

            struct stat st
            {
            };
            std::string first;
            bool        complete = false;
            if(::fstat(fd.get(), &st) != 0 || !readFirstLine(fd.get(), &first, &complete))
            {
                setError(error, errnoText("cannot read", path));
                return false;
            }
            if(!complete)
            {
                *cursor = JournalCursor{};
                return true;
            }

            JournalCursor next = *cursor;
            if(first != next.journal || uint64_t(st.st_size) < next.offset)
                next = JournalCursor{first, 0};

            std::string text;
            uint64_t    end = next.offset;
            if(!readFrom(fd.get(), next.offset, &text))
            {
                setError(error, errnoText("cannot read", path));
                return false;
            }
            if(!parseJournal(text, next.offset, path, entries, &end, error))
                return false;
            next.offset = end;
            *cursor     = next;
            return true;
        }
    } // namespace local_kernel_store
} // namespace rocblaslt
