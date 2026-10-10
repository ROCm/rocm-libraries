// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "include/user_kernel_store.hpp"
#include "include/rocblaslt_secure_env.hpp"

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <vector>

namespace fs = std::filesystem;

namespace rocblaslt
{
    namespace user_kernel_store
    {
        namespace
        {
            constexpr const char* kEnvName     = "HIPBLASLT_USER_KERNEL_LIBRARY";
            constexpr const char* kObjectsDir  = "objects";
            constexpr const char* kJournalName = "journal";
            constexpr const char* kZlibSuffix  = ".zlib";

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
             * objects; it is not a defence against a writer who can already
             * modify the library, which the directory checks exclude.
             */
            bool hashFile(std::string const& path, std::string* out, std::string* error)
            {
                Fd fd(::open(path.c_str(), O_RDONLY | O_CLOEXEC));
                if(fd.get() < 0)
                {
                    setError(error, errnoText("cannot read", path));
                    return false;
                }

                uint64_t                   a = 0xcbf29ce484222325ull, b = 0x9e3779b97f4a7c15ull;
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
                    for(ssize_t i = 0; i < got; ++i)
                    {
                        a = (a ^ buffer[i]) * 0x100000001b3ull;
                        b = (b ^ buffer[i]) * 0xff51afd7ed558ccdull;
                    }
                }

                char text[33];
                std::snprintf(text,
                              sizeof(text),
                              "%016llx%016llx",
                              static_cast<unsigned long long>(a),
                              static_cast<unsigned long long>(b));
                *out = text;
                return true;
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

            /**
             * Whoever can write the library chooses which code objects are
             * loaded, so a directory others can write is refused.
             */
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
                if(st.st_mode & (S_IWGRP | S_IWOTH))
                {
                    setError(error, path.string() + " is group- or world-writable");
                    return false;
                }
                return true;
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
             * Copies to a temporary name unique to this process and renames it
             * into place, so concurrent writers of the same object never see a
             * partial file. Content addressed: an existing object is kept.
             */
            bool copyDurable(std::string const& from, fs::path const& to, std::string* error)
            {
                std::error_code ec;
                if(fs::exists(to, ec))
                    return true;

                static std::atomic<unsigned> counter{0};
                const fs::path temp = to.string() + ".tmp." + std::to_string(::getpid()) + "."
                                      + std::to_string(counter++);

                Fd in(::open(from.c_str(), O_RDONLY | O_CLOEXEC));
                if(in.get() < 0)
                {
                    setError(error, errnoText("cannot read", from));
                    return false;
                }

                {
                    Fd out(::open(temp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644));
                    if(out.get() < 0)
                    {
                        setError(error, errnoText("cannot create", temp.string()));
                        return false;
                    }

                    std::vector<char> buffer(1 << 20);
                    for(;;)
                    {
                        const ssize_t got = ::read(in.get(), buffer.data(), buffer.size());
                        if(got < 0 && errno == EINTR)
                            continue;
                        if(got < 0 || (got > 0 && !writeAll(out.get(), buffer.data(), got)))
                        {
                            setError(error, errnoText("cannot copy to", temp.string()));
                            fs::remove(temp, ec);
                            return false;
                        }
                        if(got == 0)
                            break;
                    }

                    if(::fsync(out.get()) != 0)
                    {
                        setError(error, errnoText("cannot sync", temp.string()));
                        fs::remove(temp, ec);
                        return false;
                    }
                }

                if(::rename(temp.c_str(), to.c_str()) != 0)
                {
                    setError(error, errnoText("cannot publish", to.string()));
                    fs::remove(temp, ec);
                    return false;
                }
                syncDirectory(to.parent_path());
                return true;
            }

            bool parseLine(std::string const& line, JournalEntry* entry)
            {
                std::istringstream fields(line);
                std::string        kind;
                if(!(fields >> kind))
                    return false;

                if(kind == "R")
                {
                    std::string format;
                    if(!(fields >> entry->payload.id.datHash >> entry->payload.id.coHash >> format))
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
                         >> e.batch >> e.k))
                        return false;
                    entry->kind = JournalEntry::Kind::Exact;
                    return isHash(e.id.datHash) && isHash(e.id.coHash) && e.payloadIndex >= 0;
                }

                return false;
            }

            /** Parses complete lines; a trailing partial line is left unread. */
            void parseJournal(std::string const&         text,
                              uint64_t                   offset,
                              std::vector<JournalEntry>* entries,
                              uint64_t*                  end)
            {
                size_t start = 0;
                for(;;)
                {
                    const size_t newline = text.find('\n', start);
                    if(newline == std::string::npos)
                        break;

                    JournalEntry entry;
                    if(parseLine(text.substr(start, newline - start), &entry))
                        entries->push_back(std::move(entry));
                    start = newline + 1;
                }
                *end = offset + start;
            }

            /**
             * Appends one line to a journal opened with O_APPEND and locked
             * exclusively. A partial last line, left by a writer that crashed,
             * is terminated first so the new record starts on its own line.
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
                    return false;
                }
                return true;
            }

            /**
             * Opens and flocks a journal: exclusively for appending, creating it
             * if absent, or shared for reading. A missing journal opened for
             * reading succeeds and leaves `out` empty.
             */
            bool openJournal(std::string const& path, bool exclusive, Fd* out, std::string* error)
            {
                const int flags = exclusive ? (O_RDWR | O_CREAT | O_APPEND | O_CLOEXEC)
                                            : (O_RDONLY | O_CLOEXEC);
                const int fd    = ::open(path.c_str(), flags, 0644);
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
                if(::fstat(fd, &st) != 0 || (st.st_mode & (S_IWGRP | S_IWOTH)))
                {
                    setError(error, path + " is group- or world-writable");
                    return false;
                }

                while(::flock(fd, exclusive ? LOCK_EX : LOCK_SH) != 0)
                {
                    if(errno != EINTR)
                    {
                        setError(error, errnoText("cannot lock", path));
                        return false;
                    }
                }
                return true;
            }

            std::string payloadLine(PayloadRecord const& record)
            {
                return "R " + record.id.datHash + " " + record.id.coHash + " "
                       + (record.compressed ? "dat.zlib" : "dat");
            }

            std::string exactLine(ExactRecord const& record)
            {
                std::ostringstream line;
                line << "E " << record.id.datHash << ' ' << record.id.coHash << ' '
                     << record.payloadIndex << ' ' << record.m << ' ' << record.n << ' '
                     << record.batch << ' ' << record.k;
                return line.str();
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
                                 "context; user kernels will not be loaded or saved."
                              << std::endl;
                    return nullptr;
                }

                const char* env = rocblaslt_secure_getenv_impl(kEnvName, privileged);
                if(!env || !*env)
                    return nullptr;

                std::string error;
                Store*      opened = open(env, &error);
                if(!opened)
                    std::cerr << "\nrocblaslt warning: not using user kernel library " << env
                              << ": " << error << ". User kernels will not be loaded or saved."
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

            return new Store(rootPath.string());
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

        std::string Store::journalPath(std::string const& arch, std::string* error) const
        {
            const bool safe = !arch.empty() && arch.find('/') == std::string::npos && arch != "."
                              && arch != ".." && arch != kObjectsDir;
            if(!safe)
            {
                setError(error, "invalid architecture name " + arch);
                return {};
            }
            return (fs::path(m_root) / arch / kJournalName).string();
        }

        bool Store::storeObjects(std::string const&   shardFile,
                                 std::string const&   coPath,
                                 PayloadRecord const& record,
                                 std::string*         error) const
        {
            // Both objects are durable before any record names them.
            return copyDurable(shardFile, shardPath(record), error)
                   && copyDurable(coPath, codeObjectPath(record.id), error);
        }

        bool Store::appendPayload(std::string const&   arch,
                                  PayloadRecord const& record,
                                  std::string*         error) const
        {
            const std::string path = journalPath(arch, error);
            if(path.empty() || !makePrivateDirectory(fs::path(path).parent_path(), error))
                return false;

            Fd fd;
            if(!openJournal(path, true, &fd, error))
                return false;

            // Re-read under the lock: another process may have just added it.
            std::string text;
            if(!readFrom(fd.get(), 0, &text))
            {
                setError(error, errnoText("cannot read", path));
                return false;
            }
            std::vector<JournalEntry> entries;
            uint64_t                  end = 0;
            parseJournal(text, 0, &entries, &end);
            for(auto const& entry : entries)
                if(entry.kind == JournalEntry::Kind::Payload && entry.payload.id == record.id)
                    return true;

            return appendLine(fd.get(), path, payloadLine(record), error);
        }

        bool Store::appendExact(std::string const& arch,
                                ExactRecord const& record,
                                std::string*       error) const
        {
            const std::string path = journalPath(arch, error);
            if(path.empty() || !makePrivateDirectory(fs::path(path).parent_path(), error))
                return false;

            Fd fd;
            return openJournal(path, true, &fd, error)
                   && appendLine(fd.get(), path, exactLine(record), error);
        }

        bool Store::read(std::string const&         arch,
                         uint64_t                   offset,
                         std::vector<JournalEntry>* entries,
                         uint64_t*                  end,
                         std::string*               error) const
        {
            entries->clear();
            *end = offset;

            const std::string path = journalPath(arch, error);
            if(path.empty())
                return false;

            std::error_code ec;
            const fs::path  dir = fs::path(path).parent_path();
            if(!fs::exists(dir, ec))
                return true;
            if(!checkPrivate(dir, error))
                return false;

            Fd fd;
            if(!openJournal(path, false, &fd, error))
                return false;
            if(fd.get() < 0)
                return true;

            std::string text;
            if(!readFrom(fd.get(), offset, &text))
            {
                setError(error, errnoText("cannot read", path));
                return false;
            }
            parseJournal(text, offset, entries, end);
            return true;
        }
    } // namespace user_kernel_store
} // namespace rocblaslt
