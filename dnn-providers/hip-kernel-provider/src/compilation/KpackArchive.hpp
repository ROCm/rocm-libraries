// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace hip_kernel_provider::compilation
{

/// Which step of reading a kpack archive failed. Carried separately from the raw
/// reader code so a caller can phrase a diagnostic per stage: "wrong GPU" and "wrong
/// toc_key" are the same reader error family but send a reader to entirely different
/// places, and flattening both into one string loses that.
enum class KpackLoadStage
{
    OPEN_ARCHIVE,
    ARCH_LOOKUP,
    ENTRY_LOOKUP,
    DECOMPRESS,
};

/// A staged reader failure: the step that failed plus the raw kpack_error_t value and
/// its spelling. Mirrors the struct in ALMIOPEN-2420's KpackModuleLoader so the two
/// readers stay easy to reconcile.
struct KpackError
{
    KpackLoadStage stage = KpackLoadStage::OPEN_ARCHIVE;
    int code = 0;
    std::string codeName;
    /// The archive file itself was not there, as opposed to being there and unreadable.
    /// AC #7 wants those two phrased differently, and this saves the caller either
    /// including kpack.h to compare the enum or matching on codeName text.
    bool archiveAbsent = false;
};

/// An owning decompressed code object. Frees through kpack_free_kernel, which is not
/// operator delete, so this cannot be a vector or a unique_ptr with the default deleter.
class KpackCodeObject
{
public:
    KpackCodeObject() = default;
    KpackCodeObject(void* data, size_t size);
    ~KpackCodeObject();

    KpackCodeObject(const KpackCodeObject&) = delete;
    KpackCodeObject& operator=(const KpackCodeObject&) = delete;
    KpackCodeObject(KpackCodeObject&& other) noexcept;
    KpackCodeObject& operator=(KpackCodeObject&& other) noexcept;

    const void* data() const
    {
        return _data;
    }

    size_t size() const
    {
        return _size;
    }

    bool empty() const
    {
        return _data == nullptr || _size == 0;
    }

private:
    void* _data = nullptr;
    size_t _size = 0;
};

/// RAII owner of a kpack_archive_t.
///
/// This class and its .cpp are the only place in the provider that sees
/// <rocm_kpack/kpack.h>. Confining the C dependency here keeps it out of every other
/// header and makes an eventual switch to a different container a one-file change.
///
/// Every entry point reports failure through KpackError rather than throwing, because
/// the caller (KpackKernelLoader) is the layer that knows the descriptor label and the
/// symbol that AC #7 requires every message to name.
class KpackArchive
{
public:
    KpackArchive() = default;
    ~KpackArchive();

    KpackArchive(const KpackArchive&) = delete;
    KpackArchive& operator=(const KpackArchive&) = delete;
    KpackArchive(KpackArchive&& other) noexcept;
    KpackArchive& operator=(KpackArchive&& other) noexcept;

    /// Opens `path`. On failure returns false and fills `error` with stage
    /// OPEN_ARCHIVE. Does not check that the path exists first -- the reader's own
    /// FILE_NOT_FOUND is the authority, and a pre-check would race it.
    bool open(const std::filesystem::path& path, KpackError& error);

    bool isOpen() const
    {
        return _archive != nullptr;
    }

    /// Every architecture the archive holds a binary for. On failure returns false and
    /// fills `error` with stage ARCH_LOOKUP.
    bool architectures(std::vector<std::string>& arches, KpackError& error) const;

    /// Decompresses the binary stored under `tocKey` for `arch`. On failure returns
    /// false with stage ENTRY_LOOKUP (the key or the arch is absent) or DECOMPRESS.
    ///
    /// `tocKey` is passed to the reader as its `binary_name`: kpack names the entry by
    /// the content hash of (source, build), which is exactly what the descriptor's
    /// toc_key field carries.
    bool codeObject(const std::string& tocKey,
                    const std::string& arch,
                    KpackCodeObject& codeObject,
                    KpackError& error) const;

private:
    void close();

    /// void* rather than kpack_archive_t so this header does not include kpack.h.
    void* _archive = nullptr;
};

} // namespace hip_kernel_provider::compilation

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
