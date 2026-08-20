// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// LineStore is the generic, record-format-agnostic append-only shard file abstraction that
// layer 1 exposes to every on-disk cache built on top of it (winner cache, autotune cache,
// and future consumers alike). A "shard" is a single file: a bare version string on its
// first line, followed by zero or more caller-defined record lines, one record per line.
//
// LineStore knows nothing about what a record means. It treats every line after the first
// as an opaque std::string and hands parsing/formatting to the caller via a callback or
// template parameter. In particular, this header MUST NOT include or reference
// nlohmann/json or any other JSON type -- data_sdk has no JSON dependency today and must
// not gain one through this file. Callers that want JSON-Lines records (as layer 3 does)
// supply their own encode/decode callbacks from a layer that already depends on JSON.
//
// Concurrency: each shard has its own advisory lock, acquired for the duration of an
// append and released afterward. Locking is per-file, not per-line -- two writers append
// to the same shard serially, never interleaved or torn.
//
// Failure handling is fail-soft throughout: a shard that cannot be opened, locked, or read
// is reported through the return value, never through an exception. A version mismatch on
// read is a caller-visible decline of the whole file, not a throw and not a crash. A single
// line that the caller's parse callback rejects is skipped and does not affect any other
// line in the file -- it never poisons the read of an otherwise-good shard.
//
// Last-line-wins resolution over duplicate-keyed records (e.g. the same logical key
// appended twice because two processes both missed the same cache lookup) is the caller's
// job: LineStore guarantees line order and append atomicity, nothing about record identity.

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace hipdnn_data_sdk::utilities
{

/// Outcome of opening (or creating) a shard file, and of acquiring its lock.
///
/// Every value other than Ok is a caller-visible decline: the caller should treat the
/// shard as unavailable for this operation and fall back to whatever behavior it would
/// have used with no on-disk cache at all, not treat it as fatal.
enum class LineStoreStatus
{
    Ok,
    OpenFailed,
    LockFailed,
    IoError,
    VersionMismatch,
};

/// A handle to one open, version-checked shard file.
///
/// Obtained only via openLineStore(); a LineStoreShard is either a live handle to a
/// shard whose version line matched, or it does not exist -- callers never see a
/// half-open or version-mismatched shard through this type.
class LineStoreShard
{
public:
    LineStoreShard() = delete;

    /// True while this handle holds the shard's advisory lock.
    bool isLocked() const noexcept;

    // TODO(Stream A): implement in Phase 2
};

/// Opens the shard file at @p path for record access, creating it (and writing
/// @p expectedVersion as its bare first line) if it does not yet exist.
///
/// On an existing file, the first line is compared against @p expectedVersion. A mismatch
/// is reported as LineStoreStatus::VersionMismatch and no shard is returned -- this is a
/// decline, not a throw, since a version bump from a newer build must never crash an older
/// one reading the same cache directory.
///
/// @param path Path to the shard file; parent directories are assumed to already exist.
/// @param expectedVersion Bare version string written to (or checked against) the first
///     line. Carries no format requirement beyond being one line of text.
/// @return The open shard and LineStoreStatus::Ok on success; std::nullopt and a non-Ok
///     status describing the failure otherwise. Never throws.
inline std::pair<std::optional<LineStoreShard>, LineStoreStatus>
    openLineStore(const std::filesystem::path& path, std::string_view expectedVersion);
// TODO(Stream A): implement in Phase 2

/// Acquires @p shard's per-shard advisory lock, blocking until it is held or acquisition
/// fails. Never throws; a failure to lock (e.g. an incompatible external lock holder, or a
/// filesystem that does not support the underlying primitive) is reported as
/// LineStoreStatus::LockFailed so the caller can decline gracefully.
inline LineStoreStatus lockLineStore(LineStoreShard& shard);
// TODO(Stream A): implement in Phase 2

/// Releases a lock previously acquired by lockLineStore(). A no-op, not an error, if
/// @p shard is not currently locked.
inline void unlockLineStore(LineStoreShard& shard) noexcept;
// TODO(Stream A): implement in Phase 2

/// Appends one caller-formatted record line to @p shard.
///
/// @p shard must already hold the lock (see lockLineStore()) -- appendLine() does not
/// acquire it implicitly, so a caller that needs to re-read the file under the lock before
/// deciding what to append (e.g. to detect a concurrent duplicate write) can do so between
/// locking and appending. @p line must not itself contain a newline; the caller is
/// responsible for producing a single logical line (embedding its own delimiters, escaping,
/// or encoding as needed -- LineStore does not inspect or validate line content beyond
/// writing it followed by exactly one newline).
///
/// @return LineStoreStatus::Ok on success; LineStoreStatus::IoError on any write failure.
/// Never throws.
inline LineStoreStatus appendLine(LineStoreShard& shard, std::string_view line);
// TODO(Stream A): implement in Phase 2

/// Reads every record line from @p shard (i.e. every line after the version line) and
/// hands each one to @p parseLine in file order.
///
/// A line for which @p parseLine returns std::nullopt is skipped: it is not returned in
/// the result, and it does not stop later lines from being read or cause the call to
/// report failure. This is the skip-malformed-line contract -- a single corrupt or
/// forward-incompatible record must never poison the rest of an otherwise-good shard.
/// Resolving duplicate-keyed records (last-line-wins or otherwise) is left entirely to the
/// caller; this function preserves file order and nothing else.
///
/// @tparam ParseLine The caller's callback type, deduced from the argument. Declared as a
///     plain callable rather than a `std::function` parameter: `std::function` is a
///     non-deduced context, so a `std::function<std::optional<Record>(std::string_view)>`
///     parameter forces every caller to name Record explicitly or wrap its lambda. Taking
///     the callable directly also avoids `std::function`'s type erasure and its potential
///     allocation on a path that runs once per shard read.
/// @tparam Record The caller's record type, deduced from @p parseLine's
///     `std::optional<Record>` return type.
/// @param shard The shard to read. Does not require the lock to be held: a snapshot read
///     racing a concurrent append is expected to observe either the old or the new state
///     of the file, never a torn line, because appendLine() only ever adds whole lines.
/// @param parseLine Caller-supplied callback converting one raw line into
///     std::optional<Record>; std::nullopt skips the line.
/// @return Every successfully parsed record, in file order, and LineStoreStatus::Ok; or an
///     empty vector and a non-Ok status if the shard itself could not be read (a status
///     level failure, distinct from a per-line parse rejection). Never throws.
template <typename ParseLine,
          typename Record = typename std::invoke_result_t<ParseLine, std::string_view>::value_type>
std::pair<std::vector<Record>, LineStoreStatus> readAllLines(const LineStoreShard& shard,
                                                             ParseLine parseLine);
// TODO(Stream A): implement in Phase 2

} // namespace hipdnn_data_sdk::utilities
