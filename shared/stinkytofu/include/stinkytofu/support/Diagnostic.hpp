#pragma once

#include <optional>
#include <sstream>
#include <string>
#include <utility>

namespace stinkytofu {
/// What kind of failure a diagnostic reports, one per DiagnosticIds.def entry.
enum class DiagId {
#define DIAGNOSTIC_ID(ENUM, STR, DESC) ENUM,
#include "stinkytofu/support/DiagnosticIds.def"
};

/// The spelling format() prints for \p id, e.g. "unknown-mnemonic".
inline const char* diagIdName(DiagId id) {
    switch (id) {
#define DIAGNOSTIC_ID(ENUM, STR, DESC) /* NOLINT(bugprone-macro-parentheses) */ \
    case DiagId::ENUM:                                                          \
        return STR;
#include "stinkytofu/support/DiagnosticIds.def"
    }
    return "";
}

/// A place in built IR: a function, a block in it, and an instruction in that.
/// \p index counts a block's instructions as isIndexedInstruction() does, and
/// is -1 for a diagnostic about the whole block or function.
struct IRLocation {
    std::string function;
    std::string block;
    int index = -1;
};

/// A diagnostic message with location information.
/// Used by parsers to report errors, warnings, and notes to users, and by the
/// verifier, which locates them in IR rather than in text.
class Diagnostic {
   public:
    enum class Level { Error, Warning, Note };

   private:
    Level level;
    DiagId id;
    std::string message;
    unsigned line = 0;
    unsigned column = 0;
    std::optional<IRLocation> irLocation;

   public:
    Diagnostic(Level lvl, DiagId diagId, std::string msg, unsigned l, unsigned c)
        : level(lvl), id(diagId), message(std::move(msg)), line(l), column(c) {}

    Diagnostic(Level lvl, DiagId diagId, std::string msg, IRLocation where)
        : level(lvl), id(diagId), message(std::move(msg)), irLocation(std::move(where)) {}

    Level getLevel() const {
        return level;
    }
    DiagId getId() const {
        return id;
    }
    const std::string& getMessage() const {
        return message;
    }
    unsigned getLine() const {
        return line;
    }
    unsigned getColumn() const {
        return column;
    }
    /// Null for a diagnostic located in text.
    const IRLocation* getIRLocation() const {
        return irLocation ? &*irLocation : nullptr;
    }

    static const char* levelName(Level lvl) {
        switch (lvl) {
            case Level::Error:
                return "error";
            case Level::Warning:
                return "warning";
            case Level::Note:
                return "note";
        }
        return "";
    }

    std::string format() const {
        std::ostringstream oss;
        if (irLocation) {
            std::string where;
            auto add = [&where](const std::string& part) {
                if (!where.empty()) where += ' ';
                where += part;
            };
            if (!irLocation->function.empty()) add("@" + irLocation->function);
            if (!irLocation->block.empty()) add("^" + irLocation->block);
            if (irLocation->index >= 0) add("#" + std::to_string(irLocation->index));
            if (!where.empty()) oss << where << ": ";
        } else {
            oss << line << ":" << column << ": ";
        }
        oss << levelName(level) << ": " << message << " [" << diagIdName(id) << "]";
        return oss.str();
    }
};

}  // namespace stinkytofu
