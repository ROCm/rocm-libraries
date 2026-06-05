#pragma once

#include <sstream>
#include <string>

namespace stinkytofu {
/// A diagnostic message with location information.
/// Used by parsers to report errors, warnings, and notes to users.
class Diagnostic {
   public:
    enum class Level { Error, Warning, Note };

   private:
    Level level;
    std::string message;
    unsigned line;
    unsigned column;

   public:
    Diagnostic(Level lvl, std::string msg, unsigned l, unsigned c)
        : level(lvl), message(std::move(msg)), line(l), column(c) {}

    Level getLevel() const {
        return level;
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

    std::string format() const {
        std::ostringstream oss;
        oss << line << ":" << column << ": ";
        switch (level) {
            case Level::Error:
                oss << "error: ";
                break;
            case Level::Warning:
                oss << "warning: ";
                break;
            case Level::Note:
                oss << "note: ";
                break;
        }
        oss << message;
        return oss.str();
    }
};

}  // namespace stinkytofu
