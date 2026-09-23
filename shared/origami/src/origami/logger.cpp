// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "origami/logger.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>

static origami::LogFormat format_from_path(const std::string& path) {
    auto dot = path.rfind('.');
    if (dot != std::string::npos) {
        auto ext = path.substr(dot);
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        if (ext == ".csv") return origami::LogFormat::CSV;
    }
    return origami::LogFormat::TEXT;
}

namespace origami {

// ---------------------------------------------------------------------------
// Per-thread CSV row state
// ---------------------------------------------------------------------------

namespace {
struct CsvThreadState {
    bool row_in_progress = false;
    std::vector<std::pair<std::string, std::string>> current_row;
};
}  // namespace

static thread_local CsvThreadState tl_csv_state;

// ---------------------------------------------------------------------------
// Logger
// ---------------------------------------------------------------------------

Logger::Logger() : enabled_{false}, format_{LogFormat::TEXT}, header_written_{false} {
    const char* file_path = std::getenv("ORIGAMI_LOG_FILE");
    if (file_path == nullptr || file_path[0] == '\0') return;

    file_path_ = file_path;

    format_ = format_from_path(file_path_);
    if (format_ == LogFormat::CSV) {
        enabled_.store(true, std::memory_order_release);
    } else {
        format_ = LogFormat::TEXT;
        log_file_.open(file_path_, std::ios::out | std::ios::app);
        if (log_file_.is_open()) {
            enabled_.store(true, std::memory_order_release);
            log(LogLevel::INFO,
                "Origami logger initialized, writing to: " + file_path_,
                __FILE__, __LINE__);
        } else {
            std::cerr << "Warning: Failed to open log file: " << file_path_ << std::endl;
        }
    }
}

Logger::~Logger() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!enabled_.load(std::memory_order_relaxed)) return;

    if (format_ == LogFormat::TEXT) {
        if (log_file_.is_open()) {
            log_file_ << "[INFO ] " << "logger.cpp" << ":" << __LINE__
                      << " - Logger shutting down" << std::endl;
            log_file_.close();
        }
    } else {
        if (!rows_.empty()) {
            flush_csv_locked();
        }
    }
}

Logger& Logger::instance() {
    static Logger logger;
    return logger;
}

void Logger::log(LogLevel level, const std::string& message, const char* file, int line) {
    if (!enabled_.load(std::memory_order_acquire)) return;

    if (format_ == LogFormat::TEXT) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!enabled_.load(std::memory_order_relaxed) || !log_file_.is_open()) return;

        const char* filename = file;
        for (const char* p = file; *p; p++) {
            if (*p == '/' || *p == '\\') {
                filename = p + 1;
            }
        }

        log_file_ << "[" << level_to_string(level) << "] "
                  << filename << ":" << line << " - "
                  << message << std::endl;
    } else if (level == LogLevel::DEBUG) {
        process_debug_message(message);
    }
}

void Logger::flush() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!enabled_.load(std::memory_order_relaxed)) return;

    if (format_ == LogFormat::TEXT) {
        if (log_file_.is_open()) log_file_.flush();
    } else {
        if (!rows_.empty()) flush_csv_locked();
    }
}

void Logger::update_from_env() {
    std::lock_guard<std::mutex> lock(mutex_);

    const char* file_path = std::getenv("ORIGAMI_LOG_FILE");
    const std::string new_path = (file_path && file_path[0] != '\0') ? file_path : "";
    LogFormat new_format = new_path.empty() ? LogFormat::TEXT : format_from_path(new_path);

    if (enabled_.load(std::memory_order_relaxed)) {
        if (format_ == LogFormat::CSV && !rows_.empty()) {
            flush_csv_locked();
        }
    }
    if (log_file_.is_open()) {
        log_file_.close();
    }

    if (new_path != file_path_ || new_format != format_) {
        rows_.clear();
        columns_.clear();
        column_index_.clear();
        header_written_ = false;
    }

    file_path_ = new_path;
    format_ = new_format;
    enabled_.store(false, std::memory_order_release);

    if (new_path.empty()) return;

    if (format_ == LogFormat::TEXT) {
        log_file_.open(file_path_, std::ios::out | std::ios::app);
        if (!log_file_.is_open()) {
            std::cerr << "Warning: Failed to open log file: " << file_path_ << std::endl;
            return;
        }
    }

    enabled_.store(true, std::memory_order_release);
}

const char* Logger::level_to_string(LogLevel level) const {
    switch (level) {
        case LogLevel::DEBUG:   return "DEBUG";
        case LogLevel::INFO:    return "INFO ";
        case LogLevel::WARNING: return "WARN ";
        case LogLevel::ERROR:   return "ERROR";
        default:                return "UNKN ";
    }
}

// ---------------------------------------------------------------------------
// CSV-mode methods
// ---------------------------------------------------------------------------

void Logger::process_debug_message(const std::string& message) {
    if (!enabled_.load(std::memory_order_acquire)) return;

    static constexpr const char* BEGIN_MARKER = "======== Origami Debug Info ========";
    static constexpr const char* END_MARKER   = "=================================";

    if (message.find(BEGIN_MARKER) != std::string::npos) {
        begin_row();
        return;
    }
    if (message.find(END_MARKER) != std::string::npos) {
        end_row();
        return;
    }

    if (!tl_csv_state.row_in_progress) return;

    auto pos = message.find(": ");
    if (pos != std::string::npos) {
        record(message.substr(0, pos), message.substr(pos + 2));
    }
}

void Logger::begin_row() {
    tl_csv_state.current_row.clear();
    tl_csv_state.row_in_progress = true;
}

void Logger::record(const std::string& key, const std::string& value) {
    if (tl_csv_state.row_in_progress) {
        tl_csv_state.current_row.emplace_back(key, value);
    }
}

void Logger::end_row() {
    if (!tl_csv_state.row_in_progress) return;
    tl_csv_state.row_in_progress = false;

    std::lock_guard<std::mutex> lock(mutex_);

    for (const auto& [key, _] : tl_csv_state.current_row) {
        if (column_index_.find(key) == column_index_.end()) {
            column_index_[key] = columns_.size();
            columns_.push_back(key);
        }
    }

    std::vector<std::string> row(columns_.size());
    for (const auto& [key, value] : tl_csv_state.current_row) {
        row[column_index_[key]] = value;
    }
    rows_.push_back(std::move(row));

    tl_csv_state.current_row.clear();
}

std::string Logger::escape_csv(const std::string& field) {
    if (field.find_first_of(",\"\n\r") == std::string::npos) {
        return field;
    }
    std::string escaped;
    escaped.reserve(field.size() + 4);
    escaped += '"';
    for (char c : field) {
        if (c == '"') escaped += "\"\"";
        else escaped += c;
    }
    escaped += '"';
    return escaped;
}

void Logger::flush_csv_locked() {
    if (rows_.empty()) return;

    std::ofstream file(file_path_, std::ios::out | std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Warning: Failed to open CSV file: " << file_path_
                   << ". Dropping " << rows_.size() << " buffered row(s)." << std::endl;
        rows_.clear();
        return;
    }

    if (!header_written_) {
        bool file_empty = (file.tellp() == 0);
        if (file_empty) {
            for (size_t i = 0; i < columns_.size(); ++i) {
                if (i > 0) file << ",";
                file << escape_csv(columns_[i]);
            }
            file << "\n";
        }
        header_written_ = true;
    }

    for (const auto& row : rows_) {
        for (size_t i = 0; i < columns_.size(); ++i) {
            if (i > 0) file << ",";
            if (i < row.size()) {
                file << escape_csv(row[i]);
            }
        }
        file << "\n";
    }

    rows_.clear();
}

} // namespace origami
