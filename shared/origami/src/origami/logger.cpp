// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "origami/logger.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>

namespace origami {

Logger::Logger() : enabled_(false) {
    const char* log_file_path = std::getenv("ORIGAMI_LOG_FILE");
    
    if (log_file_path != nullptr && log_file_path[0] != '\0') {
        log_file_.open(log_file_path, std::ios::out | std::ios::app);
        
        if (log_file_.is_open()) {
            enabled_ = true;
            log(LogLevel::INFO, 
                "Origami logger initialized, writing to: " + std::string(log_file_path),
                __FILE__, __LINE__);
        } else {
            std::cerr << "Warning: Failed to open log file: " << log_file_path << std::endl;
        }
    }
}

Logger::~Logger() {
    if (enabled_ && log_file_.is_open()) {
        log(LogLevel::INFO, "Logger shutting down", __FILE__, __LINE__);
        log_file_.close();
    }
}

Logger& Logger::instance() {
    static Logger logger;
    return logger;
}

void Logger::log(LogLevel level, const std::string& message, const char* file, int line) {
    if (!enabled_) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    
    if (log_file_.is_open()) {
        const char* filename = file;
        for (const char* p = file; *p; p++) {
            if (*p == '/' || *p == '\\') {
                filename = p + 1;
            }
        }

        log_file_ << "[" << level_to_string(level) << "] "
                  << filename << ":" << line << " - "
                  << message << std::endl;
    }
}

void Logger::flush() {
    if (enabled_ && log_file_.is_open()) {
        std::lock_guard<std::mutex> lock(mutex_);
        log_file_.flush();
    }
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

} // namespace origami
