#ifndef RPP_TEST_DATA_PATHS_H
#define RPP_TEST_DATA_PATHS_H

#include <cstdlib>
#include <string>

namespace rpptest {

// Root of the test data tree. Compile-time default (RPP_TEST_DATA_DIR) can be
// overridden at runtime via the RPP_TEST_DATA_DIR environment variable.
inline std::string data_dir() {
    if (const char* env = std::getenv("RPP_TEST_DATA_DIR")) return env;
#ifdef RPP_TEST_DATA_DIR
    return RPP_TEST_DATA_DIR;
#else
    return ".";
#endif
}

}  // namespace rpptest

#endif  // RPP_TEST_DATA_PATHS_H
