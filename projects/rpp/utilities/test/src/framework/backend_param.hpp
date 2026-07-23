#ifndef RPP_TEST_BACKEND_PARAM_H
#define RPP_TEST_BACKEND_PARAM_H

#include <rpp/rpp.h>

#include <string>
#include <vector>

namespace rpptest {

// Backends the suite can be instantiated against. HIP is added only when the
// installed rpp was built with the HIP backend.
inline std::vector<RppBackend> available_backends() {
    std::vector<RppBackend> backends = {RPP_HOST_BACKEND};
#if defined(RPP_TEST_HAVE_HIP) && RPP_TEST_HAVE_HIP
    backends.push_back(RPP_HIP_BACKEND);
#endif
    return backends;
}

inline std::string backend_name(RppBackend backend) {
    return backend == RPP_HIP_BACKEND ? "HIP" : "HOST";
}

}  // namespace rpptest

#endif  // RPP_TEST_BACKEND_PARAM_H
