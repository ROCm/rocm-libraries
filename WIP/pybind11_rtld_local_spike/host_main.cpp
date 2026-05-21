// host_main.cpp - dlopen libpyplugin.so with RTLD_LOCAL|RTLD_NOW twice.
#include <dlfcn.h>

#include <cstdio>
#include <string>

using fn_t = int (*)(void);

static int run_once(const char* path, int round) {
    std::printf("[host] round %d: dlopen %s (RTLD_NOW|RTLD_LOCAL)\n", round, path);
    void* h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        std::fprintf(stderr, "[host] dlopen failed: %s\n", dlerror());
        return 101;
    }
    dlerror();
    auto fn = reinterpret_cast<fn_t>(dlsym(h, "run_python_smoke"));
    const char* err = dlerror();
    if (err) {
        std::fprintf(stderr, "[host] dlsym failed: %s\n", err);
        dlclose(h);
        return 102;
    }
    int rc = fn();
    std::printf("[host] round %d: run_python_smoke -> %d\n", round, rc);
    if (dlclose(h) != 0) {
        std::fprintf(stderr, "[host] dlclose failed: %s\n", dlerror());
    }
    return rc;
}

int main(int argc, char** argv) {
    std::string path = (argc > 1) ? argv[1] : "./libpyplugin.so";
    int rc1 = run_once(path.c_str(), 1);
    int rc2 = run_once(path.c_str(), 2);
    std::printf("[host] final: rc1=%d rc2=%d\n", rc1, rc2);
    return (rc1 == 0 && rc2 == 0) ? 0 : 1;
}
