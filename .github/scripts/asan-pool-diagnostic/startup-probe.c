// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

static void* load(const char* name) {
    printf("before dlopen %s\n", name);
    void* library = dlopen(name, RTLD_NOW | RTLD_GLOBAL);
    if (!library) {
        fprintf(stderr, "%s\n", dlerror());
        exit(1);
    }
    printf("after dlopen %s\n", name);
    return library;
}

static void* symbol(void* library, const char* name) {
    void* result = dlsym(library, name);
    if (!result) {
        fprintf(stderr, "%s: %s\n", name, dlerror());
        exit(1);
    }
    return result;
}

#define CALL(expression)                               \
    do {                                               \
        puts("before " #expression);                   \
        int status = (expression);                     \
        printf("after %s: %d\n", #expression, status); \
        if (status) return status;                     \
    } while (0)

int main(void) {
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);
    puts("probe entered main; matching ASAN runtime loaded before main");
    void* hip = load("libamdhip64.so");
    int (*initialize)(unsigned) = symbol(hip, "hipInit");
    int (*count_devices)(int*) = symbol(hip, "hipGetDeviceCount");
    int (*allocate)(void**, size_t) = symbol(hip, "hipMalloc");
    int (*clear)(void*, int, size_t) = symbol(hip, "hipMemset");
    int (*synchronize)(void) = symbol(hip, "hipDeviceSynchronize");
    int (*release)(void*) = symbol(hip, "hipFree");
    CALL(initialize(0));
    int count = 0;
    CALL(count_devices(&count));
    printf("device count: %d\n", count);
    void* allocation = NULL;
    CALL(allocate(&allocation, 4096));
    CALL(clear(allocation, 0, 4096));
    CALL(synchronize());
    CALL(release(allocation));
    void* blas = load("libhipblaslt.so");
    int (*create_handle)(void**) = symbol(blas, "hipblasLtCreate");
    int (*destroy_handle)(void*) = symbol(blas, "hipblasLtDestroy");
    void* handle = NULL;
    CALL(create_handle(&handle));
    CALL(destroy_handle(handle));
    puts("probe completed");
    return 0;
}
