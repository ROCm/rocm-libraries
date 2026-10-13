// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/*
 * LD_PRELOAD release gate for tensilelite-client, used by run_tensile_with_agt.py.
 *
 * The client's first kernel launch through hipExtModuleLaunchKernel, hipModuleLaunchKernel or hipDrvLaunchKernelEx
 * blocks until 'g' is read from AGTGATE_FIFO, so clients on several GPUs load and initialize on their own and then
 * start their warmups and timed runs together. Any other byte makes the client exit with 75.
 *
 * Environment (without AGTGATE_FIFO the shim only applies AGTGATE_ROCR_VISIBLE_DEVICES):
 *   AGTGATE_FIFO     FIFO to wait on.
 *   AGTGATE_READY    created once the client waits at the gate; AGTGATE_READY".armed" is created when the shim
 *                    loads, so the runner can tell the gate is in effect.
 *   AGTGATE_ROCR_VISIBLE_DEVICES
 *                    copied into ROCR_VISIBLE_DEVICES when the shim loads, before HIP reads it. The runner sets
 *                    ROCR_VISIBLE_DEVICES to a GPU that does not exist, so a client without the shim sees no GPU.
 *   AGTGATE_PARENT   pid of the runner: the client exits with 78 once its parent is another process (it is also
 *                    killed when the runner dies).
 *   AGTGATE_TIMEOUT  seconds after the shim loads that a client may still wait at the gate before it exits with 77
 *                    (default 3600).
 * A HIP symbol that cannot be resolved exits with 76. Gate events go to stderr as "AGTGATE ..." lines with
 * CLOCK_REALTIME stamps.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/prctl.h>
#include <time.h>
#include <unistd.h>

typedef int   hipError_t;
typedef void* hipFunction_t;
typedef void* hipStream_t;
typedef void* hipEvent_t;

static pthread_once_t once     = PTHREAD_ONCE_INIT;
static int            armed    = 0; /* AGTGATE_FIFO set and the constructor ran */
static pid_t          parent   = 0;
static double         deadline = 0;

static double seconds(clockid_t clock)
{
    struct timespec ts;
    clock_gettime(clock, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void touch(const char* path)
{
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if(fd >= 0)
    {
        dprintf(fd, "%.6f\n", seconds(CLOCK_REALTIME));
        close(fd);
    }
}

/* Exit unless the runner that started us is still our parent and the gate deadline has not passed. */
static void check_alive(void)
{
    if(parent && getppid() != parent)
    {
        fprintf(stderr, "AGTGATE runner gone %.6f\n", seconds(CLOCK_REALTIME));
        _exit(78);
    }
    if(seconds(CLOCK_MONOTONIC) > deadline)
    {
        fprintf(stderr, "AGTGATE timeout %.6f\n", seconds(CLOCK_REALTIME));
        _exit(77);
    }
}

__attribute__((constructor)) static void arm(void)
{
    const char* visible = getenv("AGTGATE_ROCR_VISIBLE_DEVICES");
    if(visible)
        setenv("ROCR_VISIBLE_DEVICES", visible, 1);
    if(!getenv("AGTGATE_FIFO"))
        return;
    prctl(PR_SET_PDEATHSIG, SIGKILL);
    const char* p = getenv("AGTGATE_PARENT");
    parent        = p ? (pid_t)atol(p) : 0;
    const char* t = getenv("AGTGATE_TIMEOUT");
    deadline      = seconds(CLOCK_MONOTONIC) + (t ? atof(t) : 3600.0);
    check_alive(); /* the runner may have died before PR_SET_PDEATHSIG took effect */
    const char* ready = getenv("AGTGATE_READY");
    if(ready)
    {
        char path[4096];
        snprintf(path, sizeof(path), "%s.armed", ready);
        touch(path);
    }
    armed = 1;
}

static void* real(const char* name, const char* version)
{
    void* f = dlvsym(RTLD_NEXT, name, version);
    if(!f)
        f = dlsym(RTLD_NEXT, name);
    if(!f)
    {
        fprintf(stderr, "AGTGATE cannot resolve %s\n", name);
        _exit(76);
    }
    return f;
}

static void wait_for_release(void)
{
    if(!armed)
        return;
    const char* fifo  = getenv("AGTGATE_FIFO");
    const char* ready = getenv("AGTGATE_READY");
    /* Nonblocking open: the reader exists as soon as it is open, so the runner's release can never get ENXIO. */
    int fd = open(fifo, O_RDONLY | O_NONBLOCK | O_CLOEXEC);
    if(fd < 0)
    {
        fprintf(stderr, "AGTGATE cannot open %s\n", fifo);
        _exit(75);
    }
    if(ready)
        touch(ready);
    fprintf(stderr, "AGTGATE waiting %.6f\n", seconds(CLOCK_REALTIME));
    char c = 0;
    for(;;)
    {
        struct pollfd p = {fd, POLLIN, 0};
        int           n = poll(&p, 1, 1000);
        if(n > 0)
        {
            ssize_t r = read(fd, &c, 1);
            if(r == 1)
                break;
            if(r == 0)
                usleep(50000); /* a writer came and went without a byte: avoid spinning on POLLHUP */
        }
        else if(n < 0 && errno != EINTR)
            _exit(75);
        check_alive();
    }
    close(fd);
    if(c != 'g')
    {
        fprintf(stderr, "AGTGATE aborted %.6f\n", seconds(CLOCK_REALTIME));
        _exit(75);
    }
    fprintf(stderr, "AGTGATE released %.6f\n", seconds(CLOCK_REALTIME));
}

hipError_t hipExtModuleLaunchKernel(hipFunction_t f, uint32_t gx, uint32_t gy, uint32_t gz,
                                    uint32_t lx, uint32_t ly, uint32_t lz, size_t sharedMemBytes,
                                    hipStream_t stream, void** kernelParams, void** extra,
                                    hipEvent_t startEvent, hipEvent_t stopEvent, uint32_t flags)
{
    static hipError_t (*fn)(hipFunction_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                            size_t, hipStream_t, void**, void**, hipEvent_t, hipEvent_t, uint32_t);
    if(!fn)
        fn = real("hipExtModuleLaunchKernel", "hip_4.2");
    pthread_once(&once, wait_for_release);
    return fn(f, gx, gy, gz, lx, ly, lz, sharedMemBytes, stream, kernelParams, extra, startEvent,
              stopEvent, flags);
}

hipError_t hipModuleLaunchKernel(hipFunction_t f, uint32_t gx, uint32_t gy, uint32_t gz, uint32_t bx,
                                 uint32_t by, uint32_t bz, uint32_t sharedMemBytes, hipStream_t stream,
                                 void** kernelParams, void** extra)
{
    static hipError_t (*fn)(hipFunction_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                            uint32_t, hipStream_t, void**, void**);
    if(!fn)
        fn = real("hipModuleLaunchKernel", "hip_4.2");
    pthread_once(&once, wait_for_release);
    return fn(f, gx, gy, gz, bx, by, bz, sharedMemBytes, stream, kernelParams, extra);
}

/* config is const HIP_LAUNCH_CONFIG*; only passed through */
hipError_t hipDrvLaunchKernelEx(const void* config, hipFunction_t f, void** kernelParams, void** extra)
{
    static hipError_t (*fn)(const void*, hipFunction_t, void**, void**);
    if(!fn)
        fn = real("hipDrvLaunchKernelEx", "hip_6.5");
    pthread_once(&once, wait_for_release);
    return fn(config, f, kernelParams, extra);
}
