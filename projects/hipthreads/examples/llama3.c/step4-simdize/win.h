#ifndef _WIN_H_
#define _WIN_H_

#define WIN32_LEAN_AND_MEAN
#include <stdint.h>
#include <time.h>
#include <windows.h>

#define ssize_t int64_t
#define ftell _ftelli64

#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0501
#endif

#include <sys/types.h>

// POSIX file calls (open/close/read/lseek, O_RDONLY) live in <unistd.h> on Linux.
// On Windows they are the underscore-prefixed CRT functions in <io.h>/<fcntl.h>.
// Only the OS handle behind the fd is used by mmap (via _get_osfhandle), so text/
// binary mode is irrelevant here.
#include <io.h>
#include <fcntl.h>
#define open _open
#define close _close
#define read _read
#define lseek _lseek
#ifndef O_RDONLY
#define O_RDONLY _O_RDONLY
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define PROT_NONE 0
#define PROT_READ 1
#define PROT_WRITE 2
#define PROT_EXEC 4

#define MAP_FILE 0
#define MAP_SHARED 1
#define MAP_PRIVATE 2
#define MAP_TYPE 0xf
#define MAP_FIXED 0x10
#define MAP_ANONYMOUS 0x20
#define MAP_ANON MAP_ANONYMOUS

#define MAP_FAILED ((void *)-1)

#define MS_ASYNC 1
#define MS_SYNC 2
#define MS_INVALIDATE 4

void *mmap(void *addr, size_t len, int prot, int flags, int fildes, ssize_t off);
int munmap(void *addr, size_t len);
int mprotect(void *addr, size_t len, int prot);
int msync(void *addr, size_t len, int flags);
int mlock(const void *addr, size_t len);
int munlock(const void *addr, size_t len);

#ifdef __cplusplus
};
#endif

#endif /*  _WIN_H_ */
