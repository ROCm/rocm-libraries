#include "win.h"
#include <errno.h>
#include <io.h>

#ifndef FILE_MAP_EXECUTE
#define FILE_MAP_EXECUTE 0x0020
#endif

static int __map_mman_error(const uint32_t err, const int deferr) {
  if (err == 0)
    return 0;
  return err;
}

static uint32_t __map_mmap_prot_page(const int prot) {
  uint32_t protect = 0;

  if (prot == PROT_NONE)
    return protect;

  if ((prot & PROT_EXEC) != 0) {
    protect = ((prot & PROT_WRITE) != 0) ? PAGE_EXECUTE_READWRITE : PAGE_EXECUTE_READ;
  } else {
    protect = ((prot & PROT_WRITE) != 0) ? PAGE_READWRITE : PAGE_READONLY;
  }

  return protect;
}

static uint32_t __map_mmap_prot_file(const int prot) {
  uint32_t desiredAccess = 0;

  if (prot == PROT_NONE)
    return desiredAccess;

  if ((prot & PROT_READ) != 0)
    desiredAccess |= FILE_MAP_READ;
  if ((prot & PROT_WRITE) != 0)
    desiredAccess |= FILE_MAP_WRITE;
  if ((prot & PROT_EXEC) != 0)
    desiredAccess |= FILE_MAP_EXECUTE;

  return desiredAccess;
}

void *mmap(void *addr, size_t len, int prot, int flags, int fildes, ssize_t off) {
  HANDLE fm, h;
  void *map = MAP_FAILED;

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4293)
#endif

  const uint32_t dwFileOffsetLow = (uint32_t)(off & 0xFFFFFFFFL);
  const uint32_t dwFileOffsetHigh = (uint32_t)((off >> 32) & 0xFFFFFFFFL);
  const uint32_t protect = __map_mmap_prot_page(prot);
  const uint32_t desiredAccess = __map_mmap_prot_file(prot);

  const ssize_t maxSize = off + (ssize_t)len;

  const uint32_t dwMaxSizeLow = (uint32_t)(maxSize & 0xFFFFFFFFL);
  const uint32_t dwMaxSizeHigh = (uint32_t)((maxSize >> 32) & 0xFFFFFFFFL);

#ifdef _MSC_VER
#pragma warning(pop)
#endif

  errno = 0;

  if (len == 0
      || (flags & MAP_FIXED) != 0
      || prot == PROT_EXEC) {
    errno = EINVAL;
    return MAP_FAILED;
  }

  h = ((flags & MAP_ANONYMOUS) == 0) ? (HANDLE)_get_osfhandle(fildes) : INVALID_HANDLE_VALUE;

  if ((flags & MAP_ANONYMOUS) == 0 && h == INVALID_HANDLE_VALUE) {
    errno = EBADF;
    return MAP_FAILED;
  }

  fm = CreateFileMapping(h, NULL, protect, dwMaxSizeHigh, dwMaxSizeLow, NULL);

  if (fm == NULL) {
    errno = __map_mman_error(GetLastError(), EPERM);
    return MAP_FAILED;
  }

  map = MapViewOfFile(fm, desiredAccess, dwFileOffsetHigh, dwFileOffsetLow, len);

  CloseHandle(fm);

  if (map == NULL) {
    errno = __map_mman_error(GetLastError(), EPERM);
    return MAP_FAILED;
  }

  return map;
}

int munmap(void *addr, size_t len) {
  if (UnmapViewOfFile(addr))
    return 0;

  errno = __map_mman_error(GetLastError(), EPERM);

  return -1;
}

int msync(void *addr, size_t len, int flags) {
  if (FlushViewOfFile(addr, len))
    return 0;

  errno = __map_mman_error(GetLastError(), EPERM);

  return -1;
}

int mlock(const void *addr, size_t len) {
  if (VirtualLock((LPVOID)addr, len))
    return 0;

  errno = __map_mman_error(GetLastError(), EPERM);

  return -1;
}

int munlock(const void *addr, size_t len) {
  if (VirtualUnlock((LPVOID)addr, len))
    return 0;

  errno = __map_mman_error(GetLastError(), EPERM);

  return -1;
}
