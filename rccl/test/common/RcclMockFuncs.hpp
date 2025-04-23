#include "info.h"
#include "comm.h"

void ncclDebugLog(ncclDebugLogLevel, unsigned long, char const*, int, char const*, ...) {};
ncclResult_t getHostName(char* hostname, int maxlen, const char delim) {
  return ncclSuccess;
}
