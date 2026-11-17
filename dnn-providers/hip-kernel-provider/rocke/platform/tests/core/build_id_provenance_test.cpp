// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Verify the linked archive exposes the stamped, not legacy, build-id
 * accessors. */
#include "rocke/rocke_build_id.h"

#include <stdio.h>
#include <string.h>

int main(void)
{
    const char* id = rocke_build_id();
    const char* version = rocke_engine_version();
    if(id == NULL || strlen(id) != 16 || strspn(id, "0123456789abcdef") != 16)
    {
        fprintf(stderr, "engine build-id is not a source-content hash\n");
        return 1;
    }
    if(version == NULL || strstr(version, "unknown") != NULL)
    {
        fprintf(stderr, "engine version is not configured\n");
        return 1;
    }
    return 0;
}
