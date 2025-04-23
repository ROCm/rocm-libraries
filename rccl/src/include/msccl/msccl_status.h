/*************************************************************************
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT License.
 ************************************************************************/

#ifndef MSCCL_STATUS_H_
#define MSCCL_STATUS_H_

#include "msccl/msccl_struct.h"

bool mscclInitialized(int rank);

void mscclSetInitialized(int rank, bool initialized = true);

void mscclRemoveRank(int rank);

mscclStatus& mscclGetStatus(int rank);

mscclSavedProxyArgs& mscclGetSavedProxyArgs(int rank);

mscclThreadLocalStatus& mscclGetThreadLocalStatus();

#endif
