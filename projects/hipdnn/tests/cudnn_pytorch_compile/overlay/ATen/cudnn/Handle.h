#pragma once
#include <ATen/hip/ATenHIPGeneral.h>
#include <ATen/hipdnn/cudnn_compat.h>
#if !AT_HIPDNN_ENABLED()
#error "PyTorch was configured without USE_HIPDNN"
#endif
