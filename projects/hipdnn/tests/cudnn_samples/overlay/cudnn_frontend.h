#pragma once
#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>
namespace cudnn_frontend = hipdnn_frontend::compatibility::cudnn_frontend;
// Upstream declares `json` at global scope (the legacy samples call json::parse); the shim keeps
// it in its namespace.
#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
using json = cudnn_frontend::json;
#endif
