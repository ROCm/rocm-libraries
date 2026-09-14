{
    "variables": {
        # Root of the ROCm/hipDNN SDK. Override with:
        #   HIPDNN_SDK=/path/to/_rocm_sdk_devel npm run build:native
        "hipdnn_sdk%": "<!(node -p \"process.env.HIPDNN_SDK || 'D:/develop/latest_wheels_nightly/Lib/site-packages/_rocm_sdk_devel'\")"
    },
    "targets": [
        {
            "target_name": "hipdnn_engine",
            "sources": ["src/engine.cpp", "src/graph_translate.cpp"],
            "include_dirs": [
                "<!@(node -p \"require('node-addon-api').include\")",
                "<(hipdnn_sdk)/include",
                "<(hipdnn_sdk)/include/hipdnn/frontend",
                "<(hipdnn_sdk)/include/hipdnn/backend",
                "<(hipdnn_sdk)/include/hipdnn/data_sdk",
                "<(hipdnn_sdk)/include/hipdnn/flatbuffers_sdk",
                "<(hipdnn_sdk)/include/hipdnn/plugin_sdk",
            ],
            "defines": [
                "NAPI_DISABLE_CPP_EXCEPTIONS",
                "HIPDNN_FRONTEND_SKIP_JSON_LIB",
                "__HIP_PLATFORM_AMD__",
            ],
            "cflags_cc": ["-std=c++17"],
            "conditions": [
                [
                    "OS=='win'",
                    {
                        "libraries": [
                            "<(hipdnn_sdk)/lib/hipdnn_backend.lib",
                            "<(hipdnn_sdk)/lib/amdhip64.lib",
                        ],
                        "msvs_settings": {
                            "VCCLCompilerTool": {
                                "AdditionalOptions": ["/std:c++17", "/EHsc"]
                            }
                        },
                    },
                ],
                [
                    "OS=='linux'",
                    {
                        "libraries": [
                            "-L<(hipdnn_sdk)/lib",
                            "-lhipdnn_backend",
                            "-lamdhip64",
                            "-Wl,-rpath,<(hipdnn_sdk)/lib",
                        ],
                        "include_dirs": ["/opt/rocm/include"],
                        "library_dirs": ["/opt/rocm/lib"],
                    },
                ],
            ],
        }
    ],
}
