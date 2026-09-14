{
    # hipdnn-config.gypi is generated next to this file by build-addon.cjs. It
    # resolves either an in-tree hipDNN build tree or an installed SDK into
    # hipdnn_include_dirs / hipdnn_defines / hipdnn_libraries, and names the
    # source and node-addon-api directories so this file works from a staged
    # copy in the build tree as well as from here.
    "includes": ["hipdnn-config.gypi"],
    "targets": [
        {
            "target_name": "hipdnn_engine",
            "sources": [
                "<(hipdnn_native_src_dir)/engine.cpp",
                "<(hipdnn_native_src_dir)/graph_translate.cpp",
            ],
            "include_dirs": [
                "<(hipdnn_napi_include_dir)",
                "<@(hipdnn_include_dirs)",
            ],
            "defines": [
                "NAPI_DISABLE_CPP_EXCEPTIONS",
                "__HIP_PLATFORM_AMD__",
                "<@(hipdnn_defines)",
            ],
            "libraries": ["<@(hipdnn_libraries)"],
            "cflags_cc": ["-std=c++17"],
            "conditions": [
                [
                    "OS=='win'",
                    {
                        "msvs_settings": {
                            "VCCLCompilerTool": {
                                "AdditionalOptions": ["/std:c++17", "/EHsc"]
                            }
                        },
                    },
                ],
            ],
        }
    ],
}
