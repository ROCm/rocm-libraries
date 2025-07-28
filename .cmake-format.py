# ROCm CMake command definitions
additional_commands = {
    "rocm_install": {
        "pargs": {"nargs": "*", "flags": []},
        "kwargs": {
            "TARGETS": {"pargs": {"nargs": "*", "flags": []}},
            "EXPORT": {"pargs": {"nargs": "1", "flags": []}},
            "ARCHIVE": {"pargs": {"nargs": "*", "flags": []}},
            "LIBRARY": {"pargs": {"nargs": "*", "flags": []}},
            "RUNTIME": {"pargs": {"nargs": "*", "flags": []}},
            "PUBLIC_HEADER": {"pargs": {"nargs": "0", "flags": []}, "kwargs": {
                "COMPONENT": {"pargs": {"nargs": "1", "flags": []}},
                "DESTINATION": {"pargs": {"nargs": "1", "flags": []}}
            }},
            "FILES": {"pargs": {"nargs": "*", "flags": []}},
            "DIRECTORY": {"pargs": {"nargs": "*", "flags": []}},
            "PROGRAMS": {"pargs": {"nargs": "*", "flags": []}},
            "DESTINATION": {"pargs": {"nargs": "1", "flags": []}},
            "COMPONENT": {"pargs": {"nargs": "1", "flags": []}},
            "DEPENDS": {"pargs": {"nargs": "*", "flags": []}},
            "PACKAGE": {"pargs": {"nargs": "1", "flags": []}},
            "NAMESPACE": {"pargs": {"nargs": "1", "flags": []}},
            "FILES_MATCHING": {"pargs": {"nargs": "0", "flags": []}},
            "PATTERN": {"pargs": {"nargs": "1", "flags": []}},
            "PERMISSIONS": {"pargs": {"nargs": "*", "flags": []}}
        }
    },
    "rocm_export_targets": {
        "pargs": {"nargs": "*", "flags": []},
        "kwargs": {
            "TARGETS": {"pargs": {"nargs": "*", "flags": []}},
            "EXPORT": {"pargs": {"nargs": "1", "flags": []}},
            "NAMESPACE": {"pargs": {"nargs": "1", "flags": []}},
            "DEPENDS": {"pargs": {"nargs": "*", "flags": []}},
            "PACKAGE": {"pargs": {"nargs": "1", "flags": []}}
        }
    },
    "rocm_package_add_dependencies": {
        "pargs": {"nargs": "*", "flags": []},
        "kwargs": {
            "DEPENDS": {"pargs": {"nargs": "*", "flags": []}}
        }
    },
    "rocm_create_package": {
        "pargs": {"nargs": "*", "flags": []},
        "kwargs": {
            "NAME": {"pargs": {"nargs": "*", "flags": []}},
            "DESCRIPTION": {"pargs": {"nargs": "*", "flags": []}},
            "MAINTAINER": {"pargs": {"nargs": "*", "flags": []}},
            "LDCONFIG": {"pargs": {"nargs": "0", "flags": []}},
            "LDCONFIG_DIR": {"pargs": {"nargs": "*", "flags": []}}
        }
    },
    "rocm_setup_client_packages": {
        "pargs": {"nargs": "0", "flags": []}
    },
    "rocm_setup_client_components": {
        "pargs": {"nargs": "0", "flags": []}
    },
    "rocm_get_openmp_package": {
        "pargs": {"nargs": "2", "flags": []}
    },
    "rocm_get_gfortran_package": {
        "pargs": {"nargs": "2", "flags": []}
    }
}


format = {
    "line_width": 100,
    "tab_size": 2,
    "dangle_parens": True,
    "max_subgroups_hwrap": 3,
    "max_pargs_hwrap": 10,
    "max_rows_cmdline": 1,
    "max_lines_hwrap": 1,
    "command_case": "lower",
    "keyword_case": "unchanged",
    "enable_sort": False
} 