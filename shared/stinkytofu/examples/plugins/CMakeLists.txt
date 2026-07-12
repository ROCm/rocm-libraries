# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# LLVM-style plugin build: MODULE (shared) for dynamic loading,
# OBJECT for static linking into the host binary.
if(BUILD_SHARED_LIBS)
    add_library(stinkytofu-plugin-helloworld MODULE
        HelloWorldPass.cpp
    )

    if(WIN32)
        # Windows DLLs must resolve every import at link time, so the plugin links
        # the host import library (the canonical LLVM Windows plugin approach). The
        # ELF DT_NEEDED concern below does not apply to PE/COFF.
        target_link_libraries(stinkytofu-plugin-helloworld PRIVATE stinkytofu)
    else()
        # Headers only — deliberately do NOT link libstinkytofu. A plugin is discovered
        # and loaded *by* the host (libstinkytofu scans the plugin directory and dlopens
        # it); host symbols resolve at load time from the host's RTLD_GLOBAL-promoted copy.
        # Linking the host here would bake a DT_NEEDED libstinkytofu.so.0 into the plugin,
        # making the dynamic linker hunt for libstinkytofu on disk whenever the plugin is
        # loaded standalone — the host->plugin relationship inverted into plugin->host.
        target_include_directories(stinkytofu-plugin-helloworld PRIVATE
            $<TARGET_PROPERTY:stinkytofu,INTERFACE_INCLUDE_DIRECTORIES>
        )
    endif()

    set_target_properties(stinkytofu-plugin-helloworld PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins"
    )

    # stinkytofu locates its own example plugin relative to the loaded libstinkytofu
    # (see PassBuilder::examplePluginPath), using STINKYTOFU_PLUGIN_INSTALL_RELDIR — the
    # single source of truth (defined at the stinkytofu top level) shared by that runtime
    # lookup, this install rule, and consumers that vendor libstinkytofu.
    target_compile_definitions(stinkytofu PRIVATE
        STINKYTOFU_HAVE_EXAMPLE_PLUGIN
        STINKYTOFU_PLUGIN_INSTALL_RELDIR="${STINKYTOFU_PLUGIN_INSTALL_RELDIR}"
        STINKYTOFU_EXAMPLE_PLUGIN_FILENAME="$<TARGET_FILE_NAME:stinkytofu-plugin-helloworld>"
    )

    # Normal (non-EXCLUDE_FROM_ALL) builds install the plugin next to libstinkytofu's
    # lib dir. When rocisa add_subdirectory's stinkytofu with EXCLUDE_FROM_ALL this
    # rule does not fire; rocisa installs the plugin into its wheel itself.
    install(TARGETS stinkytofu-plugin-helloworld
        DESTINATION "lib/${STINKYTOFU_PLUGIN_INSTALL_RELDIR}")
else()
    add_library(stinkytofu-plugin-helloworld OBJECT
        HelloWorldPass.cpp
    )

    target_link_libraries(stinkytofu-plugin-helloworld PRIVATE stinkytofu)
endif()

target_include_directories(stinkytofu-plugin-helloworld PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
