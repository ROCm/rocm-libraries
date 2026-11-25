# Installed next to the origami package config, for a static ORIGAMI_ENABLE_NN build only.
#
# A static liborigami does not carry its dependencies, so a consumer's final link needs
# yaml-cpp. The exported target lists the plain name origami_yaml_cpp, which is not a
# target at install time (naming yaml-cpp::yaml-cpp there fails install(EXPORT) when the
# fetched copy was used). This file defines it for the consumer.
#
# yaml-cpp must be installed: a fetched copy is not, because yaml-cpp 0.9.0 only
# generates install rules when it is the top-level project.
include(CMakeFindDependencyMacro)
find_dependency(yaml-cpp 0.8)

if(NOT TARGET origami_yaml_cpp)
    add_library(origami_yaml_cpp INTERFACE IMPORTED)
    target_link_libraries(origami_yaml_cpp INTERFACE yaml-cpp::yaml-cpp)
endif()
