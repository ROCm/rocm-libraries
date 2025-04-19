
cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)
if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
  project(hipfort-tools)
endif()

# ensure mymcpu utility and Makefile.hipfort are available in build directory 
add_custom_command( OUTPUT sharedir
   COMMAND /bin/mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/../share/hipfort
   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/Makefile.hipfort)

add_custom_command( OUTPUT libexecdir
   COMMAND /bin/mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/../libexec/hipfort
   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gputable.txt)

add_custom_command( OUTPUT bindir
   COMMAND /bin/mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/../bin
   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/hipfc)

add_custom_command( OUTPUT gputable.txt
   COMMAND /bin/cp -p ${CMAKE_CURRENT_SOURCE_DIR}/gputable.txt ${CMAKE_CURRENT_BINARY_DIR}/../libexec/hipfort/gputable.txt
   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gputable.txt libexecdir)
# Use relative path with respect to hipfc install directory
message(STATUS "${CMAKE_Fortran_COMPILER}")
file(RELATIVE_PATH HIPFORT_COMPILER_RELATIVE_PATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}" ${CMAKE_Fortran_COMPILER})
string(REGEX REPLACE "[/]" "\\\\/" escaped_path ${HIPFORT_COMPILER_RELATIVE_PATH})
set(sed_str s/gfortran/${escaped_path}/g)
file(RELATIVE_PATH HIPFORT_INSTALL_RELATIVE_PATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}" ${CMAKE_INSTALL_PREFIX})
string(REGEX REPLACE "[/]" "\\\\/" installed_path ${HIPFORT_INSTALL_RELATIVE_PATH})
set(sed_str2 s/_HIPFORT_INSTALL_DIR_/${installed_path}/g)
# Use relative path with respect to Makefile.hipfort install directory
file(RELATIVE_PATH HIPFORT_MAKEFILE_REL_PATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DATADIR}/hipfort" ${CMAKE_Fortran_COMPILER})
string(REGEX REPLACE "[/]" "\\\\/" mkfile_escaped_path ${HIPFORT_MAKEFILE_REL_PATH})
set(sed_str3 s/gfortran/${mkfile_escaped_path}/g)

add_custom_command( OUTPUT Makefile.hipfort
   COMMAND /bin/cp -p ${CMAKE_CURRENT_SOURCE_DIR}/Makefile.hipfort ${CMAKE_CURRENT_BINARY_DIR}/../share/hipfort/Makefile.hipfort.needs_edit
   COMMAND /bin/sed -e '${sed_str3}' ${CMAKE_CURRENT_BINARY_DIR}/../share/hipfort/Makefile.hipfort.needs_edit > ../share/hipfort/Makefile.hipfort
   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/Makefile.hipfort sharedir)

add_custom_command( OUTPUT mymcpu
   COMMAND /bin/cp -p ${CMAKE_CURRENT_SOURCE_DIR}/mymcpu ${CMAKE_CURRENT_BINARY_DIR}/../libexec/hipfort/mymcpu.needs_edit
   COMMAND /bin/sed -e "s/X\\.Y\\-Z/${HIPFORT_VERSION}/g" ${CMAKE_CURRENT_BINARY_DIR}/../libexec/hipfort/mymcpu.needs_edit > ../libexec/hipfort/mymcpu
   COMMAND /bin/chmod  755  ${CMAKE_CURRENT_BINARY_DIR}/../libexec/hipfort/mymcpu
   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/mymcpu libexecdir)

# The myarchgpu symbolic link is needed for test targets before installation
add_custom_command( OUTPUT myarchgpu
   COMMAND /bin/ln -sf ${CMAKE_CURRENT_BINARY_DIR}/../libexec/hipfort/mymcpu ${CMAKE_CURRENT_BINARY_DIR}/../libexec/hipfort/myarchgpu
   DEPENDS mymcpu libexecdir)

add_custom_command( OUTPUT hipfc
   COMMAND /bin/cp -p ${CMAKE_CURRENT_SOURCE_DIR}/hipfc ${CMAKE_CURRENT_BINARY_DIR}/../bin/hipfc.needs_edit
   COMMAND /bin/sed -i -e '${sed_str}' ${CMAKE_CURRENT_BINARY_DIR}/../bin/hipfc.needs_edit
   COMMAND /bin/sed -i -e "s/X\\.Y\\-Z/${HIPFORT_VERSION}/g" ${CMAKE_CURRENT_BINARY_DIR}/../bin/hipfc.needs_edit
   COMMAND /bin/sed -e '${sed_str2}' ${CMAKE_CURRENT_BINARY_DIR}/../bin/hipfc.needs_edit > hipfc
   COMMAND /bin/chmod 755 hipfc
   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/hipfc bindir)

add_custom_target(util_scripts ALL DEPENDS bindir libexecdir sharedir gputable.txt Makefile.hipfort hipfc mymcpu myarchgpu)

# Install hipfc binary, marking under Component devel
rocm_install(PROGRAMS
   # Version string edited
   ${CMAKE_CURRENT_BINARY_DIR}/../bin/hipfc
   DESTINATION ${CMAKE_INSTALL_BINDIR}
   COMPONENT devel
)

# Install libexec files, marking under Component devel
rocm_install(PROGRAMS
   # The first two are symlinks
   ${CMAKE_CURRENT_SOURCE_DIR}/mygpu
   ${CMAKE_CURRENT_SOURCE_DIR}/myarchgpu
   # Version string edited
   ${CMAKE_CURRENT_BINARY_DIR}/../libexec/hipfort/mymcpu
   DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/hipfort
   COMPONENT devel
)

# GPUTable.txt installed to libexec
# Install gputable.txt files, marking under Component devel
rocm_install(FILES
   ${CMAKE_CURRENT_SOURCE_DIR}/gputable.txt
   DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/hipfort
   COMPONENT devel)

# Makefile.hipfort installed to share
# Install Makefile.hipfort files, marking under Component devel
rocm_install(FILES
   ${CMAKE_CURRENT_BINARY_DIR}/../share/hipfort/Makefile.hipfort
   DESTINATION ${CMAKE_INSTALL_DATADIR}/hipfort
   COMPONENT devel)
