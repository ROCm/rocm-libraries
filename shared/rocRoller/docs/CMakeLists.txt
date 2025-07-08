################################################################################
#
# MIT License
#
# Copyright 2023-2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

if(NOT TARGET docs)
    add_custom_target(docs)
endif()

function(mark_as_doc)
    add_dependencies(docs ${ARGN})
endfunction()

#function to wipe out directories on "make clean"
function(clean_doc_output DIR)
    set_property(
        DIRECTORY
        APPEND
        PROPERTY ADDITIONAL_CLEAN_FILES ${DIR}
    )
endfunction()

#check doxygen is installed
find_program(
    DOXYGEN_EXECUTABLE
    NAMES doxygen
    PATH_SUFFIXES bin
    DOC "Doxygen documentation generator"
)
mark_as_advanced(DOXYGEN_EXECUTABLE)

#check for graphviz
find_program(
    DOT_EXECUTABLE
    NAMES dot
    PATH_SUFFIXES bin
    DOC "Graphviz install"
)
mark_as_advanced(DOT_EXECUTABLE)

# check for sphinx
find_program(
    SPHINX_EXECUTABLE
    NAMES sphinx-build
    HINTS $ENV{SPHINX_DIR}
    PATH_SUFFIXES bin
    DOC "Sphinx documentation generator"
)
mark_as_advanced(SPHINX_EXECUTABLE)

# All Pre-build ------

set(DOCS_BUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/build)
set(DOXYGEN_BUILD_DIR ${DOCS_BUILD_DIR}/sphinx/html/doxygen)
set(DOXYGEN_CONFIG_FILE ${DOCS_BUILD_DIR}/src/Doxyfile)

#wipe out html directory on "make clean"
clean_doc_output(${DOCS_BUILD_DIR})
clean_doc_output(${DOXYGEN_BUILD_DIR})

set(SPHINX_BINARY_BUILD_DIR "${DOCS_BUILD_DIR}/sphinx/_build")

# Sphinx cache with pickled ReST documents
set(SPHINX_CACHE_DIR "${DOCS_BUILD_DIR}/sphinx/_doctrees")

# HTML output directory
set(SPHINX_DEFAULT_HTML_DIR "${DOCS_BUILD_DIR}/sphinx/html")

# copy docs/src folder into build
add_custom_target(
    copysrc_to_builddir
    COMMAND mkdir -p ${DOXYGEN_BUILD_DIR}
    COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/src ${DOCS_BUILD_DIR}
    COMMENT "Copy over docs/src dir to build directory."
)
mark_as_doc(copysrc_to_builddir)
# -----------------------

# Doxygen pre-build------

set(DOXYGEN_OUTPUT ${DOXYGEN_BUILD_DIR})
add_custom_target(
    prebuild_doxygen
    COMMAND sed -i 's;@DOXY_HAVE_DOT@;YES;g' ${DOXYGEN_CONFIG_FILE}
    COMMAND sed -i 's;@DOXY_OUTPUT_DIRECTORY@;${DOXYGEN_OUTPUT};g'
            ${DOXYGEN_CONFIG_FILE}
    COMMAND
        sed -i
        's;@DOXY_INPUT@;${CMAKE_CURRENT_SOURCE_DIR}/../lib\ ${DOCS_BUILD_DIR}/src/DesignOverview.md;g'
        ${DOXYGEN_CONFIG_FILE}
    COMMAND
        sed -i
        's;@DOXY_USE_MDFILE_AS_MAINPAGE@;${DOCS_BUILD_DIR}/src/DesignOverview.md;g'
        ${DOXYGEN_CONFIG_FILE}
    DEPENDS copysrc_to_builddir
)
mark_as_doc(prebuild_doxygen)
# -----------------------

# Doxygen build -----------
# Build entire Doxygen site
add_custom_target(
    build_doxygen
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_CONFIG_FILE}
    COMMENT "Building Doxygen html."
    DEPENDS prebuild_doxygen
)
mark_as_doc(build_doxygen)

#--------------------------

set(SPHINX_HTML_DIR
    "${SPHINX_DEFAULT_HTML_DIR}"
    CACHE PATH "Path to sphinx html output"
)

# Sphinx pre-build steps --------------------

# Prebuild Sphinx
add_custom_target(
    prebuild_sphinx
    DEPENDS copysrc_to_builddir
            clip_designreadme
            copyover_mainreadme
            copyover_dockerreadme
            copyover_adddocs
            build_doxygen
)

# Aux docs
# Copy docker README.md as docker.md
add_custom_target(
    copyover_dockerreadme
    COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/../docker/README.md
            ${DOCS_BUILD_DIR}/src/docker.md
    COMMENT
        "Copying over docker README.md to docs/build/src folder as docker.md."
    DEPENDS copysrc_to_builddir
)
clean_doc_output(${DOCS_BUILD_DIR}/src/docker.md)

# Copy main README.md as main.md and repath docker.md
add_custom_target(
    copyover_mainreadme
    COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/../README.md
            ${DOCS_BUILD_DIR}/src/main.md
    COMMAND sed -i 3,36d ${DOCS_BUILD_DIR}/src/main.md
    COMMAND sed -i 's;docker/README.md;docker.md;g'
            ${DOCS_BUILD_DIR}/src/main.md
    COMMENT
        "Copying over main README.md to docs/build/src folder as main.md and clipping top 30 lines."
    DEPENDS copysrc_to_builddir
)
clean_doc_output(${DOCS_BUILD_DIR}/src/main.md)

add_custom_target(
    copyover_adddocs
    COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/README.md
            ${DOCS_BUILD_DIR}/src/addDocs.md
    COMMENT "Copying over README.md to docs/build/src folder as addDocs.md."
    DEPENDS copysrc_to_builddir
)
clean_doc_output(${DOCS_BUILD_DIR}/src/addDocs.md)

# cut the top 2-36 lines as they are a ToC that does not
# build correctly in the website
add_custom_target(
    clip_designreadme
    COMMAND sed -i 2,36d ${DOCS_BUILD_DIR}/src/DesignOverview.md
    COMMENT "Clipping first 36 lines of DesignOverview.md."
    DEPENDS copysrc_to_builddir
)
mark_as_doc(prebuild_sphinx)

# ----------------------------

# Sphinx build ---------------

#build the entire sphinx html documentation
add_custom_target(build_sphinx
    COMMAND ${SPHINX_EXECUTABLE} ${DOCS_BUILD_DIR}/src ${SPHINX_HTML_DIR}
                -b html
                -c ${DOCS_BUILD_DIR}/src
                -d "${SPHINX_CACHE_DIR}"
                -D breathe_projects.proj=${DOXYGEN_OUTPUT}/xml
                -D breathe_default_project=proj
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS prebuild_doxygen prebuild_sphinx
    COMMENT "Building html documentation with Sphinx"
)
clean_doc_output(${SPHINX_HTML_DIR})
clean_doc_output(${SPHINX_CACHE_DIR})
clean_doc_output(${SPHINX_BINARY_BUILD_DIR})
mark_as_doc(build_sphinx)

# --------------------------------

# Sphinx post-build --------------

# adds the link from the sphinx site to the doxygen site
add_custom_target(
    changedoxylink
    COMMAND
        sed -i
        's;DoxygenLink;<a\ href=doxygen\/html\/index.html>Doxygen\ link\ <\/a>;g'
        ${SPHINX_HTML_DIR}/Development.html
    DEPENDS build_sphinx
)
mark_as_doc(changedoxylink)

#---------------------------------
