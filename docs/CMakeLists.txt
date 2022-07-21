################################################################################
# 
# MIT License
# 
# Copyright (c) 2022 Advanced Micro Devices, Inc.
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

#set build doc option
option(BUILD_DOCS "Build documentation" ON)

#function to wipe out directories on "make clean"
function(clean_doc_output DIR)
    set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_CLEAN_FILES ${DIR})
endfunction()

#wipe out html directory on "make clean"
set(DOCS_BUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/build)
clean_doc_output(${DOCS_BUILD_DIR}/html)


if(BUILD_DOCS)
    #check doxygen is installed 
    find_program(DOXYGEN_EXECUTABLE NAMES doxygen
        PATH_SUFFIXES bin
        DOC "Doxygen documentation generator"
    )
    mark_as_advanced(DOXYGEN_EXECUTABLE)

    #check for graphviz
    find_program(GRAPHVIZ_EXECUTABLE NAMES graphviz
        PATH_SUFFIXES bin
        DOC "Graphviz install"
    )
    mark_as_advanced(GRAPHVIZ_EXECUTABLE)

    #build doxygen html
    if (DOXYGEN_EXECUTABLE)
        message("Doxygen build started")
        file(MAKE_DIRECTORY ${DOCS_BUILD_DIR})
        add_custom_target(docs
            COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
            WORKING_DIRECTORY ${DOCS_BUILD_DIR}
            COMMENT "Generating Doxygen HTML"
            VERBATIM )
    else (DOXYGEN_EXECUTABLE)
        message("ERROR: Package Doxygen is required.")
    endif (DOXYGEN_EXECUTABLE)
else (BUILD_DOCS)
    message("BUILD_DOCS=OFF skipping documentation.")
endif(BUILD_DOCS)
