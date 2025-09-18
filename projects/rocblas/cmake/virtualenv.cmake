# find_package(PythonInterp)
# # TODO: Check PYTHON_VERSION_MAJOR

find_program(VIRTUALENV_PYTHON_EXE ${python})
if(NOT VIRTUALENV_PYTHON_EXE)
    # look for non default name
    if(${python} MATCHES "python3")
        find_program(VIRTUALENV_PYTHON_EXE python)
    else()
        find_program(VIRTUALENV_PYTHON_EXE python3)
    endif()
endif()

set(VIRTUALENV_HOME_DIR ${CMAKE_BINARY_DIR}/virtualenv CACHE PATH "Path to virtual environment")

function(virtualenv_create)
    message("${VIRTUALENV_PYTHON_EXE} -m venv ${VIRTUALENV_HOME_DIR} --system-site-packages --clear")
    execute_process(
      RESULT_VARIABLE rc
      COMMAND ${VIRTUALENV_PYTHON_EXE} -m venv ${VIRTUALENV_HOME_DIR} --system-site-packages --clear
    )
    if(rc)
        message(FATAL_ERROR ${rc})
    endif()

    if(WIN32)
        set(VIRTUALENV_BIN_DIR ${VIRTUALENV_HOME_DIR}/Scripts CACHE PATH "Path to virtualenv bin directory")
    else()
        set(VIRTUALENV_BIN_DIR ${VIRTUALENV_HOME_DIR}/bin CACHE PATH "Path to virtualenv bin directory")
    endif()

    # verify python executable name inside virtualenv as may be python3 or python (even if installed by python3)
    find_program(VIRTUALENV_INST_PYTHON_EXE python3 PATHS ${VIRTUALENV_BIN_DIR} NO_DEFAULT_PATH)
    if(NOT VIRTUALENV_INST_PYTHON_EXE)
        find_program(VIRTUALENV_INST_PYTHON_EXE python PATHS ${VIRTUALENV_BIN_DIR} NO_DEFAULT_PATH)
    endif()

    get_filename_component(VIRTUALENV_PYTHON_EXENAME ${VIRTUALENV_INST_PYTHON_EXE} NAME CACHE)

    # report the virtual env python version
    message("virtualenv python version: ${VIRTUALENV_BIN_DIR}/${VIRTUALENV_PYTHON_EXENAME}")
    execute_process(
        COMMAND ${VIRTUALENV_BIN_DIR}/${VIRTUALENV_PYTHON_EXENAME} --version
        )

endfunction()

function(virtualenv_install)
    virtualenv_create()

    if(TENSILE_VENV_UPGRADE_PIP)
        message("${VIRTUALENV_BIN_DIR}/${VIRTUALENV_PYTHON_EXENAME} -m pip install --upgrade pip")
        execute_process(
          RESULT_VARIABLE rc
          COMMAND ${VIRTUALENV_BIN_DIR}/${VIRTUALENV_PYTHON_EXENAME} -m pip install --upgrade pip
        )
        if(rc)
            message(FATAL_ERROR ${rc})
        endif()

        message("${VIRTUALENV_BIN_DIR}/${VIRTUALENV_PYTHON_EXENAME} -m pip install --upgrade packaging setuptools")
        execute_process(
            RESULT_VARIABLE rc
            COMMAND ${VIRTUALENV_BIN_DIR}/${VIRTUALENV_PYTHON_EXENAME} -m pip install --upgrade packaging setuptools
        )
        if(rc)
            message(FATAL_ERROR ${rc})
        endif()
    endif()

    message("${VIRTUALENV_BIN_DIR}/${VIRTUALENV_PYTHON_EXENAME} -m pip install ${ARGN}")
    execute_process(
      RESULT_VARIABLE rc
      COMMAND ${VIRTUALENV_BIN_DIR}/${VIRTUALENV_PYTHON_EXENAME} -m pip install ${ARGN}
    )
    if(rc)
        message(FATAL_ERROR ${rc})
    endif()
endfunction()
