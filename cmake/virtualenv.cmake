if("$ENV{Python_ROOT}" STREQUAL "" AND NOT Python_ROOT AND "$ENV{Python_ROOT_DIR}" STREQUAL "" AND NOT Python_ROOT_DIR)
    message(STATUS "Python_ROOT is unset. Trying to set Python_ROOT by pyenv.")
    if("$ENV{PYENV_ROOT}" STREQUAL "")
        find_program(PYENV_EXE pyenv)
        if(PYENV_EXE)
            execute_process(
              COMMAND ${PYENV_EXE} version-file 
              OUTPUT_VARIABLE PYENV_ROOT
            )
            if(PYENV_ROOT)
              message(STATUS "Setting Python_ROOT to \"pyenv version-file\".")
              set(Python_ROOT ${PYENV_ROOT})
            endif()
        endif()
        if(NOT Python_ROOT)
            message(STATUS "cannot find python root. Setting Python_ROOT to /usr.")
            message(STATUS "Configure Python_ROOT variable if a different installation is preferred.")
            set(Python_ROOT /usr)
        endif()
    else()
        message(STATUS "PYENV_ROOT is set. set Python_ROOT to $ENV{PYENV_ROOT}.")
        set(Python_ROOT $ENV{PYENV_ROOT})
    endif()
endif()

find_package(Python REQUIRED COMPONENTS Interpreter)

set(VIRTUALENV_PYTHON_EXE ${Python_EXECUTABLE})

get_filename_component(VIRTUALENV_PYTHON_EXENAME ${VIRTUALENV_PYTHON_EXE} NAME CACHE)

set(VIRTUALENV_HOME_DIR ${CMAKE_BINARY_DIR}/virtualenv CACHE PATH "Path to virtual environment")

function(virtualenv_create)
    execute_process(
      COMMAND ${VIRTUALENV_PYTHON_EXE} -m venv ${VIRTUALENV_HOME_DIR} --system-site-packages --clear
      COMMAND_ECHO STDOUT
    )

    if(WIN32)
        set(VIRTUALENV_BIN_DIR ${VIRTUALENV_HOME_DIR}/Scripts CACHE PATH "Path to virtualenv bin directory")
    else()
        set(VIRTUALENV_BIN_DIR ${VIRTUALENV_HOME_DIR}/bin CACHE PATH "Path to virtualenv bin directory")
    endif()
endfunction()

function(virtualenv_install)
    virtualenv_create()
    execute_process(
      COMMAND ${VIRTUALENV_BIN_DIR}/${VIRTUALENV_PYTHON_EXENAME} -m pip install --upgrade pip
      COMMAND_ECHO STDOUT
    )
    execute_process(
      COMMAND ${VIRTUALENV_BIN_DIR}/${VIRTUALENV_PYTHON_EXENAME} -m pip install ${ARGN}
      COMMAND_ECHO STDOUT
      RESULT_VARIABLE return_code
      ERROR_VARIABLE error_message
      OUTPUT_VARIABLE output_message
    )

    if(return_code)
        message("Error Code: ${rc}")
        message("StdOut: ${output_message}")
        message(FATAL_ERROR "StdErr: ${error_message}" )
    endif()
endfunction()