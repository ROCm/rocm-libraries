#############################################################################
# Given a list of flags, this function will try each, one at a time,
# and choose the first flag that works.  If no flags work, then nothing
# will be set, unless the REQUIRED key is given, in which case an error
# will be given.
# 
# Call is:
# SET_COMPILE_FLAG(FLAGVAR FLAGVAL (Fortran|C|CXX) <REQUIRED> flag1 flag2...)
# 
# For example, if you have the flag CMAKE_C_FLAGS and you want to add
# warnings and want to fail if this is not possible, you might call this
# function in this manner:
# SET_COMPILE_FLAGS(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" C REQUIRED
#                   "-Wall"     # GNU
#                   "-warn all" # Intel
#                  )
# The optin "-Wall" will be checked first, and if it works, will be
# appended to the CMAKE_C_FLAGS variable.  If it doesn't work, then
# "-warn all" will be tried.  If this doesn't work then checking will
# terminate because REQUIRED was given.  
#
# The reasong that the variable must be given twice (first as the name then
# as the value in quotes) is because of the way CMAKE handles the passing
# of variables in functions; it is difficult to extract a variable's
# contents and assign new values to it from within a function.
#############################################################################

INCLUDE(${CMAKE_ROOT}/Modules/CheckCCompilerFlag.cmake)
INCLUDE(${CMAKE_ROOT}/Modules/CheckCXXCompilerFlag.cmake)

FUNCTION(SET_COMPILE_FLAG FLAGVAR FLAGVAL LANG)

    # Do some up front setup if Fortran
    IF(LANG STREQUAL "Fortran")
        # Create a list of error messages from compilers
        SET(FAIL_REGEX
            "ignoring unknown option"             # Intel
            "invalid argument"                    # Intel
            "unrecognized .*option"               # GNU
            "[Uu]nknown switch"                   # Portland Group
            "ignoring unknown option"             # MSVC
            "warning D9002"                       # MSVC, any lang
            "[Uu]nknown option"                   # HP
            "[Ww]arning: [Oo]ption"               # SunPro
            "command option .* is not recognized" # XL
           )
    ENDIF(LANG STREQUAL "Fortran")

    # Make a variable holding the flags.  Filter out REQUIRED if it is there
    SET(FLAG_REQUIRED FALSE)
    SET(FLAG_FOUND FALSE)
    UNSET(FLAGLIST)
    FOREACH (var ${ARGN})
        STRING(TOUPPER "${var}" UP)
        IF(UP STREQUAL "REQUIRED")
            SET(FLAG_REQUIRED TRUE)
        ELSE()
            SET(FLAGLIST ${FLAGLIST} "${var}")
        ENDIF(UP STREQUAL "REQUIRED")
    ENDFOREACH (var ${ARGN})

    # Now, loop over each flag
    FOREACH(flag ${FLAGLIST})

        UNSET(FLAG_WORKS)
        # Check the flag for the given language
        IF(LANG STREQUAL "C")
            CHECK_C_COMPILER_FLAG("${flag}" FLAG_WORKS)
        ELSEIF(LANG STREQUAL "CXX")
            CHECK_CXX_COMPILER_FLAG("${flag}" FLAG_WORKS)
        ELSEIF(LANG STREQUAL "Fortran")
            # There is no nice function to do this for FORTRAN, so we must manually
            # create a test program and check if it compiles with a given flag.
            SET(TESTFILE "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}")
            SET(TESTFILE "${TESTFILE}/CMakeTmp/testFortranFlags.f90")
            FILE(WRITE "${TESTFILE}"
"
program dummyprog
  i = 5
end program dummyprog
")
            TRY_COMPILE(FLAG_WORKS ${CMAKE_BINARY_DIR} ${TESTFILE}
                COMPILE_DEFINITIONS "${flag}" OUTPUT_VARIABLE OUTPUT)
            
            # Check that the output message doesn't match any errors
            FOREACH(rx ${FAIL_REGEX})
                IF("${OUTPUT}" MATCHES "${rx}")
                    SET(FLAG_WORKS FALSE)
                ENDIF("${OUTPUT}" MATCHES "${rx}")
            ENDFOREACH(rx ${FAIL_REGEX})

        ELSE()
            MESSAGE(FATAL_ERROR "Unknown language in SET_COMPILE_FLAGS: ${LANG}")
        ENDIF(LANG STREQUAL "C")

        # If this worked, use these flags, otherwise use other flags
        IF(FLAG_WORKS)
            # Append this flag to the end of the list that already exists
            SET(${FLAGVAR} "${FLAGVAL} ${flag}" CACHE STRING
                 "Set the ${FLAGVAR} flags" FORCE)
            SET(FLAG_FOUND TRUE)
            BREAK() # We found something that works, so exit
        ENDIF(FLAG_WORKS)

    ENDFOREACH(flag ${FLAGLIST})

    # Raise an error if no flag was found
    IF(FLAG_REQUIRED AND NOT FLAG_FOUND)
        MESSAGE(FATAL_ERROR "No compile flags were found")
    ENDIF(FLAG_REQUIRED AND NOT FLAG_FOUND)

ENDFUNCTION()
