if(HIPBLASLT_ENABLE_HOST)
    add_subdirectory(hipblaslt)
    add_subdirectory(rocblaslt)
endif()

add_subdirectory(tensilelite)
