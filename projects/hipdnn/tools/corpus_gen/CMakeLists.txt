# Corpus generation for UHD training (RFC 0019.13 §5).
#
# Header-only, and consumed by nothing yet: the generator itself is unwritten (RFC 0019.13
# gap B1). The probe lands ahead of it because its correctness is independent of the rest --
# it searches a region through an injected predicate, so it can be tested against regions
# whose shape is known rather than waiting on an engine to ask.
add_library(hipdnn_corpus_gen INTERFACE)

target_include_directories(hipdnn_corpus_gen
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

if(NOT HIPDNN_SKIP_TESTS)
    add_subdirectory(tests)
endif()
