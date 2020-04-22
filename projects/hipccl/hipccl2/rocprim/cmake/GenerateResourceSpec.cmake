#!/usr/bin/cmake -P

find_program(ROCMINFO_EXECUTABLE
  rocminfo
)

if(NOT ROCMINFO_EXECUTABLE)
  message(FATAL_ERROR "rocminfo not found")
endif()

execute_process(
  COMMAND ${ROCMINFO_EXECUTABLE}
  RESULT_VARIABLE ROCMINFO_EXIT_CODE
  OUTPUT_VARIABLE ROCMINFO_STDOUT
  ERROR_VARIABLE ROCMINFO_STDERR
)

if(ROCMINFO_EXIT_CODE)
  message(SEND_ERROR "rocminfo exited with ${ROCMINFO_EXIT_CODE}")
  message(FATAL_ERROR ${ROCMINFO_STDERR})
endif()

string(REGEX MATCHALL [[--(gfx[0-9]+)]]
  ROCMINFO_MATCHES
  ${ROCMINFO_STDOUT}
)

# Transform raw regex matches to pairs of gfx IP and device id
set(GFXIP_AND_ID)
set(ID 0)
foreach(ROCMINFO_MATCH IN LISTS ROCMINFO_MATCHES)
  string(REGEX REPLACE
    "--"
    ""
    ROCMINFO_MATCH
    ${ROCMINFO_MATCH}
  )
  list(APPEND GFXIP_AND_ID "${ROCMINFO_MATCH}:${ID}")
  math(EXPR ID "${ID} + 1")
endforeach()

#
# NOTE: Unfortunately we don't have std::partition in CMake, only list(SORT)
#

# Group together cards of matching gfx IP for JSON output
list(SORT GFXIP_AND_ID)

# Now comes the tricky part: implementing the following C++ logic
#
# stringstream JSON_PAYLOAD;
# auto actual_gfx_ip = GFXIP_AND_ID.at(0).ip;
# auto it = GFXIP_AND_ID.begin();
# while(it != GFXIP_AND_ID.end())
# {
#   auto IT = find(it, GFXIP_AND_ID.end(),
#                  [=](const tuple<string, int>& ip_id){ return ip_id.ip == actual_gfx_ip; });
#   JSON_PAYLOAD << "      \"" << actual_gfx_ip << "\": [\n";
#   foreach(it, IT, [&](const tuple<string, int>& ip_id)
#   {
#     JSON_PAYLOAD <<
#       "        {\n" <<
#       "           \"id\": \"" << ip_id.id << "\"\n" <<
#       "        },\n"
#   });
#   JSON_PAYLOAD.get(); // discard trailing comma
#   JSON_PAYLOAD << "      ],\n";
#   it = IT;
# }
# JSON_PAYLOAD.get(); // discard trailing comma
#

set()
set(ID 0)
list(GET GFXIP_AND_ID ${ID} )

set(JSON_HEAD [[
{
  "version": {
    "major": 1,
    "minor": 0
  },
  "local": [
    {
]])
set(JSON_TAIL [[
    }
  ]
}
]])