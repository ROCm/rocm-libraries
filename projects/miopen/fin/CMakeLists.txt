cmake_minimum_required(VERSION 3.25)
project(dummyFIN)

if(MIOPEN_ENABLE_FIN)
  install(TARGETS fin)
endif()
