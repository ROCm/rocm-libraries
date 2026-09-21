# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(stinkytofu_objs PRIVATE
    StinkyAsmPrinter.cpp
    StinkyAsmEmitter.cpp
    IRLexer.cpp
    IRParser.cpp
    IRConverter.cpp
    ModifierSerializer.cpp
    PatternParser.cpp
    RawAsmParser.cpp)
