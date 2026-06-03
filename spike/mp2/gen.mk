# Wrapper that drives the MicroPython embed package generation + freezing.
MICROPYTHON_TOP = ../micropython
FROZEN_MANIFEST = $(CURDIR)/manifest.py
# Add extmod modules to the qstr + MP_REGISTER_MODULE scan. Must be set BEFORE the
# include (make expands the qstr rule's prerequisites at read time). TOP-relative,
# vpath-resolved like the py core sources. Compiled separately in build.sh.
SRC_QSTR += extmod/modre.c $(CURDIR)/modcomgr.c
include $(MICROPYTHON_TOP)/ports/embed/embed.mk
