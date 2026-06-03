# Wrapper that drives the MicroPython embed package generation + freezing.
MICROPYTHON_TOP = ../micropython
FROZEN_MANIFEST = $(CURDIR)/manifest.py
include $(MICROPYTHON_TOP)/ports/embed/embed.mk
