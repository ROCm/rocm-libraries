# Drives MicroPython embed-package generation + module freezing for the provider.
# Run via build_embed.sh, which supplies the paths on the make command line:
#   MICROPYTHON_TOP  - the MicroPython source checkout
#   FROZEN_MANIFEST  - manifest.py (freezes $FROZEN_DIR)
#   CKDSL_C_DIR      - provider src/micropython/ (modcomgr.c for the qstr scan)
#   BUILD            - out-of-source build dir (frozen_content.c, genhdr)
#   PACKAGE_DIR      - out-of-source embed package dir (micropython_embed)
# Defaults keep a bare in-tree invocation working for debugging.
MICROPYTHON_TOP ?= ../micropython
FROZEN_MANIFEST ?= $(CURDIR)/manifest.py
CKDSL_C_DIR ?= $(CURDIR)
# Add extmod + native modules to the qstr + MP_REGISTER_MODULE scan. Must be set
# BEFORE the include (make expands the qstr rule's prerequisites at read time).
# extmod/modre.c is TOP-relative (vpath-resolved like the py core sources);
# modcomgr.c is given by absolute path. Both are compiled separately in build_embed.sh.
SRC_QSTR += extmod/modre.c $(CKDSL_C_DIR)/modcomgr.c
include $(MICROPYTHON_TOP)/ports/embed/embed.mk
