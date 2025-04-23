# Software Specifications

## Introduction
The vision of hipfort is to support acceleration technology in FORTRAN with an open and  portable set of standard FORTRAN module interfaces.  

### Purpose
*TBD*

### Intended Audience
*TBD*


### Intended Use
This document is intended to guide the development and continued maintenance of hipfort

**hipfort maintainers and developers** use this document to ensure that development activity is kept within scope, meets user needs, and delivers the system features and requirements documented in this software specifications document.

**hipfort users** use this document to hold maintainers and developers accountable on meeting their needs and delivering the system features and requirements specified within this software specifications document.

Maintainers, developers, and users should use this document to discuss changes to the scope, user needs, and system features and requirements.

### Scope
*TBD*

### Definitions and Acronyms

1. “Portable” includes across GPU architectures and across compilers. 
2. “Open” means open source with contributions through github with the MITx11 License.
3. “Standard FORTRAN module interfaces” means FORTRAN standard compliant interfaces to a HIP library and publicly available functional libraries.

## Description
*TBD*

### User Needs
*TBD*


### Assumptions and Dependencies
We assume that end-users of hipfort have applications of their own that are expected to run on AMD® (amdgcm) and Nvidia® (nvptx) GPU hardware. Additionally, users are in need of an API that allows for explicit GPU memory management and explicit control of GPU kernel execution.

The hipfort API is a Fortran layer that exposes, through `ISO C BINDING`, the HIP API and other ROCm accelerated libraries. Because of this, the scope of possible API calls is limited to those provided by HIP and ROCm accelerated libraries.

### Testing & Acceptance Criteria
Each major release of hipfort is required to be tested with the following Fortran compilers
1. GNU Fortran (> v9.0)
2. AOMP ( Clang/Flang ) ( > v??)

and across the following GPU platforms
1. AMD® ..
2. Nvidia® K80, P100, V100, T4, P4 (> sm35)
