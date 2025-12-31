################################################################################
#
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

"""
MacroAdapter Module

This module provides a dual-backend adapter for rocisa.code.Macro that supports
both rocisa and StinkyTofu backends, mirroring the ModuleAdapter architecture.

Architecture:
    - Dual backend: rocisa.code.Macro + StinkyTofu AsmMacroDefinition (when available)
    - Automatically routes wrapped instructions to both backends
    - Maintains API compatibility with rocisa.code.Macro

Key Features:
    - Drop-in replacement for rocisa.code.Macro
    - Dual backend support (rocisa + StinkyTofu)
    - Automatically unwraps and routes instruction wrappers
    - Gradual migration path for macro-heavy code

Usage:
    from Tensile.StinkyTofu.MacroAdapter import Macro
    from Tensile.StinkyTofu.ItemAdapters import VMovB32
    from rocisa.container import vgpr

    # Create macro (same API as rocisa.code.Macro)
    macro = Macro("MY_MACRO", args=["dst", "src"])

    # Add wrapped instructions (routed to both backends)
    macro.add(VMovB32(dst=vgpr("\\dst"), src=vgpr("\\src")))

    # Get backend-specific instances
    rocisa_macro = macro.rocisa()
    stinky_macro = macro.stinky()  # When StinkyTofu Macro bindings available
"""

import os
from typing import List, Optional, Any

try:
    import rocisa.code as rocisa_code
    rocisa_binding = True
except ImportError:
    print("Warning: rocisa.code module not found.")
    rocisa_binding = False

try:
    import stinkytofu as st
except ImportError:
    print("Warning: StinkyTofu module not found. Adapter will use rocisa-only mode.")
    st = None


class Macro:
    """
    Dual-backend Macro adapter supporting both rocisa and StinkyTofu.
    
    This class maintains both rocisa.code.Macro and StinkyTofu AsmMacroDefinition
    backends (when available), routing wrapped instructions to both. It mirrors
    the architecture of ModuleAdapter for consistency.
    
    Attributes:
        USE_STINKYTOFU: Class-level flag to enable StinkyTofu output globally
        DEBUG_CONVERSION: Class-level flag to print conversion debug info
    
    Usage:
        from Tensile.StinkyTofu.MacroAdapter import Macro
        from Tensile.StinkyTofu.ItemAdapters import VMovB32
        
        macro = Macro("my_macro", args=["dst", "src"])
        macro.add(VMovB32(dst=vgpr("\\dst"), src=vgpr("\\src")))
    """
    
    # Class-level configuration flags (mirroring ModuleAdapter)
    USE_STINKYTOFU = os.getenv("USE_STINKYTOFU", "1") == "1"
    DEBUG_CONVERSION = os.getenv("DEBUG_STINKYTOFU", "0") == "1"
    
    def __init__(self, name: str = "", args: List[str] = None):
        """
        Initialize Macro with both rocisa and StinkyTofu backing.
        
        Args:
            name: Macro name
            args: List of macro parameter names (optional, defaults to [])
        """
        self._name = name
        self._args = args if args is not None else []
        self._rocisa_macro = None
        self._st = None
        self._stinky_macro = None
        self._stinky_body = None  # IRList for macro body
        
        # Statistics for tracking conversion coverage (instance-level)
        self._conversion_stats = {
            'rocisa_items': 0,
            'stinkytofu_items': 0,
            'failed_conversions': 0,
        }
        
        # Initialize rocisa macro
        if rocisa_binding:
            self._rocisa_macro = rocisa_code.Macro(name, self._args)
        
        # Initialize StinkyTofu if available and enabled
        if st and self.USE_STINKYTOFU:
            try:
                # Get architecture from global context
                arch = self._get_architecture()
                self._st = st.StinkyAsmIR(arch)
                self._stinky_body = self._st.createIRList(f"{name}_body")
                
                # TODO: When AsmMacroDefinition is exposed in Python bindings:
                # self._stinky_macro = st.AsmMacroDefinition(name, self._args, self._stinky_body)
                
                if self.DEBUG_CONVERSION:
                    print(f"[MacroAdapter] Initialized macro '{name}' with arch {arch}")
                    print(f"[MacroAdapter] StinkyTofu Macro support pending Python bindings")
            except Exception as e:
                if self.DEBUG_CONVERSION:
                    print(f"Warning: Failed to initialize StinkyTofu for macro '{name}': {e}")
                self._st = None
                self._stinky_body = None
                self._stinky_macro = None
    
    def rocisa(self):
        """Get underlying rocisa.code.Macro instance."""
        return self._rocisa_macro
    
    def stinky(self):
        """
        Get underlying StinkyTofu AsmMacroDefinition instance.
        
        Returns:
            AsmMacroDefinition instance or None if StinkyTofu not available
        
        Note:
            Currently returns None until AsmMacroDefinition is exposed in Python bindings.
            This method is provided for API consistency with ModuleAdapter.
        """
        return self._stinky_macro
    
    def _get_architecture(self) -> List[int]:
        """
        Get GPU architecture version as [major, minor, stepping].
        
        Returns:
            List of [major, minor, stepping], e.g., [12, 5, 0] for gfx1250
        """
        # TODO: Integrate with Tensile global context to get actual target architecture
        # Default to gfx1250
        return [12, 5, 0]
    
    def add(self, item):
        """
        Add item to macro.
        
        Routes to both rocisa and StinkyTofu backends, automatically unwrapping
        instruction wrappers and routing them appropriately.
        
        Note: Unlike Module.add(), Macro.add() doesn't support position parameter.
        Items are always appended.
        
        Args:
            item: Item to add (Instruction wrapper, Module, Label, etc.)
            
        Returns:
            The added item (or its rocisa equivalent)
        """
        result = item
        
        # Add to rocisa macro if available
        if self._rocisa_macro is not None:
            self._conversion_stats['rocisa_items'] += 1
            if self._is_wrapper(item):
                # Wrapped instruction - extract rocisa instruction
                result = self._rocisa_macro.add(item._rocisa_inst)
            elif self._is_module_adapter(item):
                # ModuleAdapter - extract rocisa module
                result = self._rocisa_macro.add(item.rocisa())
            elif isinstance(item, Macro):
                # MacroAdapter - extract rocisa macro
                result = self._rocisa_macro.add(item.rocisa())
            else:
                # Regular rocisa item - add directly
                result = self._rocisa_macro.add(item)
        
        # Also add to StinkyTofu if enabled
        if self._stinky_body is not None:
            try:
                self._add_to_stinky(item)
            except Exception as e:
                if self.DEBUG_CONVERSION:
                    print(f"[MacroAdapter] Failed to add item to StinkyTofu: {e}")
                self._conversion_stats['failed_conversions'] += 1
        
        return result
    
    def _add_to_stinky(self, item):
        """
        Add item to StinkyTofu macro body (IRList).
        
        Checks if item is a wrapper (has dual backend) or original rocisa.
        Wrappers handle their own backend routing.
        
        Args:
            item: Item to add (wrapper or rocisa object)
        """
        # Check if item is a wrapper with dual backend support
        if self._is_wrapper(item):
            if self.DEBUG_CONVERSION:
                wrapper_class = item.__class__.__name__
                print(f"[MacroAdapter] Added wrapper: {wrapper_class} to macro body")
            
            # Get native StinkyTofu instruction from wrapper
            self._stinky_body.add(item.create(self._st))
            self._conversion_stats['stinkytofu_items'] += 1
            return
        
        # Handle ModuleAdapter
        if self._is_module_adapter(item):
            # Add Module's StinkyTofu content to macro body
            if hasattr(item, '_stinky_module') and item._stinky_module is not None:
                self._stinky_body.addModule(item._stinky_module)
                self._conversion_stats['stinkytofu_items'] += item._conversion_stats.get('stinkytofu_items', 0)
            return
        
        # Handle nested MacroAdapter
        if isinstance(item, Macro):
            # Add nested macro's body to this macro's body
            if item._stinky_body is not None:
                self._stinky_body.addModule(item._stinky_body)
                self._conversion_stats['stinkytofu_items'] += item._conversion_stats.get('stinkytofu_items', 0)
            return
        
        # Not a wrapper - skip (rocisa-only item)
        if self.DEBUG_CONVERSION:
            print(f"[MacroAdapter] Skipping rocisa-only item: {type(item).__name__}")
    
    def _is_wrapper(self, item) -> bool:
        """
        Check if item is a dual-backend wrapper.
        
        Wrappers have _backend and _rocisa_inst attributes set by the
        @instruction_wrapper decorator.
        
        Args:
            item: Object to check
            
        Returns:
            True if item is a wrapper, False otherwise
        """
        return (hasattr(item, '_backend') and
                hasattr(item, '_rocisa_inst') and
                hasattr(item, '__class__'))
    
    def _is_module_adapter(self, item) -> bool:
        """
        Check if item is a ModuleAdapter instance.
        
        We avoid importing ModuleAdapter to prevent circular imports,
        so we check for the presence of characteristic methods.
        
        Args:
            item: Object to check
            
        Returns:
            True if item is likely a ModuleAdapter, False otherwise
        """
        return (hasattr(item, 'rocisa') and
                hasattr(item, '_rocisa_module') and
                callable(getattr(item, 'rocisa', None)))
    
    def items(self):
        """Return items in the macro (from rocisa backend)."""
        if self._rocisa_macro is not None:
            return self._rocisa_macro.items()
        return []
    
    def prettyPrint(self, indent: str = "") -> str:
        """Pretty print macro contents (from rocisa backend)."""
        if self._rocisa_macro is not None:
            return self._rocisa_macro.prettyPrint(indent)
        return ""
    
    def toString(self) -> str:
        """
        Generate macro definition string.
        
        Returns StinkyTofu output if enabled and available, otherwise rocisa output.
        
        Returns:
            Macro definition as string (.macro ... .endm)
        """
        # Get rocisa assembly
        rocisa_asm = ""
        if self._rocisa_macro is not None:
            try:
                rocisa_asm = self.prettyPrint()
            except Exception as e:
                if self.DEBUG_CONVERSION:
                    print(f"[MacroAdapter] Failed to get rocisa output: {e}")
        
        # Get StinkyTofu assembly (when available)
        stinky_asm = ""
        if self._stinky_macro is not None:
            try:
                # TODO: Implement when AsmMacroDefinition is exposed
                # stinky_asm = self._stinky_macro.toString()
                pass
            except Exception as e:
                if self.DEBUG_CONVERSION:
                    print(f"[MacroAdapter] Failed to get StinkyTofu output: {e}")
        
        # Return appropriate output based on USE_STINKYTOFU flag
        if self.USE_STINKYTOFU and stinky_asm:
            return stinky_asm
        return rocisa_asm
    
    def dump(self):
        """
        Dump macro contents for debugging.
        
        Prints both rocisa and StinkyTofu representations and conversion statistics.
        """
        print("\n" + "="*70)
        print(f"Macro: {self._name}")
        print(f"Arguments: {self._args}")
        print("="*70)
        
        # Print rocisa version
        if self._rocisa_macro is not None:
            print("\n--- RocISA Macro ---")
            try:
                print(self.prettyPrint(""))
            except Exception as e:
                print(f"(prettyPrint failed: {e})")
        
        # Print StinkyTofu version (when available)
        if self._stinky_body is not None:
            print("\n--- StinkyTofu Macro Body (IRList) ---")
            try:
                print(self._stinky_body.toString())
            except Exception as e:
                print(f"(toString failed: {e})")
        
        # Print conversion statistics
        print("\n--- Conversion Statistics ---")
        print(f"Rocisa items:       {self._conversion_stats['rocisa_items']}")
        print(f"StinkyTofu items:   {self._conversion_stats['stinkytofu_items']}")
        print(f"Failed conversions: {self._conversion_stats['failed_conversions']}")
        
        print("="*70 + "\n")
    
    def __getattr__(self, name):
        """
        Delegate unknown attributes to internal _rocisa_macro.
        
        This catches any rocisa::Macro methods not explicitly defined above.
        """
        if self._rocisa_macro is not None and hasattr(self._rocisa_macro, name):
            attr = getattr(self._rocisa_macro, name)
            if callable(attr):
                # For methods, create a wrapper that delegates
                def method_wrapper(*args, **kwargs):
                    return attr(*args, **kwargs)
                return method_wrapper
            else:
                # For properties, return directly
                return attr
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
