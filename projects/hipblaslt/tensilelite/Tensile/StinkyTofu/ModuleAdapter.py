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
StinkyTofu Adapter Module

This module provides a transitional adapter layer that allows gradual migration
from rocisa.Module to StinkyTofu IRListModule.

Migration Strategy:
    Phase 1: Use Module class with validation enabled (dual-path)
    Phase 2: Enable StinkyTofu output while continuing validation
    Phase 3: Disable validation once stable
    Phase 4: Remove rocisa.Module inheritance

Usage:
    from Tensile.StinkyTofuAdapter import Module, StructuredModule

    # Enable validation globally
    Module.USE_STINKYTOFU = False  # Use rocisa output, but validate

    # Create module (drop-in replacement for rocisa.Module)
    kernel = Module("my_kernel")
    kernel.add(instruction)
    asm = kernel.toString()

    # Or use structured module with header/middle/footer sections
    structured = StructuredModule("structured_kernel")
    structured.header.add(header_instruction)
    structured.middle.add(main_instruction)
    structured.footer.add(footer_instruction)

    # Debug both representations
    kernel.dump()
"""

import os
import sys
import re
from typing import Optional, List, Any

try:
    import rocisa.code as rocisa_code  # rocisa bindings
    rocisa_binding = True
except ImportError:
    print("Warning: rocisa.Code module not found. Running in StinkyTofu-only mode.")
    rocisa_binding = False

try:
    import stinkytofu as st
except ImportError:
    print("Warning: StinkyTofu module not found. Adapter will use rocisa-only mode.")
    st = None


class Module:
    """
    StinkyTofu-backed Module that maintains rocisa.Module compatibility.

    This class uses composition instead of inheritance, keeping rocisa.Module
    as an internal member. It can operate in multiple modes:

    - Validation mode: Both rocisa and StinkyTofu generate output, compare for correctness
    - StinkyTofu mode: Uses StinkyTofu for output generation
    - Legacy mode: Pure rocisa (when StinkyTofu is disabled)

    Attributes:
        USE_STINKYTOFU: Class-level flag to enable StinkyTofu output globally
        DEBUG_CONVERSION: Class-level flag to print conversion debug info
    """

    # Class-level configuration flags
    USE_STINKYTOFU = os.getenv("USE_STINKYTOFU", "1") == "1"
    DEBUG_CONVERSION = os.getenv("DEBUG_STINKYTOFU", "0") == "1"

    def __init__(self, name: str = ""):
        """
        Initialize Module with both rocisa and StinkyTofu backing.

        Args:
            name: Module/kernel name
        """
        self._name = name
        self._rocisa_module = None
        self._st = None
        self._stinky_module = None
        self._stinky_items = []  # Lazy list of items to be added to StinkyTofu

        # Initialize rocisa module as member (not parent)
        if rocisa_binding:
            self._rocisa_module = rocisa_code.Module(name)

        # StinkyTofu will be initialized lazily when needed

    def rocisa(self):
        return self._rocisa_module

    def _get_architecture(self) -> List[int]:
        """
        Get GPU architecture version as [major, minor, stepping].

        Returns:
            List of [major, minor, stepping], e.g., [12, 5, 0] for gfx1250
        """

        # TODO: Integrate with Tensile global context to get actual target architecture
        # Default to gfx1250
        return [12, 5, 0]

    def _ensure_stinky_module(self):
        """
        Lazily create StinkyTofu module and build it from accumulated items.

        This is called on-demand when the StinkyTofu module is actually needed
        (e.g., in toString() or dump()).
        """
        if self._stinky_module is not None:
            return  # Already created

        if not (st and self.USE_STINKYTOFU):
            return  # StinkyTofu not available or not enabled

        try:
            # Create StinkyTofu infrastructure
            arch = self._get_architecture()
            self._st = st.StinkyAsmIR(arch)
            self._stinky_module = self._st.createIRList(self._name)

            if self.DEBUG_CONVERSION:
                print(f"[StinkyTofu] Lazily initialized module '{self._name}' with arch {arch}")

            # Process all accumulated items
            for item in self._stinky_items:
                self._build_stinky_item(item)

        except Exception as e:
            if self.DEBUG_CONVERSION:
                print(f"[StinkyTofu] Failed to lazily initialize module '{self._name}': {e}")
            self._st = None
            self._stinky_module = None

    def add(self, item, pos: int = -1):
        """
        Add item to module.

        Routes to appropriate backend based on whether item is a wrapper.

        Args:
            item: Item to add (Instruction, Module, Macro, Label, etc.)
            pos: Position to insert (-1 for append)

        Returns:
            The added item (ModuleAdapter/wrapper for dual-backend items, rocisa item otherwise)
        """
        result = item

        # Add to rocisa module if available
        if self.rocisa() is not None:
            if self._is_wrapper(item):
                # Add unwrapped to rocisa, but return the wrapper
                self.rocisa().add(item._rocisa_inst, pos)
                result = item
            elif isinstance(item, Module):
                # item is ModuleAdapter (StinkyModule) - add rocisa backend but return ModuleAdapter
                self.rocisa().add(item.rocisa(), pos)
                result = item
            elif self._is_macro_adapter(item):
                # item is MacroAdapter - add rocisa backend but return MacroAdapter
                self.rocisa().add(item.rocisa(), pos)
                result = item
            else:
                result = self.rocisa().add(item, pos)

        # Also add to StinkyTofu if enabled (lazy - just append to list)
        if st and self.USE_STINKYTOFU:
            self._add_to_stinky(item)

        return result

    # def setItems(self, items: List):
    #     """
    #     Set items in module, replacing all existing items.

    #     This method unwraps any wrapped items before passing to rocisa.

    #     Args:
    #         items: List of items to set (can include wrappers, Modules, etc.)
    #     """
    #     if self.rocisa() is not None:
    #         # Unwrap all items for rocisa
    #         unwrapped_items = []
    #         for item in items:
    #             if item is None:
    #                 unwrapped_items.append(None)
    #             elif self._is_wrapper(item):
    #                 unwrapped_items.append(item._rocisa_inst)
    #             elif isinstance(item, Module):
    #                 unwrapped_items.append(item.rocisa())
    #             elif self._is_macro_adapter(item):
    #                 unwrapped_items.append(item.rocisa())
    #             else:
    #                 unwrapped_items.append(item)

    #         self.rocisa().setItems(unwrapped_items)

    #     # For StinkyTofu, clear and replace items list
    #     if st and self.USE_STINKYTOFU:
    #         self._stinky_items.clear()
    #         self._stinky_module = None  # Force rebuild on next use
    #         for item in items:
    #             if item is not None:
    #                 self._add_to_stinky(item)

    def _add_to_stinky(self, item):
        """
        Lazily add item to StinkyTofu by appending to items list.

        Items are not processed immediately; they're stored and will be
        processed when the StinkyTofu module is built (on-demand).

        Args:
            item: Item to add (wrapper, Module, Macro, or rocisa object)
        """
        # Simply append to the lazy items list
        # Actual processing happens in _ensure_stinky_module()
        self._stinky_items.append(item)

    def _build_stinky_item(self, item):
        """
        Build a single item into the StinkyTofu module.

        This is called by _ensure_stinky_module() when lazily building the module.

        Args:
            item: Item to build (wrapper, Module, Macro, or rocisa object)
        """
        if self._stinky_module is None:
            return  # Module not ready yet

        try:
            # Check if item is a wrapper with dual backend support
            if self._is_wrapper(item):
                if self.DEBUG_CONVERSION:
                    wrapper_class = item.__class__.__name__
                    print(f"[StinkyTofu] Building wrapper: {wrapper_class}")

                # Get native StinkyTofu instruction from wrapper
                stinky_inst = item.create(self._st)
                if stinky_inst is not None:
                    self._stinky_module.add(stinky_inst)
                else:
                    if self.DEBUG_CONVERSION:
                        print(f"[StinkyTofu] Wrapper {wrapper_class} returned None for StinkyTofu instruction")
                return

            # Handle ModuleAdapter - flatten its contents
            if isinstance(item, Module):
                # Flatten the nested module by recursively building its _stinky_items
                # Don't call _ensure_stinky_module() - just process its items directly
                for nested_item in item._stinky_items:
                    self._build_stinky_item(nested_item)
                return

            # Handle MacroAdapter
            if self._is_macro_adapter(item):
                # Add MacroAdapter's StinkyTofu body (IRList) to module
                if hasattr(item, '_stinky_body') and item._stinky_body is not None:
                    self._stinky_module.addModule(item._stinky_body)
                return

        except Exception as e:
            if self.DEBUG_CONVERSION:
                print(f"[StinkyTofu] Failed to build item: {e}")

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

    def _is_macro_adapter(self, item) -> bool:
        """
        Check if item is a MacroAdapter instance.

        We avoid importing MacroAdapter to prevent circular imports,
        so we check for the presence of characteristic methods.

        Args:
            item: Object to check

        Returns:
            True if item is likely a MacroAdapter, False otherwise
        """
        return (hasattr(item, 'rocisa') and
                hasattr(item, '_rocisa_macro') and
                callable(getattr(item, 'rocisa', None)))

    def toString(self) -> str:
        """
        Generate assembly string.

        Returns StinkyTofu output if enabled, otherwise rocisa output.

        Returns:
            Assembly code as string
        """
        # Get rocisa assembly
        rocisa_asm = ""
        if self.rocisa() is not None:
            rocisa_asm = str(self.rocisa())

        # If StinkyTofu not enabled, return rocisa output
        if not self.USE_STINKYTOFU:
            return rocisa_asm

        # Lazily build StinkyTofu module if needed
        self._ensure_stinky_module()

        # Get StinkyTofu assembly
        if self._stinky_module is None:
            return rocisa_asm

        try:
            stinky_asm = self._stinky_module.emitAssembly()
        except Exception as e:
            print(f"[StinkyTofu] Error generating assembly: {e}")
            return rocisa_asm

        return stinky_asm

    def addModuleAsFlatItems(self, module_item):
        """
        Add module as flat items (flattening the module structure).

        Handles both rocisa.Module and StinkyModule (ModuleAdapter).
        """
        if isinstance(module_item, Module):
            if self.rocisa() is not None:
                self.rocisa().addModuleAsFlatItems(module_item.rocisa())
            if st and self.USE_STINKYTOFU:
                def get_flatten_items(module_item):
                    flat_items = []
                    for item in module_item._stinky_items:
                        if isinstance(item, Module):
                            flat_items += get_flatten_items(item)
                        else:
                            flat_items.append(item)
                    return flat_items

                flat_items = get_flatten_items(module_item)
                self._stinky_items += flat_items

        return module_item

    def dump(self):
        """
        Dump both rocisa and StinkyTofu representations for debugging.

        Prints internal IR structure from both implementations.
        """
        print("\n" + "="*70)
        print(f"Module: {self._name}")
        print("="*70)

        if self.rocisa() is not None:
            print("\n--- RocISA Module ---")
            try:
                print(self.prettyPrint(""))
            except Exception as e:
                print(f"Error dumping RocISA module: {e}")

        # Lazily build StinkyTofu module if needed
        if st and self.USE_STINKYTOFU:
            self._ensure_stinky_module()

        if self._stinky_module is not None:
            print("\n--- StinkyTofu IRList ---")
            try:
                print(self._stinky_module)
            except Exception as e:
                print(f"Error dumping StinkyTofu module: {e}")

        # TODO: Print conversion statistics

    def appendModule(self, module_item):
        """
        Append module to the current module.
        """
        if isinstance(module_item, Module):
            if self.rocisa() is not None:
                self.rocisa().appendModule(module_item.rocisa())
            if st and self.USE_STINKYTOFU:
                self._stinky_items += module_item._stinky_items
        else:
            if self.rocisa() is not None:
                self.rocisa().appendModule(module_item)

        return self

    def addItems(self, items):
        """
        Add items to the module.
        """
        for item in items:
            self.add(item)

    def getItem(self, index: int):
        """
        Get item by index.
        """
        if self.rocisa() is not None:
            return self.rocisa().getItem(index)
        else:
            return self._stinky_items[index]

    def setItem(self, index: int, item):
        """
        Set item by index.
        """
        if self.rocisa() is not None:
            self.rocisa().setItem(index, item)
        else:
            self._stinky_items[index] = item

    def items(self):
        """
        Return items in the module.
        """
        if self.rocisa() is not None:
            return self.rocisa().items()
        else:
            return self._stinky_items

    def replaceItemByIndex(self, index: int, item):
        """
        Replace item by index.
        """
        if self.rocisa() is not None:
            self.rocisa().replaceItemByIndex(index, item)
            self._stinky_items[index] = item

    def removeItemByIndex(self, index: int):
        """
        Remove item by index.
        """
        if self.rocisa() is not None:
            self.rocisa().removeItemByIndex(index)
            self._stinky_items.pop(index)

    def popFirstItem(self):
        """
        Pop first item.
        """
        if self.rocisa() is not None:
            self._stinky_items.pop(0)
            return self.rocisa().popFirstItem()
        else:
            return self._stinky_items.pop(0)

    def popFirstNItems(self, n: int):
        """
        Pop first n items.
        """
        if self.rocisa() is not None:
            self._stinky_items = self._stinky_items[n:]
            return self.rocisa().popFirstNItems(n)
        else:
            items = self._stinky_items[:n]
            self._stinky_items = self._stinky_items[n:]
            return items

    def addComment(self, comment: str):

        """
        Add comment to the module.
        """
        if self.rocisa() is not None:
            self.rocisa().addComment(comment)
            self._stinky_items.append(self.rocisa().items()[-1])

    def addCommentAlign(self, comment: str):

        """
        Add comment to the module.
        """
        if self.rocisa() is not None:
            self.rocisa().addCommentAlign(comment)
            self._stinky_items.append(self.rocisa().items()[-1])

    def addComment0(self, comment: str):

        """
        Add comment to the module.
        """
        if self.rocisa() is not None:
            self.rocisa().addComment0(comment)
            self._stinky_items.append(self.rocisa().items()[-1])

    def addComment1(self, comment: str):

        """
        Add comment to the module.
        """
        if self.rocisa() is not None:
            self.rocisa().addComment1(comment)
            self._stinky_items.append(self.rocisa().items()[-1])

    def addComment2(self, comment: str):
        """
        Add comment to the module.
        """
        if self.rocisa() is not None:
            self.rocisa().addComment2(comment)
            self._stinky_items.append(self.rocisa().items()[-1])

    def addSpaceLine(self):
        """
        Add space line to the module.
        """
        if self.rocisa() is not None:
            self.rocisa().addSpaceLine()
            self._stinky_items.append(self.rocisa().items()[-1])

    def __str__(self):
        """
        Return string representation of the module.
        """
        if self.rocisa() is not None:
            return str(self.rocisa())

    def __getattr__(self, name):
        """
        Delegate unknown attributes to internal _rocisa_module.

        This catches any rocisa::Module methods not explicitly defined above.
        """
        if self.rocisa() is not None and hasattr(self.rocisa(), name):
            attr = getattr(self.rocisa(), name)
            if callable(attr):
                # For methods, create a wrapper that delegates
                def method_wrapper(*args, **kwargs):
                    return attr(*args, **kwargs)
                return method_wrapper
            else:
                # For properties, return directly
                return attr
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class StructuredModule(Module):
    """
    Structured Module with header, middle, and footer sections.

    This class provides a structured organization for modules, similar to
    rocisa::StructuredModule. It automatically creates three sub-modules:
    - header: For initialization code, register definitions, etc.
    - middle: For the main body of code
    - footer: For cleanup code, return statements, etc.

    Usage:
        module = StructuredModule("my_kernel")
        module.header.add(instruction1)
        module.middle.add(instruction2)
        module.footer.add(instruction3)

    The three sections are automatically added to the parent module in order.
    """

    def __init__(self, name: str = ""):
        """
        Initialize StructuredModule with header, middle, and footer sections.

        Args:
            name: Module name
        """
        super().__init__(name)

        # Create the three structured sections
        self.header = Module("header")
        self.middle = Module("middle")
        self.footer = Module("footer")

        # Add them to the parent module in order
        super().add(self.header)
        super().add(self.middle)
        super().add(self.footer)
