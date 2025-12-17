"""
FileCheck-style testing utilities for StinkyTofu.

Inspired by LLVM's FileCheck tool, this module provides pattern matching
capabilities for verifying assembly output.
"""

import re
from typing import List, Optional, Tuple


class FileCheckError(Exception):
    """Exception raised when FileCheck patterns don't match."""
    pass


class FileCheck:
    """
    FileCheck-style pattern matcher for assembly verification.
    
    Usage:
        checker = FileCheck(assembly_output)
        checker.check("v_add_u32")
        checker.check_next("v_mul_f32")
        checker.check_label("kernel_entry")
        checker.check_not("v_invalid_inst")
    """
    
    def __init__(self, text: str):
        """
        Initialize FileCheck with text to verify.
        
        Args:
            text: The assembly output to check
        """
        self.lines = text.split('\n')
        self.current_line = 0
        self.errors = []
        
    def reset(self):
        """Reset the current line position."""
        self.current_line = 0
        
    def check(self, pattern: str, regex: bool = False) -> bool:
        """
        Check that pattern appears anywhere in remaining lines.
        
        Args:
            pattern: Pattern to search for
            regex: If True, treat pattern as regex
            
        Returns:
            True if found, False otherwise
            
        Raises:
            FileCheckError: If pattern not found
        """
        for i in range(self.current_line, len(self.lines)):
            line = self.lines[i]
            if regex:
                if re.search(pattern, line):
                    self.current_line = i + 1
                    return True
            else:
                if pattern in line:
                    self.current_line = i + 1
                    return True
                    
        raise FileCheckError(
            f"CHECK failed: pattern '{pattern}' not found after line {self.current_line}"
        )
        
    def check_next(self, pattern: str, regex: bool = False) -> bool:
        """
        Check that pattern appears on the very next non-empty line.
        
        Args:
            pattern: Pattern to search for
            regex: If True, treat pattern as regex
            
        Returns:
            True if found
            
        Raises:
            FileCheckError: If pattern not found on next line
        """
        # Skip empty lines and comments-only lines
        while self.current_line < len(self.lines):
            line = self.lines[self.current_line].strip()
            if line and not line.startswith('//'):
                break
            self.current_line += 1
            
        if self.current_line >= len(self.lines):
            raise FileCheckError(
                f"CHECK-NEXT failed: no more lines (looking for '{pattern}')"
            )
            
        line = self.lines[self.current_line]
        if regex:
            match = re.search(pattern, line)
        else:
            match = pattern in line
            
        if not match:
            raise FileCheckError(
                f"CHECK-NEXT failed: pattern '{pattern}' not found on line {self.current_line}: '{line}'"
            )
            
        self.current_line += 1
        return True
        
    def check_label(self, label: str) -> bool:
        """
        Check for a label pattern (typically used for function names).
        Similar to CHECK-LABEL in FileCheck.
        
        Args:
            label: Label pattern to search for
            
        Returns:
            True if found
            
        Raises:
            FileCheckError: If label not found
        """
        return self.check(label, regex=False)
        
    def check_not(self, pattern: str, regex: bool = False) -> bool:
        """
        Check that pattern does NOT appear in remaining lines.
        
        Args:
            pattern: Pattern that should not be present
            regex: If True, treat pattern as regex
            
        Returns:
            True if pattern not found
            
        Raises:
            FileCheckError: If pattern is found
        """
        for i in range(self.current_line, len(self.lines)):
            line = self.lines[i]
            if regex:
                if re.search(pattern, line):
                    raise FileCheckError(
                        f"CHECK-NOT failed: pattern '{pattern}' found on line {i}: '{line}'"
                    )
            else:
                if pattern in line:
                    raise FileCheckError(
                        f"CHECK-NOT failed: pattern '{pattern}' found on line {i}: '{line}'"
                    )
        return True
        
    def check_dag(self, patterns: List[str], regex: bool = False) -> bool:
        """
        Check that all patterns appear in any order (like CHECK-DAG).
        
        Args:
            patterns: List of patterns to find
            regex: If True, treat patterns as regex
            
        Returns:
            True if all patterns found
            
        Raises:
            FileCheckError: If any pattern not found
        """
        start_line = self.current_line
        found_patterns = set()
        
        for i in range(start_line, len(self.lines)):
            line = self.lines[i]
            for idx, pattern in enumerate(patterns):
                if idx in found_patterns:
                    continue
                if regex:
                    if re.search(pattern, line):
                        found_patterns.add(idx)
                        self.current_line = max(self.current_line, i + 1)
                else:
                    if pattern in line:
                        found_patterns.add(idx)
                        self.current_line = max(self.current_line, i + 1)
                        
        if len(found_patterns) != len(patterns):
            missing = [patterns[i] for i in range(len(patterns)) if i not in found_patterns]
            raise FileCheckError(
                f"CHECK-DAG failed: patterns not found: {missing}"
            )
            
        return True
        
    def check_same(self, pattern: str, regex: bool = False) -> bool:
        """
        Check that pattern appears on the same line as the last match.
        
        Args:
            pattern: Pattern to search for
            regex: If True, treat pattern as regex
            
        Returns:
            True if found
            
        Raises:
            FileCheckError: If pattern not found on current line
        """
        if self.current_line == 0 or self.current_line > len(self.lines):
            raise FileCheckError(
                "CHECK-SAME failed: no previous match"
            )
            
        # Go back one line since check() advances
        line = self.lines[self.current_line - 1]
        
        if regex:
            match = re.search(pattern, line)
        else:
            match = pattern in line
            
        if not match:
            raise FileCheckError(
                f"CHECK-SAME failed: pattern '{pattern}' not found on line {self.current_line - 1}: '{line}'"
            )
            
        return True
        
    def check_count(self, pattern: str, count: int, regex: bool = False) -> bool:
        """
        Check that pattern appears exactly 'count' times.
        
        Args:
            pattern: Pattern to count
            count: Expected number of occurrences
            regex: If True, treat pattern as regex
            
        Returns:
            True if count matches
            
        Raises:
            FileCheckError: If count doesn't match
        """
        matches = 0
        for line in self.lines[self.current_line:]:
            if regex:
                matches += len(re.findall(pattern, line))
            else:
                matches += line.count(pattern)
                
        if matches != count:
            raise FileCheckError(
                f"CHECK-COUNT failed: pattern '{pattern}' found {matches} times, expected {count}"
            )
            
        return True
        
    def check_regex(self, pattern: str) -> Optional[re.Match]:
        """
        Check for regex pattern and return the match object.
        
        Args:
            pattern: Regex pattern to search for
            
        Returns:
            Match object if found
            
        Raises:
            FileCheckError: If pattern not found
        """
        for i in range(self.current_line, len(self.lines)):
            line = self.lines[i]
            match = re.search(pattern, line)
            if match:
                self.current_line = i + 1
                return match
                
        raise FileCheckError(
            f"CHECK (regex) failed: pattern '{pattern}' not found after line {self.current_line}"
        )


def extract_registers(line: str) -> List[str]:
    """
    Extract register names from an assembly line.
    
    Args:
        line: Assembly line
        
    Returns:
        List of register names (e.g., ['v[0]', 'v[1]', 'a[0:3]'])
    """
    # Match vgpr, sgpr, acc patterns
    patterns = [
        r'a\[\d+:\d+\]',   # a[0:3], etc. (ranges first)
        r'v\[\d+:\d+\]',   # v[0:3], etc.
        r's\[\d+:\d+\]',   # s[0:3], etc.
        r'v\[\d+\]',       # v[0], v[1], etc. (single)
        r's\[\d+\]',       # s[0], s[1], etc.
        r'v\d+',           # v0, v1, etc. (without brackets)
        r's\d+',           # s0, s1, etc.
    ]
    
    registers = []
    for pattern in patterns:
        registers.extend(re.findall(pattern, line))
    return registers


def extract_instruction(line: str) -> Optional[Tuple[str, List[str]]]:
    """
    Extract instruction mnemonic and operands from assembly line.
    
    Args:
        line: Assembly line
        
    Returns:
        Tuple of (mnemonic, operands) or None if not an instruction
    """
    # Remove comments
    line = re.sub(r'//.*$', '', line).strip()
    if not line:
        return None
        
    # Match instruction format: mnemonic operands
    match = re.match(r'([a-z_][a-z0-9_]*)\s+(.*)', line, re.IGNORECASE)
    if not match:
        return None
        
    mnemonic = match.group(1)
    operands_str = match.group(2)
    
    # Split operands by comma
    operands = [op.strip() for op in operands_str.split(',')]
    
    return (mnemonic, operands)

