"""Load TileGym kernel sources and test files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


_SKIP_FILES = {"__init__", "utils"}

_KERNEL_DIR = "src/tilegym/ops/cutile"
_TEST_DIR = "tests/ops"


class TileGymLoader:
    """Loads cuTile kernel implementations and their test files from TileGym."""

    def __init__(self, tilegym_dir: str):
        self._root = Path(tilegym_dir)
        if not self._root.is_dir():
            raise FileNotFoundError(f"TileGym directory not found: {tilegym_dir}")
        self._kernel_dir = self._root / _KERNEL_DIR
        self._test_dir = self._root / _TEST_DIR

    def list_kernels(self) -> list[str]:
        """Return sorted list of kernel names (without .py extension)."""
        names = []
        for p in sorted(self._kernel_dir.glob("*.py")):
            name = p.stem
            if name not in _SKIP_FILES:
                names.append(name)
        return names

    def get_kernel_source(self, name: str) -> str:
        """Read the full source of a kernel implementation."""
        path = self._kernel_dir / f"{name}.py"
        return path.read_text(encoding="utf-8")

    def get_test_source(self, name: str) -> Optional[str]:
        """Read the test file for a kernel, or None if not found."""
        path = self._test_dir / f"test_{name}.py"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None

    def get_kernel_pair(self, name: str) -> dict:
        """Return kernel source + test source as a dict."""
        return {
            "kernel_name": name,
            "kernel_source": self.get_kernel_source(name),
            "test_source": self.get_test_source(name) or "",
        }

    def get_all_pairs(self) -> list[dict]:
        """Return all kernel pairs with test files."""
        return [self.get_kernel_pair(name) for name in self.list_kernels()]
