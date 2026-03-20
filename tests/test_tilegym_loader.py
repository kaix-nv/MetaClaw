# tests/test_tilegym_loader.py
import os
import pytest
from eval.cutile.tilegym_loader import TileGymLoader


TILEGYM_DIR = os.path.join(os.path.dirname(__file__), "..", "TileGym")
SKIP_NO_TILEGYM = pytest.mark.skipif(
    not os.path.isdir(TILEGYM_DIR), reason="TileGym not found"
)


@SKIP_NO_TILEGYM
def test_loader_finds_kernels():
    loader = TileGymLoader(TILEGYM_DIR)
    kernels = loader.list_kernels()
    assert len(kernels) >= 20
    assert "softmax" in kernels
    assert "matmul" in kernels


@SKIP_NO_TILEGYM
def test_loader_reads_kernel_source():
    loader = TileGymLoader(TILEGYM_DIR)
    source = loader.get_kernel_source("softmax")
    assert "ct.kernel" in source or "@ct.kernel" in source
    assert len(source) > 100


@SKIP_NO_TILEGYM
def test_loader_reads_test_source():
    loader = TileGymLoader(TILEGYM_DIR)
    test_src = loader.get_test_source("softmax")
    assert "def test_op" in test_src
    assert len(test_src) > 100


@SKIP_NO_TILEGYM
def test_loader_get_kernel_pair():
    loader = TileGymLoader(TILEGYM_DIR)
    pair = loader.get_kernel_pair("softmax")
    assert "kernel_source" in pair
    assert "test_source" in pair
    assert "kernel_name" in pair
    assert pair["kernel_name"] == "softmax"


@SKIP_NO_TILEGYM
def test_loader_skips_init_and_utils():
    loader = TileGymLoader(TILEGYM_DIR)
    kernels = loader.list_kernels()
    assert "__init__" not in kernels
    assert "utils" not in kernels


def test_loader_bad_dir():
    with pytest.raises(FileNotFoundError):
        TileGymLoader("/nonexistent/path")
