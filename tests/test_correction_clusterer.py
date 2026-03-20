# tests/test_correction_clusterer.py
from eval.cutile.correction_clusterer import CorrectionClusterer


def test_cluster_by_keywords():
    corrections = [
        {"correction": "No, use ct.matmul instead of manual loop", "kernel_name": "a"},
        {"correction": "No, use ct.sum instead of manual reduction loop", "kernel_name": "b"},
        {"correction": "No, wrong tile shape for tensor cores", "kernel_name": "c"},
        {"correction": "No, tile dimensions must be multiples of 16 for tensor cores", "kernel_name": "d"},
        {"correction": "No, missing TMA configuration in ct.load", "kernel_name": "e"},
    ]
    clusterer = CorrectionClusterer()
    clusters = clusterer.cluster(corrections)
    assert len(clusters) >= 2  # at least "builtin ops" and "tile shapes"
    assert all("corrections" in c for c in clusters)
    assert all("label" in c for c in clusters)


def test_empty_corrections():
    clusterer = CorrectionClusterer()
    clusters = clusterer.cluster([])
    assert clusters == []


def test_single_correction():
    clusterer = CorrectionClusterer()
    clusters = clusterer.cluster([{"correction": "No, wrong API", "kernel_name": "x"}])
    assert len(clusters) == 1
