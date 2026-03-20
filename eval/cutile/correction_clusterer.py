"""Group corrections by error pattern using keyword matching."""

from __future__ import annotations

import re
from collections import defaultdict

# Keyword clusters for cuTile error patterns
_CLUSTER_KEYWORDS = {
    "builtin-ops": ["manual loop", "manual reduction", "ct.matmul", "ct.sum", "ct.mma", "builtin", "built-in"],
    "tile-shape": ["tile shape", "tile size", "tile dimensions", "multiples of", "tensor core"],
    "tma-config": ["tma", "ct.load", "ct.store", "memory accelerator"],
    "memory-layout": ["transpose", "layout", "stride", "contiguous", "permute"],
    "api-misuse": ["doesn't exist", "not available", "wrong function", "wrong api", "import"],
    "dtype": ["dtype", "float32", "bfloat16", "float16", "precision", "cast"],
}


class CorrectionClusterer:
    """Groups corrections by error pattern using keyword matching."""

    def cluster(self, corrections: list[dict]) -> list[dict]:
        """Group corrections into clusters. Each cluster: {label, corrections: []}."""
        if not corrections:
            return []

        buckets: dict[str, list[dict]] = defaultdict(list)

        for corr in corrections:
            text = corr.get("correction", "").lower()
            matched = False
            for label, keywords in _CLUSTER_KEYWORDS.items():
                if any(kw in text for kw in keywords):
                    buckets[label].append(corr)
                    matched = True
                    break
            if not matched:
                buckets["other"].append(corr)

        return [
            {"label": label, "corrections": items}
            for label, items in buckets.items()
        ]
