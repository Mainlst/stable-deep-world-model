"""Compatibility shim for VTA model definitions.

The original monolithic implementation has been split into
`src_vta/models/` for better maintainability and research iteration speed.
Import `VTA` (and other building blocks) from there.
"""

from models import (  # noqa: F401
    Decoder,
    Encoder,
    HierarchicalRSSM,
    LatentDistribution,
    PostBoundaryDetector,
    PriorBoundaryDetector,
    VTA,
)

__all__ = [
    "Decoder",
    "Encoder",
    "HierarchicalRSSM",
    "LatentDistribution",
    "PostBoundaryDetector",
    "PriorBoundaryDetector",
    "VTA",
]
