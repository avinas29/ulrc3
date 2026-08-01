"""Content-type detection and heterogeneous segmentation."""

from .doctype import LABELS, Region, detect, entropy, segment  # noqa: F401

__all__ = ["LABELS", "Region", "detect", "entropy", "segment"]
