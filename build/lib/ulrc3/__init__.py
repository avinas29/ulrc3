"""ULRC3 -- Ultra Low-Resource LLM Context Compression Engine.

A semantic compiler for LLM context: parse the prompt into a typed, addressable
intermediate representation, optimise it under a token budget with hard
preservation constraints, and emit an auditable compressed program.

    >>> from ulrc3 import Compressor, Config, Mode
    >>> r = Compressor(Config(mode=Mode.BALANCED)).compress(long_text, query="...")
    >>> r.text, r.ratio, r.verification.obligation_recall
"""

from .config import Config, Mode
from .engine import Compressor, compress
from .request import Doc, Request
from .types import CompressionResult, Protection, Verification
from .version import __version__

__all__ = [
    "Compressor",
    "Config",
    "CompressionResult",
    "Doc",
    "Mode",
    "Protection",
    "Request",
    "Verification",
    "compress",
    "__version__",
]
