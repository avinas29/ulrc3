"""Pins the heuristic tokenizer's accuracy against real BPE.

The heuristic exists so the engine works with zero dependencies. Its error
directly sets how wrong a token budget can be, so it is measured, not assumed —
and the bound is asserted here so a future edit to the constants cannot silently
regress it.
"""

from __future__ import annotations

import statistics

import pytest
from fixtures import ALL_FIXTURES

from ulrc3.tokenization import HeuristicTokenizer, TiktokenTokenizer


@pytest.fixture(scope="module")
def exact():
    try:
        return TiktokenTokenizer("cl100k_base")
    except Exception:  # pragma: no cover - tiktoken not installed
        pytest.skip("tiktoken unavailable")


@pytest.mark.parametrize("name", sorted(ALL_FIXTURES))
def test_per_document_error_is_bounded(exact, name):
    text = ALL_FIXTURES[name]
    e = exact.count(text)
    a = HeuristicTokenizer().count(text)
    err = abs(a - e) / max(1, e)
    assert err < 0.15, f"{name}: heuristic off by {err * 100:.1f}% ({a} vs {e})"


def test_mean_error_is_bounded(exact):
    heur = HeuristicTokenizer()
    errs = []
    for text in ALL_FIXTURES.values():
        e = exact.count(text)
        errs.append(abs(heur.count(text) - e) / max(1, e))
    mean = statistics.fmean(errs)
    assert mean < 0.10, f"mean heuristic error {mean * 100:.1f}% exceeds the documented 6.7%"


def test_heuristic_never_returns_zero_for_nonempty_text(exact):
    heur = HeuristicTokenizer()
    for text in ("a", "!", "3", "  x  ", "日本語"):
        assert heur.count(text) >= 1
