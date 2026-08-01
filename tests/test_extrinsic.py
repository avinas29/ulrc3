"""Extrinsic harness tests — all offline, no API key, no network.

The scorer and the ledger are the parts that can silently corrupt a result, so
they are the parts that get tested. Several of these encode bugs that actually
occurred during the first real run against Gemini: a reply truncated mid-number
scoring as a compressor failure, and gold answers that asked for a computed
total while scoring against the inputs.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from ulrc3.metrics.extrinsic import Ledger, build_prompt, score_answer


def test_score_requires_all_gold_spans():
    assert score_answer("NDL-8650 and 65.75 percent", ["NDL-8650", "65.75"]) == 1.0
    assert score_answer("NDL-8650 and 65", ["NDL-8650", "65.75"]) == 0.5
    assert score_answer("nothing here", ["NDL-8650"]) == 0.0


def test_score_is_formatting_insensitive():
    """'$1,413.40' and '1413.40' are the same answer."""
    assert score_answer("$1,413.40", ["1413.40"]) == 1.0
    assert score_answer("  Business Plan  ", ["business plan"]) == 1.0


def test_not_in_context_scores_zero():
    assert score_answer("NOT IN CONTEXT", ["anything"]) == 0.0
    assert score_answer("", ["anything"]) == 0.0


def test_empty_gold_is_vacuously_correct():
    assert score_answer("whatever", []) == 1.0


def test_prompt_contains_context_and_question():
    p = build_prompt("CTX BODY", "What is X?")
    assert "CTX BODY" in p and "What is X?" in p
    assert "NOT IN CONTEXT" in p, "the abstain instruction must be present"


def test_ledger_budget_is_enforced_and_persisted():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ledger.json")
        led = Ledger.load(path, daily_budget=2)
        assert led.remaining == 2
        led.put("m", "p1", "r1")
        led.put("m", "p2", "r2")
        assert led.remaining == 0

        # a fresh load sees the spend: a crashed run cannot re-spend the quota
        again = Ledger.load(path, daily_budget=2)
        assert again.remaining == 0
        assert again.get("m", "p1") == "r1"


def test_ledger_cache_hit_costs_nothing():
    with tempfile.TemporaryDirectory() as d:
        led = Ledger.load(os.path.join(d, "l.json"), daily_budget=5)
        led.put("m", "prompt", "reply")
        spent = led.spent
        assert led.get("m", "prompt") == "reply"
        assert led.spent == spent, "a cache read must not consume budget"


def test_ledger_resets_on_a_new_day():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "l.json")
        led = Ledger.load(path, daily_budget=3)
        led.put("m", "p", "r")
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        data["day"] = "2000-01-01"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        assert Ledger.load(path, daily_budget=3).remaining == 3


def test_judge_refuses_without_a_key(monkeypatch):
    from ulrc3.metrics.extrinsic import GeminiJudge

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with tempfile.TemporaryDirectory() as d, pytest.raises(RuntimeError, match="no API key"):
        GeminiJudge(ledger=Ledger.load(os.path.join(d, "l.json")))


def test_output_budget_has_headroom_for_thinking_plus_answer():
    """Regression: a tight output budget truncated the visible answer.

    Reasoning models draw thinking tokens from the same budget, so we measured
    "65" where the answer was "65.75" and scored our own compressor down for it.
    (`thinkingConfig: {thinkingBudget: 0}` is the obvious fix but several models
    reject it with HTTP 400, so the budget carries the headroom instead.)
    """
    from ulrc3.metrics.extrinsic import GeminiJudge

    assert GeminiJudge.__init__.__defaults__ is not None
    import inspect

    sig = inspect.signature(GeminiJudge.__init__)
    assert sig.parameters["max_output_tokens"].default >= 2048


def test_cache_key_separates_generation_params():
    """A reply produced under different params must not be replayed as current.

    Without this, a broken generation setting survives a "fixed and re-run"
    cycle looking fixed, because every cell is a cache hit.
    """
    with tempfile.TemporaryDirectory() as d:
        led = Ledger.load(os.path.join(d, "l.json"), daily_budget=5)
        led.put("m", "prompt", "old", params="max=256")
        assert led.get("m", "prompt", params="max=256") == "old"
        assert led.get("m", "prompt", params="max=2048") is None


def test_failed_trial_is_missing_not_zero():
    """An HTTP error is an absent observation, not a wrong answer."""
    import math

    from ulrc3.metrics.extrinsic import evaluate_condition

    class Boom:
        name = "boom"

        def ask(self, prompt: str) -> str:
            raise RuntimeError("HTTP 400")

    t = evaluate_condition(Boom(), "i", "s", "ctx", "q?", ["gold"], "ulrc3", 10)
    assert t.error and math.isnan(t.score)


# --------------------------------------------------------------------------
# packing: making a 20-request/day quota usable
# --------------------------------------------------------------------------
def test_packed_prompt_contains_every_item():
    from ulrc3.metrics.extrinsic import build_packed_prompt

    p = build_packed_prompt([("ctx A", "Q one?"), ("ctx B", "Q two?"), ("ctx C", "Q three?")])
    for frag in ("ctx A", "ctx B", "ctx C", "Q one?", "Q two?", "Q three?"):
        assert frag in p
    assert p.count("--- ITEM ") == 3


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("1. alpha\n2. beta", ["alpha", "beta"]),
        ("1) alpha\n2 : beta", ["alpha", "beta"]),          # formatting drift
        ("1. alpha\ncontinued\n2. beta", ["alpha continued", "beta"]),  # wrapped
        ("1. alpha", ["alpha", ""]),                         # model answered one
        ("garbage", ["", ""]),                               # unparseable
    ],
)
def test_parse_packed_reply(reply, expected):
    from ulrc3.metrics.extrinsic import parse_packed_reply

    assert parse_packed_reply(reply, len(expected)) == expected


def test_parse_packed_reply_never_shifts_answers():
    """A skipped item must leave a hole, not slide later answers up.

    Shifting would silently score item 3's answer against item 2's gold, which
    is the kind of error that produces a confident wrong conclusion.
    """
    from ulrc3.metrics.extrinsic import parse_packed_reply

    assert parse_packed_reply("1. a\n3. c", 3) == ["a", "", "c"]


def test_packed_transport_failure_marks_all_items_missing():
    """One bad request must not manufacture k false negatives."""
    import math

    from ulrc3.metrics.extrinsic import evaluate_packed

    class Boom:
        name = "boom"

        def ask(self, prompt: str) -> str:
            raise RuntimeError("HTTP 500")

    items = [(f"i{k}", "s", "ctx", "q?", ["gold"], 10) for k in range(4)]
    out = evaluate_packed(Boom(), "ulrc3", items)
    assert len(out) == 4
    assert all(t.error and math.isnan(t.score) for t in out)


def test_packed_scores_each_item_against_its_own_gold():
    from ulrc3.metrics.extrinsic import evaluate_packed

    class Fixed:
        name = "fixed"

        def ask(self, prompt: str) -> str:
            return "1. apple\n2. banana"

    items = [
        ("i1", "s", "c", "q?", ["apple"], 5),
        ("i2", "s", "c", "q?", ["cherry"], 5),
    ]
    out = evaluate_packed(Fixed(), "ulrc3", items)
    assert out[0].score == 1.0
    assert out[1].score == 0.0
