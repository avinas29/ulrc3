"""Extrinsic evaluation: does a *real* model still answer correctly?

Every other metric in this package is an upper bound. Answerability says the gold
span is still present in the compressed context; it cannot say the model found
it. This module closes that gap by putting a model in the loop.

Design constraints that shaped it:

* **Free-tier quotas are tiny** (~20 requests/day). So the runner keeps a
  persistent ledger on disk, refuses to exceed a declared budget, and caches
  every response keyed by (model, prompt hash) so re-analysis never re-spends.
* **No key in the repo.** The key is read from the environment only
  (``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``), passed as a header, never as a URL
  parameter, and never written to disk — only the *response text* is cached.
* **Provider-agnostic.** ``Judge`` is a protocol; the Gemini implementation is
  one adapter. Swapping in OpenAI/Anthropic is a subclass.

Scoring is deliberately strict and mechanical -- normalised containment of the
gold answer in the model's reply -- rather than an LLM-as-judge, because an
LLM judge would spend the same scarce quota and add its own error bars to a
measurement whose whole purpose is to remove error bars.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..ir.obligations import canon

DEFAULT_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-flash-latest"


class QuotaExhausted(RuntimeError):
    """Raised when the declared request budget (or the provider's) is spent."""


@dataclass
class Ledger:
    """Persistent request budget + response cache.

    The ledger is what makes a 20-request/day quota usable for research: a run
    that crashes half way, or an analysis you want to re-run, costs nothing
    extra because every completed call is on disk.
    """

    path: str
    daily_budget: int = 20
    spent: int = 0
    day: str = ""
    cache: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str, daily_budget: int = 20) -> Ledger:
        today = time.strftime("%Y-%m-%d")
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    d = json.load(fh)
                led = cls(
                    path=path,
                    daily_budget=daily_budget,
                    spent=int(d.get("spent", 0)),
                    day=d.get("day", today),
                    cache=dict(d.get("cache", {})),
                )
                if led.day != today:  # new day, quota resets
                    led.spent = 0
                    led.day = today
                return led
            except Exception:
                pass
        return cls(path=path, daily_budget=daily_budget, day=today)

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(
                {"day": self.day, "spent": self.spent, "cache": self.cache}, fh, indent=1
            )
        os.replace(tmp, self.path)

    @property
    def remaining(self) -> int:
        return max(0, self.daily_budget - self.spent)

    def key(self, model: str, prompt: str, params: str = "") -> str:
        """Cache identity = model + prompt + generation params.

        Omitting the params silently replays a reply produced under a *different*
        configuration, which is how a broken generation setting can survive a
        "fixed and re-run" cycle looking fixed.
        """
        return hashlib.blake2b(
            f"{model}\x00{params}\x00{prompt}".encode(), digest_size=16
        ).hexdigest()

    def get(self, model: str, prompt: str, params: str = "") -> str | None:
        return self.cache.get(self.key(model, prompt, params))

    def put(self, model: str, prompt: str, reply: str, params: str = "") -> None:
        self.cache[self.key(model, prompt, params)] = reply
        self.spent += 1
        self.save()


class Judge(Protocol):
    name: str

    def ask(self, prompt: str) -> str: ...


class GeminiJudge:
    """Minimal Gemini adapter over stdlib urllib (no SDK dependency)."""

    def __init__(
        self,
        ledger: Ledger,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        max_output_tokens: int = 2048,
        timeout: float = 180.0,  # packed prompts can carry 70k+ tokens
        endpoint: str = DEFAULT_ENDPOINT,
    ) -> None:
        self.model = model
        self.name = f"gemini:{model}"
        self.ledger = ledger
        self.timeout = timeout
        self.endpoint = endpoint
        self.max_output_tokens = max_output_tokens
        #: wall-clock of the last *fresh* request; None when served from cache.
        self.last_latency_ms: float | None = None
        self._key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self._key:
            raise RuntimeError(
                "no API key: set GEMINI_API_KEY in the environment (never in a file)"
            )

    @property
    def _params(self) -> str:
        return f"max={self.max_output_tokens};t=0.0"

    def ask(self, prompt: str) -> str:
        cached = self.ledger.get(self.model, prompt, self._params)
        if cached is not None:
            self.last_latency_ms = None  # replay: not a timing observation
            return cached
        if self.ledger.remaining <= 0:
            raise QuotaExhausted(
                f"declared budget of {self.ledger.daily_budget} requests is spent "
                f"for {self.ledger.day}"
            )

        body = json.dumps(
            {
                "contents": [{"parts": [{"text": prompt}]}],
                # Reasoning models draw thinking tokens from the SAME output
                # budget, so a tight `maxOutputTokens` silently truncates the
                # visible answer mid-token: we measured "65" where the answer
                # was "65.75" and scored our own compressor down for it.  The
                # obvious fix -- `thinkingConfig: {thinkingBudget: 0}` -- is
                # rejected with HTTP 400 by several models, so we simply give
                # the budget enough headroom for thinking *and* the answer.
                "generationConfig": {
                    "maxOutputTokens": self.max_output_tokens,
                    "temperature": 0.0,
                },
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.endpoint}/{self.model}:generateContent",
            data=body,
            method="POST",
            headers={
                # header, never a URL parameter: query strings land in logs
                "x-goog-api-key": self._key,
                "Content-Type": "application/json",
            },
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode()[:400]
            if exc.code == 429:
                raise QuotaExhausted(f"provider quota exhausted: {detail}") from exc
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc

        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        text = _extract_text(payload)
        self.ledger.put(self.model, prompt, text, self._params)
        return text


def _extract_text(payload: dict[str, Any]) -> str:
    """Pull the reply out, tolerating thinking-model response shapes."""
    out: list[str] = []
    for cand in payload.get("candidates", []):
        for part in cand.get("content", {}).get("parts", []):
            t = part.get("text")
            if t:
                out.append(t)
    return "\n".join(out).strip()


# --------------------------------------------------------------------------
# prompting + scoring
# --------------------------------------------------------------------------
QA_TEMPLATE = """Answer the question using ONLY the context below.
Be terse: reply with the answer itself, no explanation.
If the context does not contain the answer, reply exactly: NOT IN CONTEXT

CONTEXT:
{context}

QUESTION: {question}
ANSWER:"""


def build_prompt(context: str, question: str) -> str:
    return QA_TEMPLATE.format(context=context, question=question)


PACKED_TEMPLATE = """You will be given {n} independent items. Each has its own
CONTEXT and its own QUESTION. Answer each using ONLY that item's context.

Reply with exactly {n} lines, numbered, in order, nothing else:
1. <answer>
2. <answer>
...
If an item's context does not contain the answer, write NOT IN CONTEXT.
Be terse: the answer itself, no explanation.

{items}
ANSWERS:"""

_ITEM_TEMPLATE = """--- ITEM {i} ---
CONTEXT:
{context}

QUESTION {i}: {question}

"""


def build_packed_prompt(items: Sequence[tuple[str, str]]) -> str:
    """Pack several independent (context, question) pairs into one request.

    A 20-request/day free tier makes one-question-per-request unusable for
    research: it caps you at n=6 forever. Packing k items amortises the request
    across k measurements, turning the same quota into n=36/day.

    The cost is honest and bounded: packing changes the task slightly (the model
    sees more material and could interfere across items). But it applies
    *identically to every condition*, so the between-condition comparison — which
    is what the experiment is actually about — stays fair. Absolute accuracies
    from packed runs should not be compared against unpacked ones, which is why
    the pack size is recorded in the cache key and the results file.
    """
    body = "".join(
        _ITEM_TEMPLATE.format(i=i + 1, context=c, question=q) for i, (c, q) in enumerate(items)
    )
    return PACKED_TEMPLATE.format(n=len(items), items=body)


_NUMBERED = re.compile(r"^\s*(\d+)\s*[.):]\s*(.*)$")


def parse_packed_reply(reply: str, n: int) -> list[str]:
    """Split a numbered reply into `n` answers, tolerating formatting drift.

    Returns "" for any item the model did not answer, so a malformed reply
    degrades to missing answers for the affected items rather than silently
    shifting every subsequent answer onto the wrong question.
    """
    out = [""] * n
    current: int | None = None
    for line in reply.splitlines():
        m = _NUMBERED.match(line)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < n:
                current = idx
                out[idx] = m.group(2).strip()
                continue
            current = None
        elif current is not None and line.strip():
            out[current] = (out[current] + " " + line.strip()).strip()
    return out


def score_answer(reply: str, gold: Sequence[str]) -> float:
    """Fraction of gold spans present in the reply (normalised containment).

    Strict and mechanical on purpose: an LLM judge would consume the same scarce
    quota and reintroduce the very uncertainty this measurement exists to remove.
    """
    if not gold:
        return 1.0
    low = canon(reply)
    if not low or low.startswith("not in context"):
        return 0.0
    hits = 0
    for g in gold:
        cg = canon(str(g))
        if cg and cg in low:
            hits += 1
    return hits / len(gold)


@dataclass
class Trial:
    iid: str
    suite: str
    condition: str
    tokens: int
    score: float
    reply: str = ""
    error: str = ""
    #: wall-clock of the request that produced this trial, in ms.  For a packed
    #: request every item in the pack shares the pack's latency; None when the
    #: reply came from cache (a replay is not a timing observation).
    latency_ms: float | None = None


def evaluate_packed(
    judge: Judge,
    condition: str,
    items: Sequence[tuple[str, str, str, str, Sequence[str], int]],
) -> list[Trial]:
    """One request, many measurements.

    ``items`` = [(iid, suite, context, question, gold, tokens)].  A transport
    failure marks every item in the pack as *missing* rather than wrong: one bad
    request must not manufacture k false negatives.
    """
    prompt = build_packed_prompt([(ctx, q) for _i, _s, ctx, q, _g, _t in items])
    try:
        reply = judge.ask(prompt)
    except QuotaExhausted:
        raise
    except Exception as exc:
        return [
            Trial(iid, suite, condition, tok, float("nan"),
                  error=f"{type(exc).__name__}: {exc}")
            for iid, suite, _c, _q, _g, tok in items
        ]

    lat = getattr(judge, "last_latency_ms", None)
    answers = parse_packed_reply(reply, len(items))
    out: list[Trial] = []
    for (iid, suite, _ctx, _q, gold, tok), ans in zip(items, answers, strict=True):
        if not ans:
            out.append(Trial(iid, suite, condition, tok, float("nan"),
                             error="no answer parsed for this item", latency_ms=lat))
        else:
            out.append(Trial(iid, suite, condition, tok, score_answer(ans, gold),
                             reply=ans[:400], latency_ms=lat))
    return out


def evaluate_condition(
    judge: Judge,
    iid: str,
    suite: str,
    context: str,
    question: str,
    gold: Sequence[str],
    condition: str,
    tokens: int,
) -> Trial:
    try:
        reply = judge.ask(build_prompt(context, question))
    except QuotaExhausted:
        raise
    except Exception as exc:  # record and continue; scored as *missing*, not 0
        return Trial(
            iid, suite, condition, tokens, float("nan"),
            error=f"{type(exc).__name__}: {exc}",
        )
    return Trial(
        iid, suite, condition, tokens, score_answer(reply, gold), reply=reply[:400],
        latency_ms=getattr(judge, "last_latency_ms", None),
    )
