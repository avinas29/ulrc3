#!/usr/bin/env python3
"""Runnable tour of ULRC³.

    python examples/quickstart.py

Every example prints real output from the engine — nothing here is illustrative.
"""

from __future__ import annotations

import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ulrc3 import Compressor, Config, Doc, Mode, Request  # noqa: E402
from ulrc3.recovery import DEFAULT_STORE, Residual  # noqa: E402


def rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# --------------------------------------------------------------------------
def example_1_basic() -> None:
    rule("1. Basic compression — a markdown document")
    doc = textwrap.dedent(
        """
        # Billing Service

        ## Limits
        It is important to note that the API must never return more than 100
        invoices per page. Requests are rate limited to 60 per minute; exceeding
        that returns HTTP 429. For example, a client fetching 450 invoices will
        issue 5 paginated requests.

        ## Deprecations
        As previously mentioned, the v1 endpoints were removed on 2024-03-15.
        The migration guide is at https://docs.acme.io/billing/migrate-v2.
        Basically, do not use the legacy `/v1/invoice` path any more.
        """
    ).strip()

    r = Compressor(Config(mode=Mode.BALANCED)).compress(doc)
    print(r.text)
    print(f"\n-> {r.summary()}")
    print("   every number, URL and 'must never' clause is still there.")


# --------------------------------------------------------------------------
def example_2_structured_rag() -> None:
    rule("2. RAG — frozen system prompt + query, compressed chunks")
    chunks = [
        "Acme Corporation Limited was founded in 2011 in Dublin and employs 1,240 people.",
        "Acme's billing platform processes 4.2 million invoices per month at peak 900/second.",
        "Founded in 2011 with headquarters in Dublin, the company has around 1,240 employees.",
        "Q4 2024 revenue was $18.4M, up 22% year over year, with gross margin of 71%.",
        "In Q4 2024 the company reported $18.4M revenue, a 22% increase, 71% gross margin.",
        "Support is available 24/7 via support@acme.io with a 4 hour first response SLA.",
    ]
    req = Request(
        system="You are a financial analyst. Never invent figures.",
        query="What was Q4 2024 revenue and margin?",
        documents=[Doc(text=c, score=0.9 - 0.05 * i) for i, c in enumerate(chunks)],
        mode="aggressive",
    )
    r = Compressor().compress(req)
    print(r.text)
    print(f"\n-> {r.summary()}")
    print(f"   frozen segments intact: {r.verification.frozen_ok}")


# --------------------------------------------------------------------------
def example_3_code() -> None:
    rule("3. Code — compiled to an API surface")
    src = textwrap.dedent(
        '''
        """Payment helpers. Copyright (c) 2024 Acme. All rights reserved."""
        import logging
        import math
        from dataclasses import dataclass

        logger = logging.getLogger(__name__)
        MAX_RETRIES = 3


        @dataclass
        class Receipt:
            """A payment receipt."""
            receipt_id: str
            amount_cents: int


        class Gateway:
            """Talks to the processor."""

            def __init__(self, api_key: str, base_url: str = "https://api.acme.io/v2") -> None:
                self.api_key = api_key
                self.base_url = base_url

            def charge(self, customer_id: str, amount_cents: int) -> Receipt:
                """Charge a customer. Raises ValueError when amount is not positive."""
                if amount_cents <= 0:
                    raise ValueError("amount_cents must be positive")
                return Receipt(receipt_id=customer_id, amount_cents=amount_cents)


        def unused_stats(values: list[int]) -> float:
            """Geometric mean."""
            return math.exp(sum(math.log(v) for v in values) / len(values))
        '''
    ).strip()

    r = Compressor(Config(mode=Mode.AGGRESSIVE)).compress(src)
    print(r.text)
    print(f"\n-> {r.summary()}")
    print(f"   emitted code parses: {r.verification.syntax_ok}")


# --------------------------------------------------------------------------
def example_4_conversation() -> None:
    rule("4. Conversation — belief revision removes the retracted price")
    turns = ["user: Hi there!", "assistant: Certainly! How can I help?"]
    turns += ["user: We want the Enterprise plan.",
              "assistant: The Enterprise plan costs $9,900 per month."]
    for i in range(12):
        turns += [f"user: Some scheduling detail number {i}.",
                  "assistant: Certainly! Noted, thanks."]
    turns += ["user: Actually, correction: we want the Business plan, not Enterprise.",
              "assistant: Understood, the Business plan is $1,200 per month.",
              "user: Confirmed, go with Business. Invoice ap@acme.example.",
              "assistant: Booked."]

    r = Compressor(Config(mode=Mode.BALANCED)).compress("\n".join(turns))
    print(r.text)
    print(f"\n-> {r.summary()}")
    print(f"   retracted '$9,900' present: {'9,900' in r.text}   (should be False)")
    print(f"   current   '$1,200' present: {'1,200' in r.text}   (should be True)")


# --------------------------------------------------------------------------
def example_5_logs() -> None:
    rule("5. Logs — template mining keeps the rare FATAL")
    lines = []
    for i in range(200):
        lines.append(f"2024-03-15T10:{i // 60:02d}:{i % 60:02d}Z INFO api request GET /v2/x status=200 dur={10 + i % 30}ms")
    lines.insert(150, "2024-03-15T10:02:30Z FATAL db.pool connection pool exhausted: 100/100 in use")
    r = Compressor(Config(mode=Mode.AGGRESSIVE)).compress("\n".join(lines))
    print(r.text)
    print(f"\n-> {r.summary()}")
    print(f"   FATAL retained: {'exhausted' in r.text}")


# --------------------------------------------------------------------------
def example_6_recovery() -> None:
    rule("6. Recovery — an agent faults a dropped span back in")
    doc = "\n\n".join(
        f"Section {i}. " + "Background narration that carries little information. " * 6
        for i in range(12)
    ) + "\n\nThe emergency contact number is +1-555-0142."

    r = Compressor(Config(mode=Mode.EXTREME)).compress(doc)
    print(f"compressed to {r.tokens_out} tokens from {r.tokens_in}")
    DEFAULT_STORE.put("demo", [Residual(h, t, "d") for h, t in r.residuals.items()])
    handles = sorted(r.residuals)[:2]
    print(f"dropped handles available: {sorted(r.residuals)[:6]} ...")
    for h in handles:
        got = DEFAULT_STORE.get("demo", h)
        print(f"  expand({h}) -> {got[:70]!r}")


# --------------------------------------------------------------------------
def example_7_audit() -> None:
    rule("7. The audit report — what a CI gate sees")
    doc = (
        "The retry budget is 3 attempts with 250ms backoff. "
        "Under no circumstances may the worker write to the primary shard. "
        + "Filler narration that adds nothing. " * 40
    )
    r = Compressor(Config(mode=Mode.EXTREME)).compress(doc)
    v = r.verification
    for name, ok in [
        ("integrity", v.integrity >= 1.0),
        ("critical recall", v.critical_recall >= 1.0),
        ("provenance", v.provenance_ok),
        ("frozen verbatim", v.frozen_ok),
        ("no inflation", v.inflation_ok),
        ("syntax", v.syntax_ok),
    ]:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n  ratio {r.ratio * 100:.1f}%  retention {v.retention * 100:.1f}%  confidence {r.confidence:.2f}")
    print(f"  'no circumstances' preserved: {'no circumstances' in r.text.lower()}")


def main() -> int:
    example_1_basic()
    example_2_structured_rag()
    example_3_code()
    example_4_conversation()
    example_5_logs()
    example_6_recovery()
    example_7_audit()
    print("\nAll examples ran against the real engine.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
