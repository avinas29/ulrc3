"""Benchmark suites.

Two sources of data:

**Synthetic, deterministic, gold-labelled.**  Generated from seeded templates so
every instance has *exact* gold answer spans.  This is what makes offline
measurement meaningful: answerability is computed by string containment against
a known answer, not by an LLM judge with its own error bars.  The generators
model the published benchmarks' structure -- needle-in-a-haystack (Kamradt),
multi-hop chains (HotpotQA), numeric chains (GSM8K), repo QA (RepoBench),
long-dialogue memory (ShareGPT/LongBench), log triage, RAG with distractors.

**Real datasets, when present.**  ``load_external()`` picks up HotpotQA,
LongBench, NarrativeQA or a JSONL of ``{context, question, answers}`` from
``bench/data/`` if you drop them there.  Nothing is downloaded automatically --
a benchmark that silently hits the network is not reproducible.

Every instance is ``(id, suite, request, query, answers, entities)``.
"""

from __future__ import annotations

import json
import os
import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@dataclass
class Instance:
    iid: str
    suite: str
    text: str = ""
    documents: list[str] = field(default_factory=list)
    query: str = ""
    system: str = ""
    instruction: str = ""
    answers: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    distractors: list[str] = field(default_factory=list)
    #: The answer a *model* should produce.  Distinct from ``answers``, which is
    #: what must survive in the context: for a reasoning task the inputs must be
    #: present but the model must reply with the computed result.  Conflating
    #: the two scored a perfectly correct "$1413.40" as zero.
    final_answer: list[str] = field(default_factory=list)
    doctype: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def source(self) -> str:
        parts = [p for p in ([self.system, self.instruction, self.text] + self.documents) if p]
        if self.query:
            parts.append(self.query)
        return "\n".join(parts)


# --------------------------------------------------------------------------
# filler corpora
# --------------------------------------------------------------------------
_TOPICS = [
    "quarterly planning", "capacity forecasting", "customer onboarding", "incident response",
    "data retention", "vendor evaluation", "release management", "cost optimisation",
    "access reviews", "schema migration", "load testing", "documentation standards",
]
_VERBS = ["reviewed", "considered", "discussed", "evaluated", "documented", "revisited", "audited"]

#: **Distractors, not filler.**
#:
#: These sentences are the haystack a needle hides in, and their only job is to
#: be *irrelevant to the question* -- not to be deletable.  The previous pool
#: was eight hedging phrases ("It is important to note that...", "Generally
#: speaking..."), six of which matched the engine's own ``HEDGE``/``FILLER``
#: lexicon verbatim, and they made up 41-59% of every document.  A benchmark
#: whose padding is drawn from the compressor's own delete-list measures the
#: lexicon, not the compressor: an independent reviewer diffing
#: ``bench/datasets.py`` against ``ulrc3/text/lexicon.py`` would find the
#: overlap in a minute and discount every number in the repository.
#:
#: These replacements are ordinary declarative domain prose.  They carry real
#: entities, numbers and dates, so removing them requires deciding they are
#: *unhelpful for this query* -- which is the actual claim being tested -- and
#: they are matched by no rule in the engine.  Compression on this corpus fell
#: by roughly 9 points when they were introduced; that drop is the measurement
#: error the old pool was hiding.
_FILLER = [
    "The Helsinki data centre completed its annual power maintenance on 14 March 2023.",
    "Vendor contracts for the analytics tier renew on a rolling 18-month schedule.",
    "Support tickets are routed to the platform rota after two failed triage attempts.",
    "The design review board meets on alternate Thursdays in meeting room B4.",
    "Storage costs for cold archives fell to 0.004 USD per gigabyte last quarter.",
    "Regional failover was exercised twice in 2023 with a mean recovery of 41 minutes.",
    "The internal style guide requires British spelling in customer-facing copy.",
    "Onboarding for contractors requires a background check and a signed NDA.",
    "Build agents run on Ubuntu 22.04 with a 90-minute job timeout.",
    "The procurement team tracks 62 active suppliers across four purchasing regions.",
    "Quarterly access reviews are signed off by the owning director before month end.",
    "Test fixtures are regenerated whenever the upstream schema version changes.",
    "The Dublin office moved to a four-day support rota in September.",
    "Archived tickets older than 400 days are exported to the compliance bucket.",
    "Marketing attribution uses a 30-day last-touch window across all channels.",
    "The hardware refresh cycle replaces developer laptops every 36 months.",
]


def _paragraph(rng: random.Random, n: int = 4) -> str:
    out = []
    for _ in range(n):
        t = rng.choice(_TOPICS)
        v = rng.choice(_VERBS)
        out.append(f"The working group {v} {t} during the session.")
        out.append(rng.choice(_FILLER))
    return " ".join(out)


def _haystack(rng: random.Random, paragraphs: int) -> list[str]:
    return [_paragraph(rng, rng.randint(3, 6)) for _ in range(paragraphs)]


# --------------------------------------------------------------------------
# suites
# --------------------------------------------------------------------------
def needle(n: int = 24, size: int = 40, seed: int = 11) -> Iterator[Instance]:
    """Needle in a haystack, at controlled relative depths."""
    rng = random.Random(seed)
    for i in range(n):
        paras = _haystack(rng, size)
        code = f"NDL-{rng.randint(1000, 9999)}"
        value = f"{rng.randint(11, 99)}.{rng.randint(10, 99)}"
        needle_txt = (
            f"The verification code for the Atlas deployment is {code} "
            f"and the approved threshold is {value} percent."
        )
        depth = i / max(1, n - 1)
        pos = int(depth * (len(paras) - 1))
        paras.insert(pos, needle_txt)
        yield Instance(
            iid=f"needle-{i:03d}",
            suite="needle",
            text="\n\n".join(paras),
            query="What is the verification code for the Atlas deployment and the approved threshold?",
            answers=[code, value],
            entities=["Atlas"],
            meta={"depth": round(depth, 3)},
        )


def multihop(n: int = 20, distractors: int = 12, seed: int = 12) -> Iterator[Instance]:
    """Two-hop chains split across documents, with lexically similar distractors."""
    rng = random.Random(seed)
    for i in range(n):
        person = f"Dr. {rng.choice(['Okafor','Lindqvist','Moreau','Tanaka','Rossi','Haddad'])}"
        lab = f"{rng.choice(['Helix','Quantum','Northstar','Meridian'])} Lab"
        city = rng.choice(["Reykjavik", "Valparaiso", "Gothenburg", "Kaunas"])
        year = rng.randint(2015, 2023)
        docs = [
            f"{person} has directed {lab} since {year}. The appointment followed an internal review.",
            f"{lab} is located in {city} and employs {rng.randint(40, 400)} researchers.",
        ]
        for _ in range(distractors):
            other = f"Dr. {rng.choice(['Silva','Novak','Ferreira','Kim','Weber'])}"
            olab = f"{rng.choice(['Orion','Vertex','Cascade','Summit'])} Lab"
            docs.append(
                f"{other} has directed {olab} since {rng.randint(2010, 2022)}. "
                f"{olab} is located in {rng.choice(['Porto','Ottawa','Dresden','Perth'])}. "
                + _paragraph(rng, 2)
            )
        rng.shuffle(docs)
        yield Instance(
            iid=f"multihop-{i:03d}",
            suite="multihop",
            documents=docs,
            query=f"In which city is the lab directed by {person} located?",
            answers=[city, lab],
            entities=[person, lab, city],
            meta={"hops": 2, "distractors": distractors},
        )


def numeric(n: int = 20, noise: int = 14, seed: int = 13) -> Iterator[Instance]:
    """Numeric reasoning chains buried in narrative (GSM8K-shaped)."""
    rng = random.Random(seed)
    for i in range(n):
        units = rng.randint(12, 60)
        price = rng.randint(15, 90)
        discount = rng.choice([5, 10, 15, 20])
        shipping = rng.randint(20, 120)
        subtotal = units * price
        total = round(subtotal * (1 - discount / 100) + shipping, 2)
        facts = [
            f"The order contains {units} units.",
            f"Each unit is priced at ${price}.",
            f"A volume discount of {discount}% applies to the subtotal.",
            f"Shipping is a flat ${shipping} per order.",
        ]
        paras = _haystack(rng, noise)
        for _j, f in enumerate(facts):
            paras.insert(rng.randint(0, len(paras)), f)
        yield Instance(
            iid=f"numeric-{i:03d}",
            suite="numeric",
            text="\n\n".join(paras),
            query="What is the order total after discount and shipping?",
            answers=[str(units), str(price), str(discount), str(shipping)],
            final_answer=[f"{total:.2f}".rstrip("0").rstrip(".")],
            meta={"expected_total": total},
        )


PY_TEMPLATE = '''"""{mod} utilities.

Copyright (c) 2024 Example Corp. All rights reserved.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)
DEFAULT_TIMEOUT_MS = {timeout}


@dataclass
class {cls}Config:
    """Configuration for {cls}."""

    endpoint: str
    timeout_ms: int = DEFAULT_TIMEOUT_MS
    retries: int = {retries}


class {cls}:
    """{cls} client."""

    def __init__(self, config: {cls}Config) -> None:
        self.config = config
        self._calls = 0

    def {method}(self, {arg}: str, limit: int = {limit}) -> list[str]:
        """Fetch up to `limit` records for {arg}."""
        if limit <= 0:
            raise ValueError("limit must be positive")
        self._calls += 1
        return [f"{{{arg}}}-{{i}}" for i in range(limit)]

    def close(self) -> None:
        """Release resources."""
        self._calls = 0


def build_{lower}(endpoint: str) -> {cls}:
    """Construct a {cls} with default configuration."""
    return {cls}({cls}Config(endpoint=endpoint))


def _unused_helper(values: list[int]) -> int:
    total = 0
    for v in values:
        if v > 0:
            total += v
    return total
'''


def code_suite(n: int = 16, files: int = 6, seed: int = 14) -> Iterator[Instance]:
    """Repository-scale code QA: find a symbol's signature among many files."""
    rng = random.Random(seed)
    names = ["Ledger", "Router", "Indexer", "Scheduler", "Registry", "Collector", "Broker"]
    for i in range(n):
        docs = []
        target_cls = rng.choice(names)
        target_method = rng.choice(["fetch", "lookup", "resolve", "collect"])
        target_arg = rng.choice(["tenant_id", "account_id", "series_id"])
        target_limit = rng.randint(10, 500)
        for j in range(files):
            cls = target_cls if j == 0 else rng.choice([x for x in names if x != target_cls])
            method = target_method if j == 0 else rng.choice(["fetch", "lookup", "resolve"])
            arg = target_arg if j == 0 else rng.choice(["tenant_id", "key", "uid"])
            limit = target_limit if j == 0 else rng.randint(10, 500)
            docs.append(
                PY_TEMPLATE.format(
                    mod=cls.lower(), cls=cls, lower=cls.lower(), method=method, arg=arg,
                    limit=limit, timeout=rng.choice([250, 500, 1000]), retries=rng.randint(2, 5),
                )
            )
        rng.shuffle(docs)
        yield Instance(
            iid=f"code-{i:03d}",
            suite="code",
            documents=docs,
            query=f"What is the signature of {target_cls}.{target_method} and its default limit?",
            answers=[f"def {target_method}", target_arg, str(target_limit), target_cls],
            doctype="code",
            meta={"files": files},
        )


def memory_suite(n: int = 16, turns: int = 40, seed: int = 15) -> Iterator[Instance]:
    """Long dialogue with corrections: the final value must win."""
    rng = random.Random(seed)
    for i in range(n):
        plan_a = rng.choice(["Enterprise", "Platinum", "Scale"])
        plan_b = rng.choice(["Business", "Starter", "Growth"])
        price_a = rng.randint(3000, 6000)
        price_b = rng.randint(800, 2500)
        email = f"ap{rng.randint(10,99)}@example.test"
        lines = []
        for t in range(turns):
            if t == 4:
                lines.append(f"user: We want the {plan_a} plan.")
                lines.append(f"assistant: The {plan_a} plan costs ${price_a} per month.")
            elif t == turns // 2:
                lines.append(f"user: Actually, correction: we want the {plan_b} plan, not {plan_a}.")
                lines.append(f"assistant: Understood, the {plan_b} plan is ${price_b} per month.")
            elif t == turns - 6:
                lines.append(f"user: Never email invoices anywhere except {email}.")
                lines.append("assistant: Noted.")
            else:
                lines.append(f"user: {rng.choice(_FILLER)}")
                lines.append(f"assistant: Certainly! {rng.choice(_FILLER)}")
        yield Instance(
            iid=f"memory-{i:03d}",
            suite="memory",
            text="\n".join(lines),
            query="Which plan did the customer finally choose, at what price, and where should invoices go?",
            answers=[plan_b, str(price_b), email],
            distractors=[str(price_a)],
            doctype="conversation",
            meta={"turns": turns, "superseded": plan_a},
        )


def logs_suite(n: int = 14, lines: int = 400, seed: int = 16) -> Iterator[Instance]:
    """High-redundancy logs with a rare fatal event."""
    rng = random.Random(seed)
    for i in range(n):
        order = rng.randint(10000, 99999)
        latency = rng.randint(1000, 9000)
        out = []
        for j in range(lines):
            ts = f"2024-03-15T10:{j // 60 % 60:02d}:{j % 60:02d}Z"
            if j == lines - 30:
                out.append(f"{ts} FATAL db.pool connection pool exhausted: 100/100 in use")
            elif j == lines - 20:
                out.append(
                    f"{ts} ERROR billing.worker payment failed for order {order}: "
                    f"gateway timeout after {latency}ms"
                )
            else:
                out.append(
                    f"{ts} INFO api.gateway request GET /v2/invoices status=200 "
                    f"dur={rng.randint(8, 40)}ms"
                )
        yield Instance(
            iid=f"logs-{i:03d}",
            suite="logs",
            text="\n".join(out),
            query="What fatal error occurred and which order failed?",
            answers=[str(order), "pool exhausted", "100/100"],
            doctype="logs",
            meta={"lines": lines},
        )


def rag_suite(n: int = 20, chunks: int = 14, seed: int = 17) -> Iterator[Instance]:
    """Retrieved chunks: one gold, several near-duplicates, many distractors."""
    rng = random.Random(seed)
    for i in range(n):
        rev = f"${rng.randint(10, 90)}.{rng.randint(0, 9)}M"
        growth = f"{rng.randint(5, 45)}%"
        margin = f"{rng.randint(50, 85)}%"
        gold = (
            f"Quarterly revenue for Q4 2024 was {rev}, up {growth} year over year. "
            f"Gross margin was {margin}."
        )
        para = (
            f"In Q4 2024 the company reported revenue of {rev}, an increase of {growth} "
            f"year over year, with a gross margin of {margin}."
        )
        docs = [gold, para]
        for _ in range(chunks - 2):
            docs.append(_paragraph(rng, rng.randint(3, 5)))
        rng.shuffle(docs)
        yield Instance(
            iid=f"rag-{i:03d}",
            suite="rag",
            documents=docs,
            query="What was Q4 2024 revenue, growth and gross margin?",
            answers=[rev, growth, margin],
            meta={"chunks": chunks},
        )


def apidocs_suite(n: int = 14, endpoints: int = 18, seed: int = 18) -> Iterator[Instance]:
    """API reference: preserve endpoint, params, status codes."""
    rng = random.Random(seed)
    verbs = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    for i in range(n):
        target = f"/v2/{rng.choice(['invoices','accounts','ledgers','payouts'])}/{{id}}"
        tverb = rng.choice(verbs)
        tparam = rng.choice(["expand", "include_deleted", "cursor"])
        tcode = rng.choice([402, 409, 422, 429])
        blocks = [
            f"## {tverb} {target}\n\n"
            f"Retrieves a single record. Parameters: `{tparam}` (string, optional), "
            f"`limit` (integer, max 100).\n\n"
            f"Returns 200 on success, {tcode} when the request is rejected.\n\n"
            + _paragraph(rng, 3)
        ]
        for _ in range(endpoints - 1):
            blocks.append(
                f"## {rng.choice(verbs)} /v2/{rng.choice(['tasks','users','webhooks'])}\n\n"
                f"Standard endpoint. Parameters: `{rng.choice(['page','sort','filter'])}` (string).\n\n"
                f"Returns 200 on success, {rng.choice([400,401,404])} on error.\n\n"
                + _paragraph(rng, 3)
            )
        rng.shuffle(blocks)
        yield Instance(
            iid=f"apidocs-{i:03d}",
            suite="apidocs",
            text="\n\n".join(blocks),
            query=f"What parameters does {tverb} {target} accept and what error code can it return?",
            answers=[tparam, str(tcode), target.split("/")[2]],
            meta={"endpoints": endpoints},
        )


def mixed_suite(n: int = 12, seed: int = 19) -> Iterator[Instance]:
    """Polyglot documents: prose + code + json + logs in one blob."""
    rng = random.Random(seed)
    for i in range(n):
        token = f"tok_{rng.randint(100000, 999999)}"
        rate = rng.randint(30, 300)
        blob = (
            "# Incident Report\n\n"
            + _paragraph(rng, 4)
            + f"\n\nThe rate limit must never exceed {rate} requests per minute.\n\n"
            + "```python\n"
            + f"def authenticate(token: str = \"{token}\") -> bool:\n"
            + "    if not token:\n        return False\n    return True\n"
            + "```\n\n"
            + "```json\n"
            + f'{{"limit": {rate}, "token": "{token}", "enabled": true}}\n'
            + "```\n\n"
            + "2024-03-15T10:22:05Z ERROR auth.service token rejected\n"
            + "2024-03-15T10:22:06Z ERROR auth.service token rejected\n\n"
            + _paragraph(rng, 3)
        )
        yield Instance(
            iid=f"mixed-{i:03d}",
            suite="mixed",
            text=blob,
            query="What is the rate limit and the default token?",
            answers=[str(rate), token, "authenticate"],
            meta={},
        )


SUITES = {
    "needle": needle,
    "multihop": multihop,
    "numeric": numeric,
    "code": code_suite,
    "memory": memory_suite,
    "logs": logs_suite,
    "rag": rag_suite,
    "apidocs": apidocs_suite,
    "mixed": mixed_suite,
}


def load(suite: str, quick: bool = False) -> list[Instance]:
    fn = SUITES[suite]
    if quick:
        return list(fn(n=4))  # type: ignore[call-arg]
    return list(fn())


def load_all(quick: bool = False) -> dict[str, list[Instance]]:
    return {name: load(name, quick) for name in SUITES}


# --------------------------------------------------------------------------
# external datasets (opt-in, never downloaded automatically)
# --------------------------------------------------------------------------
def load_external(path: str | None = None, limit: int = 200) -> list[Instance]:
    """Load ``{context|documents, question, answers}`` JSONL from ``bench/data/``.

    Compatible with dumps of HotpotQA, 2WikiMultihopQA, NaturalQuestions,
    LongBench and NarrativeQA once converted to that shape (one JSON object per
    line).  Absent files yield an empty list -- the suite simply does not run.
    """
    root = path or DATA_DIR
    out: list[Instance] = []
    if not os.path.isdir(root):
        return out
    for fn in sorted(os.listdir(root)):
        if not fn.endswith((".jsonl", ".json")):
            continue
        full = os.path.join(root, fn)
        try:
            with open(full, encoding="utf-8") as fh:
                if fn.endswith(".jsonl"):
                    rows = [json.loads(line) for line in fh if line.strip()]
                else:
                    payload = json.load(fh)
                    rows = payload if isinstance(payload, list) else [payload]
        except Exception:
            continue
        for i, row in enumerate(rows[:limit]):
            ctx = row.get("context") or row.get("input") or ""
            docs = row.get("documents") or row.get("contexts") or []
            if isinstance(ctx, list):
                docs = docs or ctx
                ctx = ""
            answers = row.get("answers") or row.get("answer") or []
            if isinstance(answers, str):
                answers = [answers]
            out.append(
                Instance(
                    iid=f"{os.path.splitext(fn)[0]}-{i:04d}",
                    suite=f"external:{os.path.splitext(fn)[0]}",
                    text=ctx if isinstance(ctx, str) else "",
                    documents=[d if isinstance(d, str) else json.dumps(d) for d in docs],
                    query=row.get("question") or row.get("query") or "",
                    answers=[str(a) for a in answers],
                )
            )
    return out
