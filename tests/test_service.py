"""CLI, HTTP API, recovery layer, cache and parallel batch tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
from fixtures import ALL_FIXTURES, MARKDOWN_DOC, RAG_CHUNKS

from ulrc3 import Compressor, Config, Mode
from ulrc3.cache import CacheKey, ChunkCache, DocumentCache, budget_class, query_class
from ulrc3.cli import main as cli_main
from ulrc3.recovery import Residual, ResidualStore

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- CLI
def test_cli_compress_stdout(capsys, tmp_path):
    p = tmp_path / "doc.md"
    p.write_text(MARKDOWN_DOC, encoding="utf-8")
    rc = cli_main(["compress", str(p), "-m", "balanced"])
    out = capsys.readouterr().out
    assert rc == 0 and len(out) > 40
    assert len(out) < len(MARKDOWN_DOC)


def test_cli_json_output(capsys, tmp_path):
    p = tmp_path / "doc.md"
    p.write_text(MARKDOWN_DOC, encoding="utf-8")
    cli_main(["compress", str(p), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["tokens_out"] < payload["tokens_in"]
    assert payload["verification"]["integrity"] == 1.0
    assert payload["passes"]


def test_cli_verify_exit_code(tmp_path, capsys):
    p = tmp_path / "doc.md"
    p.write_text(MARKDOWN_DOC, encoding="utf-8")
    assert cli_main(["verify", str(p)]) == 0
    assert "PASS" in capsys.readouterr().out


def test_cli_inspect_reports_passes(tmp_path, capsys):
    p = tmp_path / "doc.py"
    p.write_text(ALL_FIXTURES["code"], encoding="utf-8")
    cli_main(["inspect", str(p)])
    out = capsys.readouterr().out
    assert "cascade-select" in out and "integrity" in out


def test_cli_module_entrypoint_runs():
    res = subprocess.run(
        [sys.executable, "-m", "ulrc3.cli", "--version"],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert res.returncode == 0 and "ulrc3" in res.stdout


# ---------------------------------------------------------------- recovery
def test_residual_store_roundtrip_and_ttl():
    store = ResidualStore(max_chars=1000, ttl_seconds=60)
    store.put("s1", [Residual("7", "the original paragraph", "d")])
    assert store.get("s1", "7") == "the original paragraph"
    assert store.expand("s1", ["7", "999"]) == {"7": "the original paragraph"}
    store.drop("s1")
    assert store.get("s1", "7") is None


def test_residual_store_evicts_by_size():
    store = ResidualStore(max_chars=200, ttl_seconds=0)
    for i in range(20):
        store.put(f"s{i}", [Residual("1", "x" * 100, "d")])
    assert store.stats()["chars"] <= 200


def test_compression_result_exposes_residual_handles():
    res = Compressor(Config(mode=Mode.AGGRESSIVE)).compress(MARKDOWN_DOC)
    assert res.residuals, "dropped spans must remain addressable"
    handle, text = next(iter(res.residuals.items()))
    assert text.strip() and text in MARKDOWN_DOC


# ---------------------------------------------------------------- cache
def test_chunk_cache_lru_and_hit_rate():
    c = ChunkCache(capacity=2)
    k1, k2, k3 = (CacheKey(f"h{i}", 1, "q", "balanced") for i in range(3))
    c.put(k1, "a")
    c.put(k2, "b")
    assert c.get(k1) == "a"
    c.put(k3, "c")
    assert c.get(k2) is None  # evicted (least recently used)
    assert 0.0 < c.hit_rate <= 1.0


def test_budget_and_query_classes_are_stable():
    assert budget_class(1000) == budget_class(1010)
    assert query_class("what is the revenue") == query_class("the revenue, what is")
    assert query_class("revenue") != query_class("kubernetes networking")


def test_document_cache_reuses_chunks():
    calls = {"n": 0}

    def fake(text: str) -> str:
        calls["n"] += 1
        return text[:40]

    body = "".join(f"paragraph {i} with some text to chunk over\n" for i in range(400))
    dc = DocumentCache(ChunkCache())
    out1, hits1 = dc.compress_cached(body, fake, 500)
    first_calls = calls["n"]
    out2, hits2 = dc.compress_cached(body, fake, 500)
    assert out1 == out2
    assert hits2 == 1.0 and calls["n"] == first_calls


# ---------------------------------------------------------------- batch
def test_compress_many_matches_sequential():
    payloads = [ALL_FIXTURES["markdown"], ALL_FIXTURES["logs"], ALL_FIXTURES["json"]]
    eng = Compressor(Config(mode=Mode.BALANCED))
    seq = [eng.compress(p).text for p in payloads]
    batch = [r.text for r in eng.compress_many(payloads)]
    assert seq == batch


# ---------------------------------------------------------------- HTTP
@pytest.fixture(scope="module")
def client():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from ulrc3.server.app import app

    return TestClient(app)


def test_health(client):
    r = client.get("/v1/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_compress_endpoint(client):
    r = client.post(
        "/v1/compress",
        json={
            "documents": list(RAG_CHUNKS),
            "query": "What was Q4 2024 revenue?",
            "system": "You are a careful analyst. Never invent figures.",
            "mode": "aggressive",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["tokens_out"] < body["tokens_in"]
    assert body["verification"]["integrity"] == 1.0
    assert body["verification"]["frozen_ok"]
    assert "Never invent figures" in body["text"]
    assert body["session"]


def test_expand_endpoint_round_trip(client):
    r = client.post("/v1/compress", json={"text": MARKDOWN_DOC, "mode": "extreme", "session": "sess-1"})
    handles = r.json()["expandable"]
    assert handles
    e = client.post("/v1/expand", json={"session": "sess-1", "handles": handles[:3]})
    assert e.status_code == 200
    spans = e.json()["spans"]
    assert spans and all(v.strip() for v in spans.values())


def test_estimate_endpoint(client):
    r = client.post("/v1/estimate", json={"text": MARKDOWN_DOC})
    body = r.json()
    assert body["tokens"] > 0
    assert body["projected"]["balanced"] < body["tokens"]


def test_empty_request_is_rejected(client):
    assert client.post("/v1/compress", json={"text": ""}).status_code == 400


def test_stream_endpoint_emits_events(client):
    with client.stream("POST", "/v1/compress/stream", json={"text": MARKDOWN_DOC}) as r:
        events = [json.loads(line) for line in r.iter_lines() if line.strip()]
    kinds = [e["event"] for e in events]
    assert kinds[0] == "start" and kinds[-1] == "done"
    assert "result" in kinds and "pass" in kinds


# -- hardening -------------------------------------------------------------
def test_oversized_payload_is_rejected_not_processed(client, monkeypatch):
    """A payload cap must reject at the edge.

    Without it one client holds a worker for unbounded CPU: a 20 000-key JSON
    body measured 4.7 s single-threaded before the guard existed.
    """
    from ulrc3.server import app as srv

    monkeypatch.setattr(srv, "_MAX_CHARS", 1000)
    r = client.post("/v1/compress", json={"text": "x" * 5000})
    assert r.status_code == 413
    assert "payload too large" in r.json()["detail"]


def test_api_key_enforced_only_when_configured(client, monkeypatch):
    """Unset key = open service (the demo); set key = every request checked."""
    from ulrc3.server import app as srv

    assert client.post("/v1/compress", json={"text": "hello world"}).status_code == 200
    monkeypatch.setattr(srv, "_API_KEY", "secret")
    assert client.post("/v1/compress", json={"text": "hello world"}).status_code == 401
    ok = client.post(
        "/v1/compress", json={"text": "hello world"}, headers={"x-api-key": "secret"}
    )
    assert ok.status_code == 200
