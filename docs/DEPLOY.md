# Deploy & demo

Two things you need: a URL that works when a judge clicks it, and a 3-minute
story that lands. This covers both.

---

## 1. Deploy to Render (5 minutes)

The repo ships a blueprint, so there is no dashboard configuration to get wrong.

```bash
cd ulrc3
git init && git add -A && git commit -m "ULRC3 context compression engine"
gh repo create ulrc3 --public --source=. --push     # or push to GitHub manually
```

Then in Render:

1. **New → Blueprint**
2. Pick the repo. Render reads `render.yaml` and configures everything.
3. **Apply**. First build takes ~4 minutes (it compiles the image and pre-caches
   the BPE table so the first request isn't slow).

Your service lands at `https://ulrc3.onrender.com` (or whatever name Render
assigns). That URL serves:

| path | what |
|---|---|
| `/` | **the demo console** — this is what you show judges |
| `/docs` | auto-generated OpenAPI docs, good for "is this a real API?" |
| `/v1/compress` | the endpoint |
| `/v1/health` | liveness + cache stats |

Verified locally with Render's exact contract (`PORT` injected at runtime,
non-root user, read-only filesystem, healthcheck on `/v1/health`).

### ⚠️ The one thing that will bite you

**Render's free plan spins the service down after ~15 minutes idle.** The next
request then takes **~50 seconds** while the container cold-starts. If a judge
clicks your link cold, they see a spinner and move on.

Mitigations, in order of reliability:

1. **Open the URL yourself 2 minutes before you present.** It stays warm for 15
   minutes of activity. This is enough for any live demo.
2. Add a free uptime pinger (UptimeRobot, cron-job.org) hitting `/v1/health`
   every 10 minutes. Keeps it permanently warm.
3. **Have localhost running as a backup.** `make serve` on your laptop, browser
   tab already open. If the network or Render misbehaves mid-pitch, switch tabs.

Upgrading to Render's paid Starter tier removes the spin-down entirely — worth
$7 for a competition weekend if you can.

### Other hosts

The image is a plain Docker container binding `$PORT`, so it deploys unchanged
to Fly.io, Railway, Cloud Run or Heroku:

```bash
fly launch --dockerfile Dockerfile      # Fly
gcloud run deploy --source .            # Cloud Run
```

---

## 2. The demo (3 minutes)

The console has four sample buttons. They are ordered deliberately — each one
shows a thing the competition cannot do.

### Opening line

> "Every prompt compressor asks *which tokens can I delete*. That question has
> no notion of correctness — you can't ask it whether the invoice number
> survived. We treat the prompt as source code and compile it, then **verify**
> the result. Watch the badges."

### Beat 1 — **API docs** (the baseline claim)

Click **API docs → Compress**.

Point at: **~45–75% reduction**, and the five green badges.

> "Integrity 100% means nothing we kept was partially destroyed. No invented
> words means the output vocabulary is a subset of the input — zero
> hallucination, checked per request, not promised."

### Beat 2 — **Logs** (the knockout)

Click **Logs → Compress**. Watch it go to **~96%**.

> "Sixty near-identical log lines collapse into one template with a count. But
> look — the FATAL line and the failing order number are still there. Every
> baseline we tested scores **zero** on this: importance-ranking a sea of
> identical INFO lines picks identical INFO lines."

### Beat 3 — **Python code** (the differentiator)

Click **Python code → Compress**.

> "Function bodies are gone; every **signature is byte-identical**. Imports the
> retained code needs are kept, by dependency closure. And the output still
> parses — we run `ast.parse` on it, so 'compression that breaks your code' is
> a bug we detect."

### Beat 4 — **Chat memory** (the one nobody else has)

Click **Chat memory → Compress**.

> "The user said Enterprise at $9,900, then corrected to Business at $1,200.
> The retracted price is **gone** — not down-ranked, forbidden. Every
> importance-based method we tested keeps both, so the model sees two
> contradictory prices. We measured 100% contradiction rate for them, 0% for us."

### Closing — the number that matters

> "We verified this end-to-end against a real model, not just our own metrics.
> On 87% fewer tokens Gemini answered **80.1%** correctly versus **82.2%** on
> the full prompt — **97.5% retention**. Truncation at the same token count
> scored **10.6%**. That's +69 points, p < 0.0001."

Then open **Pipeline telemetry** to show the 13 passes with per-pass timings.
It reads as engineering rather than a demo script.

---

## 3. Questions judges will ask

**"Is this just summarisation?"**
No — it never generates a word. Every output token is a span of your input plus
~90 structural markers. That's what makes the zero-hallucination check
decidable. A summariser cannot make that claim at any model size.

**"What's your model?"**
None. No proxy LM, no GPU, no API key, no download. 42,000 tokens/second on one
CPU core. LLMLingua needs a 7B model to decide what to delete from your prompt.

**"How do I know it didn't drop something important?"**
It tells you, per request. Show them the badges, then `/docs` → the
`verification` object. `ulrc3 verify` exits non-zero on violation, so it works
as a CI gate.

**"What doesn't work?"**
Answer this one honestly — it's the strongest thing you can say. Number
retention inside aggregate content (logs, table rows) is ~43% by design;
`numeric` reasoning needs `conservative` mode; the latency speedup is not
established (n=2). It's all in [ROADMAP.md](ROADMAP.md) §1 including four
numbers we had asserted before measuring, and later corrected.

**"Did you test whether your own ideas work?"**
Yes, and one failed. The ablation study shows Phantom Attention — one of our
headline algorithms — contributes ~0.1 points and no measurable quality. We
report it and recommend deleting the module. See
[BENCHMARKS.md](BENCHMARKS.md) §6.

---

## 4. Pre-flight checklist

```bash
make test                       # 280 tests
python examples/quickstart.py   # 7 worked examples, no network
docker compose up               # local, http://localhost:8000
```

- [ ] Render URL opens and shows the console
- [ ] Warmed it within the last 10 minutes
- [ ] Localhost running in a second tab as backup
- [ ] `/docs` loads (proves it's a real API)
- [ ] You can state the 87% / 97.5% / +69pts numbers without notes
