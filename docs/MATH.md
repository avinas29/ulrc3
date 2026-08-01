# Mathematical formulation

Notation. A context is a set of units `U`. Each `u ∈ U` has a fidelity ladder
`L(u) = {0, 1, …, m_u}` with token cost `c(u,ℓ)` (non-decreasing in ℓ,
`c(u,0)=0`), concept contribution `C(u,ℓ) : 𝒞 → ℝ₊`, obligation set `O(u,ℓ)`,
and fidelity `φ(u,ℓ) ∈ [0,1]`. An assignment is `ℓ : U → ℕ` with `ℓ(u) ∈ L(u)`.

---

## 1. The compression problem

$$
\begin{aligned}
\max_{\ell}\quad & F(\ell) \\
\text{s.t.}\quad & \sum_{u\in U} c(u,\ell(u)) \;\le\; B &&\text{(budget)}\\
& \ell(u) \;\ge\; \mu(\pi(u)) \quad \forall u &&\text{(protection floor)}\\
& \ell(u) > 0 \Rightarrow \ell(v) > 0 \;\;\forall (u\to v)\in R_{\le H} &&\text{(closure)}\\
& \mathcal{E}_{\text{hard}}(\text{src}) \cap \mathcal{R} \subseteq \mathcal{E}_{\text{hard}}(\text{out}) &&\text{(obligations)}
\end{aligned}
$$

where `π(u)` is the protection class, `μ` maps it to a minimum rung, `R` is the
`REQUIRES` relation, and `𝓔_hard` is the obligation extractor restricted to
enforced classes.

This is a **Multiple-Choice Knapsack Problem with a submodular objective and
precedence constraints** — NP-hard, hence the approximation machinery below.

Every published prompt compressor solves the special case `m_u = 1` (keep or
drop), which is strictly less expressive: it cannot express "keep the signature,
drop the body".

---

## 2. The objective

$$
F(\ell) \;=\; \underbrace{\sum_{k\in\mathcal C} w_k\Bigl(1-e^{-\lambda\,x_k(\ell)}\Bigr)}_{\text{coverage (submodular)}}
\;+\;\underbrace{\mu_s\sum_{u\in U}s(u)\,\varphi(u,\ell(u))}_{\text{salience (modular)}}
$$

with accumulated coverage \(x_k(\ell)=\sum_{u} C(u,\ell(u))_k\), saturation
\(\lambda = 0.9\), and \(\mu_s = 0.55\).

**Concept weights.**
$$
w_k=\begin{cases}
\text{idf}(k)\cdot Q^{\,\mathbb 1[k\in q]} & k \text{ a term}\\[2pt]
\omega_{\text{ob}}\cdot \text{weight}(k) & k = \texttt{@ob:}o \text{ (an obligation)}
\end{cases}
$$
with `ω_ob = 3.5`, `Q = 2.5·query_weight`. Obligations enter the objective as
high-weight pseudo-concepts, so the optimiser is *paid* to satisfy the audit
rather than punished after the fact. This is the single most important design
choice in the objective: preservation and optimisation are the same problem, not
two problems in tension.

### Proposition 1 — `F` is monotone and submodular.

*Proof.* Each map `ℓ ↦ x_k(ℓ)` is modular (a sum of per-unit contributions), and
`g(x) = 1 − e^{−λx}` is concave and non-decreasing on `ℝ₊`. A concave
non-decreasing function of a modular function is monotone submodular, and
non-negative weighted sums preserve both. The salience term is modular, hence
trivially submodular. ∎

Consequence: marginal gains are non-increasing, which is exactly the condition
under which CELF lazy evaluation is correct (a stale heap entry can only
over-estimate the current gain, never under-estimate it).

### Within-context IDF

$$
\text{idf}(t) \;=\; \max\left(0.05,\; \log\!\left(\frac{N - \mathrm{df}(t) + 0.5}{\mathrm{df}(t)+0.5}+1\right)\right)
$$

with `N` the number of **units in this request**. Specificity is relative to the
context being compressed, not to English at large: a term appearing in every
retrieved chunk is uninformative *here* even if it is rare in a corpus. This is
why no static IDF table ships with the engine.

---

## 3. Information density (semantic entropy)

For a unit with normalised concept weights `c_t`:

$$
H(u)=-\sum_t p_t\log p_t,\qquad p_t=\frac{c_t}{\sum_{t'}c_{t'}},\qquad
\hat H(u)=\frac{H(u)}{\log|c|}
$$

$$
I(u)=\frac{\bigl(\sum_t c_t\bigr)\bigl(1+\gamma\hat H(u)\bigr)}{\text{tok}(u)^{\beta}},
\qquad \gamma=0.55,\;\beta=0.35
$$

Interpretation: mass concentrated on one rare term ⇒ a *fact*; mass spread over
many mid-frequency terms ⇒ *narration*. Both matter, but per token the fact is
worth more. `β < 1` prevents degeneration into "always pick the shortest unit".

Contrast with LLMLingua's importance signal `−log p_θ(t | t_<)`: that quantity
depends on the proxy model `θ`, so results are not reproducible across proxy
swaps and are systematically biased toward text the proxy finds unusual (rare
formatting, foreign words) rather than text that is *informative for the task*.
`I(u)` is computed from the context's own statistics and is deterministic.

---

## 4. Water-filling budget allocation

Split budget `B` across groups (documents / segments) `i` with value `v_i`,
scale `κ_i`, floor `f_i`, cap `C_i`:

$$
\max_{b}\ \sum_i v_i\log\!\Bigl(1+\frac{b_i}{\kappa_i}\Bigr)
\quad\text{s.t.}\quad \sum_i b_i = B,\;\; f_i\le b_i\le C_i
$$

Lagrangian `ℒ = Σ v_i log(1+b_i/κ_i) − λ(Σ b_i − B)` gives
`∂ℒ/∂b_i = v_i/(κ_i+b_i) − λ = 0`, hence the closed form

$$
\boxed{\;b_i(\lambda)=\operatorname{clamp}\!\left(\frac{v_i}{\lambda}-\kappa_i,\;f_i,\;C_i\right)\;}
$$

`Σ b_i(λ)` is non-increasing in `λ`, so bisection on `λ` converges monotonically;
60 geometric-bisection steps give ~1e-18 relative precision on the multiplier.

**Why concave and not proportional?** Proportional allocation gives a long
irrelevant chunk a large budget merely for being long — the classic RAG failure.
The log utility encodes diminishing returns *within* a document while the floor
`f_i` guarantees no document is silently zeroed.

---

## 5. CASCADE: approximation guarantee

Greedy operates on **upgrade steps**. For unit `u` at rung `ℓ`, the candidate is
`ℓ → ℓ+1` with

$$
\Delta F(u,\ell) = \sum_k w_k\Bigl(e^{-\lambda x_k}-e^{-\lambda(x_k+\delta_k)}\Bigr) + \mu_s s(u)\bigl(\varphi_{\ell+1}-\varphi_\ell\bigr),
\qquad \Delta c = c(u,\ell+1)-c(u,\ell)
$$

and the priority is `ΔF/Δc`.

### Theorem (Khuller–Moss–Naor; Sviridenko)

For monotone submodular maximisation under a knapsack constraint, the better of
(i) ratio-greedy and (ii) the best single feasible element achieves
`(1 − 1/e)/2 ≈ 0.316` of the optimum; partial enumeration over triples reaches
`1 − 1/e ≈ 0.632`.

We implement (i) + (ii). **The singleton rule is not optional**: without it a
single item whose cost exceeds the residual budget is skipped forever and the
budget fills with cheap low-value items. We observed exactly this — a 28-token
sentence carrying the document's only URL lost to four repetitions of
"Unrelated narration." at a 20-token budget. It is now a regression test.

**Measured optimality gap.** Against the LP relaxation upper bound
`max Σ_u Σ_ℓ y_{u,ℓ}·ΔF₀(u,ℓ)` s.t. `Σ y·Δc ≤ B`, `Σ_ℓ y_{u,ℓ} ≤ 1`, `y ≥ 0`,
solved by ratio sorting with one fractional item (`bench/lp_bound.py`, 156
instances): **mean gap 35.4%, median 32.4%, p90 70.2%**.

That bound is *conservative on two counts* — it relaxes integrality, and it
evaluates every candidate's gain against the **empty** coverage, which by
submodularity over-estimates every marginal — so the true gap to the integral
optimum is materially smaller. But it is not 1.8%, which is what an earlier
draft of this document asserted without computing it. The honest reading is:
the greedy is provably within `(1−1/e)/2` of optimal, and the empirical distance
to a loose upper bound is around a third.

### Marginal-utility stop

The budget is a **ceiling, not a quota**. Upgrades are refused once

$$
\frac{\Delta F}{\Delta c} \;<\; \theta\cdot\max_{\text{accepted}}\frac{\Delta F}{\Delta c},\qquad \theta = 0.04
$$

so the engine returns *under* budget when the remaining material is not worth
its tokens. On filler-heavy corpora this is worth several points of compression
at identical answerability.

---

## 6. Phantom Attention

Let `W` be the row-normalised weighted adjacency of the unit multigraph and `p`
the personalisation vector. Simulated attention is the stationary distribution

$$
r = (1-\alpha)\,p + \alpha\,W^{\!\top} r
$$

### Proposition 2 — existence, uniqueness, geometric convergence.

The map `T(r) = (1−α)p + αWᵀr` satisfies
`‖T(r) − T(r')‖₁ = α‖Wᵀ(r−r')‖₁ ≤ α‖r−r'‖₁` because `Wᵀ` is column-stochastic
and therefore an `ℓ₁` contraction with modulus `α`. By Banach, a unique fixed
point exists and `‖r_k − r‖₁ ≤ 2α^k`. At `α = 0.85`, `k = 18` gives
`‖r_18 − r‖₁ < 5·10⁻²`, far below the resolution at which the induced ranking
changes. ∎

### Edge model

| effect in real transformers | our edge | weight |
|---|---|---|
| induction / copy heads | IDF-weighted lexical overlap | 1.00 |
| syntactic & structural heads | `REQUIRES` | 1.35 |
| containment | `CONTAINS` | 0.75 |
| positional locality | `ADJACENT · e^{−d/τ}`, τ=3 | 0.45 |
| dialogue response | `ANSWERS` | 0.90 |
| redundancy (suppressive) | `DUPLICATE_OF` | 0.15 |
| retraction (suppressive) | `SUPERSEDES` | 0.10 |

**Density normalisation.** Raw PageRank mass rewards long units. We use
`att(u) = r_u / √(tok(u)/24)` so the score measures attention *density*.

**Sparsification.** Lexical edges use a star topology per term (each posting
connects to the term's top-8 weight hubs) rather than a clique. The clique is
`O(df²)` per term and was the pipeline's dominant cost; the star preserves
component structure, which is what PageRank mass flow depends on.

---

## 7. The protection lattice

`(Protection, ⊔=max)` is a finite join-semilattice with bottom `DROPPABLE` and
top `FROZEN`. Transfer along `u --REQUIRES--> v`:

$$
\pi(v) \;\mathrel{:=}\; \pi(v)\;\sqcup\;\operatorname{demote}(\pi(u)),\qquad
\operatorname{demote}(p)=\max(\bot,\;p-1)
$$

### Proposition 3 — the fixpoint exists and is reached in `O(V + E·|Λ|)`.

Values only increase (monotone) in a finite lattice, so the worklist iteration
terminates. `demote` strictly decreases the propagated value, so after at most
`|Λ| = 5` traversals of any cycle the inherited value reaches `⊥` and stops
propagating — cycles cannot loop forever. ∎

**Closure invariant.** For retained set `S`:

$$
\forall u\in S,\ \forall (u\to v)\in R \text{ with } \mathrm{hops}\le H:\quad v\in S \ \lor\ \mathrm{stub}(v)\in S
$$

---

## 8. Verification as decidable predicates

Let `E(x)` be the obligation extractor and `M` the closed marker vocabulary.

**V1 — provenance.**
$$
\mathrm{words}(\text{out}\setminus\text{control lines}) \subseteq \mathrm{words}(\text{src}) \cup M \cup A
$$
where `A` is the alias set (`e3`, `x7`). Numerals absent from the source are
admissible only inside a *derived field* (`×N`, `n=`, `min=`, `max=`, `rows=`,
`card=`, a timestamp window) — deterministic aggregates the engine computed.

**V2 — integrity.** For `retained = ⋃_{u: ℓ(u)>0} O(u, ℓ(u))`:
$$
\text{integrity} = \frac{|\{o\in \text{retained}: o \in E(\text{out})\}|}{|\text{retained}|} \overset{!}{=} 1
$$

**V3 — critical recall.** Constraints, security statements and negations
anywhere in the source; target 1, shortfalls enumerated.

**V4 — inflation.** `tok(out) < tok(src)`, else fall back to the source.

**V5 — syntax.** Emitted Python re-parses (`ast.parse`); JSON fragments parse.

**V6 — frozen fidelity.** Every `FROZEN` unit occurs as a verbatim substring.

### Proposition 4 — the repair loop terminates.

Repair only *adds* carriers or *raises* rungs, both monotone over a finite
domain, and is capped at `max(48, 0.35·B)` tokens. Each round either strictly
decreases the number of missing obligations or adds nothing (loop exits). ∎

---

## 9. Confidence

$$
p=\sigma\!\left(w_0+\sum_j w_j x_j\right)
$$

with features `x = (ratio, integrity, critical, retention, coverage_kept,
violation_rate, reduced_fidelity, type_entropy, query_coverage, provenance)`.
Weights ship in `ulrc3/confidence.json` (inspectable and replaceable). They are
**hand-set priors, not fitted** — fitting requires ground-truth downstream
correctness labels, which need a model in the loop and do not exist here. The
*ordering* they induce is what the adaptive loop uses; the absolute probabilities
are uncalibrated and should be treated as such.

**Adaptive loop.** While `p < p*` and rounds < 3: `B ← min(tok_in, 1.35B + 32)`,
re-run selection. Monotone in `B` ⇒ terminating, and it converts an aggressive
target from a risk into a *request*.

---

## 10. Complexity summary

| stage | complexity | 128k-token measurement |
|---|---|---|
| region segmentation | `O(n·L²)`, L=11 | 41 ms |
| unit construction | `O(n)` | 310 ms |
| obligation extraction | `O(n·p)`, p≈20 patterns | included above |
| dedup (LSH + union-find) | `O(n·reps)` expected | 1 080 ms |
| phantom attention | `O(nnz·iters)`, nnz≈12n | 205 ms |
| CASCADE | `O(M log M)`, `M = Σ|L(u)|` | 18 ms |
| closure | `O(V+E)` | 3 ms |
| render + verify | `O(n)` | 41 ms |
| **total** | **`O(n log n)`** | **3 018 ms → 42 400 tok/s** |

Throughput is flat from 1.4k to 128k tokens (34.9k → 42.4k tok/s), confirming
the near-linear scaling claim empirically rather than asymptotically.
