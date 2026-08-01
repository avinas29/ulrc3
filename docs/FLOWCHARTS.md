# Flowcharts

Control and data flow at four levels of zoom.

---

## 1. Request lifecycle

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant E as Compressor
    participant B as IR builder
    participant P as Pass manager
    participant V as Verifier
    participant S as Residual store

    C->>E: Request(system, tools, instruction, query, documents)
    E->>E: resolve config · count input tokens · compute budget
    E->>B: build IR
    B->>B: structural oracles (ast.parse / json.loads)
    B->>B: Viterbi region segmentation
    B->>B: dispatch each region to its pipeline
    B->>B: within-context IDF · symbols · edges
    B->>B: seed protection · propagate fixpoint
    B-->>E: CIR (units, ladders, obligations, lattice, edges)

    E->>P: prep passes (levels, dedup, attention, salience)
    loop adaptive control (≤3 rounds)
        E->>P: select · closure · order
        E->>V: render · verify · confidence
        loop repair (≤4 rounds)
            V-->>V: missing obligation → promote rung or add carrier
            V->>V: re-render · re-audit
        end
        alt confidence < target
            E->>E: budget ← 1.35·budget + 32
        else
            E->>E: exit loop
        end
    end
    E->>E: inflation guard (fallback to source if violated)
    E->>S: store dropped spans under handles
    E-->>C: text · verification · confidence · stats · residuals
    C->>S: (optional) expand(handles) — the recovery layer
```

---

## 2. Front-end dispatch

```mermaid
flowchart TD
    A[document] --> B{role is<br/>system/tools/<br/>instruction/query?}
    B -->|yes| Z[FROZEN units<br/>verbatim, unreorderable]
    B -->|no| C{json.loads<br/>succeeds?}
    C -->|yes| J[json pipeline]
    C -->|no| D{≥2 def/class/import<br/>AND ast.parse succeeds?}
    D -->|yes| K[code pipeline<br/>whole file, no splitting]
    D -->|no| E[per-line feature scoring<br/>40 features × 11 classes]
    E --> F[Viterbi decode<br/>with switch penalty]
    F --> G[typed regions]
    G --> H1[prose / markdown]
    G --> H2[code]
    G --> H3[conversation]
    G --> H4[logs]
    G --> H5[json / yaml]
    G --> H6[table / sql]
    G --> H7[legal]
    G --> H8[apidocs]
    H2 --> I{symbols found<br/>or code-shaped?}
    I -->|no| H1
    I -->|yes| K
```

The self-demotion at the bottom matters: a code front-end that finds no code has
misclaimed the region, and prose handling beats an opaque blob.

---

## 3. CASCADE selection

```mermaid
flowchart TD
    A[all units at protection floor] --> B{floor ≥ budget?}
    B -->|yes| Z[return: protection outranks budget<br/>report floor_tokens]
    B -->|no| C[water-fill remaining budget<br/>across documents by value density]
    C --> D[per group: build max-heap<br/>keyed by ΔF/Δc of next rung]
    D --> E{heap non-empty<br/>and budget left?}
    E -->|no| K
    E -->|yes| F[pop best candidate]
    F --> G{unit moved<br/>since queued?}
    G -->|yes| E
    G -->|no| H[recompute ΔF/Δc now]
    H --> I{affordable?}
    I -->|no| I2[record as singleton candidate] --> E
    I -->|yes| J{ratio dropped<br/>materially? CELF}
    J -->|yes| J2[requeue with new ratio] --> E
    J -->|no| L{ratio < θ·best?<br/>marginal-utility stop}
    L -->|yes| K
    L -->|no| M[upgrade rung<br/>update coverage<br/>queue next rung] --> E
    K[greedy done] --> N{best singleton<br/>beats greedy total?}
    N -->|yes| O[reset group to floor<br/>take the singleton]
    N -->|no| P[keep greedy solution]
    O --> Q[spillover round with leftover budget]
    P --> Q
```

---

## 4. Verification and repair

```mermaid
flowchart TD
    A[rendered output] --> B[provenance check<br/>words ⊆ source ∪ markers ∪ aliases]
    B --> C[obligation audit<br/>same extractor on src and out]
    C --> D{integrity<br/>violations?}
    D -->|yes| E[repair: promote rung in place]
    D -->|no| F{critical<br/>violations?}
    F -->|yes| E
    F -->|no| G[inflation check]
    E --> E2{rung carries<br/>the obligation?}
    E2 -->|yes| E3[raise level · spend Δ tokens]
    E2 -->|no| E4[merge enclosing sentences<br/>into one FACT carrier]
    E4 --> E5{code unit and<br/>fragment not prose?}
    E5 -->|yes| E6[skip — never fake] --> H
    E5 -->|no| E7[emit carrier unit]
    E3 --> H{spent ≥ cap<br/>max 48, 35% of budget?}
    E7 --> H
    H -->|no| C
    H -->|yes| G
    G --> I{tokens_out<br/>≥ tokens_in?}
    I -->|yes| J[fallback to source<br/>report inflation]
    I -->|no| K[syntax check: ast.parse / json.loads]
    K --> L[frozen fidelity: substring containment]
    L --> M[confidence estimator]
    M --> N{p < min_confidence<br/>and rounds left?}
    N -->|yes| O[raise budget · re-select]
    N -->|no| P[return result]
```

---

## 5. The fidelity ladder as a decision space

```mermaid
graph LR
    subgraph KD["keep-or-drop (LLMLingua family)"]
        A1[function] -->|keep| A2["184 tok<br/>full body"]
        A1 -->|drop| A3["0 tok<br/>symbol gone"]
    end
    subgraph FL["fidelity ladder (ULRC³)"]
        B1[function] -->|drop| B2["0 tok"]
        B1 -->|sig| B3["11 tok<br/>contract preserved"]
        B1 -->|stub| B4["18 tok<br/>+ doc line"]
        B1 -->|full| B5["184 tok"]
    end
```

At a 20-token budget the left graph must choose between blowing the budget by
9× and losing the symbol. The right graph buys the contract for 11.

Measured effect of collapsing the right graph into the left (`no_ladder`
ablation): **−4.6 points of compression** at equal quality across all suites,
and **51.8% → 0%** on a Python module, where every protected definition must
then be emitted in full.
