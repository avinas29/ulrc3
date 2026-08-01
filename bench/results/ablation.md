# Ablation study

All suites, balanced mode, n=156 instances per row.
Δ columns are relative to the full system.

| ablation | ratio % | Δratio | answerability % | Δans | contradiction % | identifiers % | Δident | integrity % | latency ms |
|---|---|---|---|---|---|---|---|---|---|
| full system | 74.8 | +0.0 | 99.8 | +0.0 | 0.0 | 99.4 | +0.0 | 100.0 | 44.2 |
| no_ladder | 68.0 | -6.8 | 100.0 | +0.2 | 0.0 | 100.0 | +0.6 | 100.0 | 45.0 |
| no_attention | 73.8 | -1.0 | 99.8 | +0.0 | 0.0 | 99.4 | +0.0 | 100.0 | 40.6 |
| no_lexical | 74.9 | +0.1 | 99.8 | +0.0 | 0.0 | 99.4 | +0.0 | 100.0 | 42.5 |
| no_dedup | 73.5 | -1.4 | 100.0 | +0.2 | 0.0 | 99.4 | +0.0 | 100.0 | 34.0 |
| no_closure | 75.2 | +0.3 | 99.8 | +0.0 | 0.0 | 99.4 | +0.0 | 100.0 | 43.8 |
| no_order | 74.9 | +0.0 | 99.8 | +0.0 | 0.0 | 99.4 | +0.0 | 100.0 | 44.2 |
| no_repair | 77.5 | +2.6 | 88.8 | -11.0 | 0.0 | 98.3 | -1.1 | 100.0 | 43.2 |

## Reading the table

A mechanism earns its place if removing it *either* costs compression at equal quality *or* costs quality at equal compression. A row that matches the full system on both is a mechanism we should delete.
